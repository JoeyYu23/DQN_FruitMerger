#!/bin/bash
# ========================================
# 快速上传到云服务器脚本
# ========================================
# 使用方法:
#   ./upload_to_server.sh <服务器IP> <用户名>
#   例如: ./upload_to_server.sh 192.168.1.100 root
# ========================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -lt 2 ]; then
    echo "使用方法: $0 <服务器IP> <用户名> [远程路径]"
    echo "例如: $0 192.168.1.100 root /root"
    exit 1
fi

SERVER_IP=$1
SERVER_USER=$2
REMOTE_PATH=${3:-/root}

print_info "=========================================="
print_info "上传 DQN_FruitMerger 到云服务器"
print_info "=========================================="
echo "服务器IP: $SERVER_IP"
echo "用户名: $SERVER_USER"
echo "远程路径: $REMOTE_PATH"
echo ""

# 检查SSH连接
print_info "测试SSH连接..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes ${SERVER_USER}@${SERVER_IP} exit 2>/dev/null; then
    print_info "✓ SSH连接成功"
else
    print_warn "SSH连接测试失败，可能需要输入密码"
fi
echo ""

# 创建临时目录
TEMP_DIR=$(mktemp -d)
print_info "创建临时目录: $TEMP_DIR"

# 打包项目
print_info "打包项目文件..."
ARCHIVE_NAME="DQN_FruitMerger_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$TEMP_DIR/$ARCHIVE_NAME" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='*.pth' \
    --exclude='*.pdparams' \
    --exclude='weights/*' \
    --exclude='videos/*' \
    --exclude='output/*' \
    --exclude='.DS_Store' \
    -C "$(dirname "$0")" \
    $(basename "$0" .sh | xargs dirname | xargs basename)

ARCHIVE_SIZE=$(du -h "$TEMP_DIR/$ARCHIVE_NAME" | cut -f1)
print_info "✓ 打包完成: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
echo ""

# 上传到服务器
print_info "上传到服务器..."
scp "$TEMP_DIR/$ARCHIVE_NAME" ${SERVER_USER}@${SERVER_IP}:${REMOTE_PATH}/

if [ $? -eq 0 ]; then
    print_info "✓ 上传成功"
else
    print_error "上传失败"
    rm -rf "$TEMP_DIR"
    exit 1
fi
echo ""

# 在服务器上解压
print_info "在服务器上解压..."
ssh ${SERVER_USER}@${SERVER_IP} << ENDSSH
cd ${REMOTE_PATH}
echo "解压文件..."
tar -xzf $ARCHIVE_NAME
echo "✓ 解压完成"

echo ""
echo "设置权限..."
cd DQN_FruitMerger
chmod +x deploy_server.sh
chmod +x train_cloud.sh 2>/dev/null || true
echo "✓ 权限设置完成"

echo ""
echo "清理压缩包..."
cd ${REMOTE_PATH}
rm -f $ARCHIVE_NAME
echo "✓ 清理完成"
ENDSSH

if [ $? -eq 0 ]; then
    print_info "✓ 服务器端配置完成"
else
    print_error "服务器端配置失败"
fi
echo ""

# 清理本地临时文件
rm -rf "$TEMP_DIR"
print_info "✓ 清理临时文件"
echo ""

print_info "=========================================="
print_info "上传完成！"
print_info "=========================================="
echo ""
echo "下一步操作:"
echo ""
echo "1. SSH登录到服务器:"
echo "   ssh ${SERVER_USER}@${SERVER_IP}"
echo ""
echo "2. 进入项目目录:"
echo "   cd ${REMOTE_PATH}/DQN_FruitMerger"
echo ""
echo "3. 查看部署指南:"
echo "   cat SERVER_DEPLOYMENT_GUIDE.md"
echo ""
echo "4. 运行自动部署脚本:"
echo "   ./deploy_server.sh"
echo ""
echo "5. 或者手动安装依赖:"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118"
echo "   pip install -r requirements_server.txt"
echo ""
print_info "=========================================="

# 询问是否立即SSH登录
echo ""
read -p "是否立即SSH登录到服务器? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ssh ${SERVER_USER}@${SERVER_IP}
fi
