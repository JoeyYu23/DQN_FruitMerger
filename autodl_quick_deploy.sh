#!/bin/bash
# ========================================
# AutoDL 一键部署脚本
# ========================================
# 在AutoDL服务器上运行此脚本
# 会自动克隆/更新代码并部署环境
# ========================================

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# GitHub仓库地址
REPO_URL="https://github.com/JoeyYu23/DQN_FruitMerger.git"
PROJECT_DIR="DQN_FruitMerger"

print_info "=========================================="
print_info "AutoDL 一键部署 DQN FruitMerger"
print_info "=========================================="
echo ""
echo "仓库: $REPO_URL"
echo "目标: /root/$PROJECT_DIR"
echo ""

# 步骤1: 检查环境
print_step "1/4 检查环境..."
echo "Python: $(python3 --version 2>&1)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '未检测到GPU')"
echo "CUDA: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo '未知')"
echo ""

# 步骤2: 克隆或更新代码
print_step "2/4 获取代码..."
cd /root

if [ -d "$PROJECT_DIR" ]; then
    print_info "项目目录已存在，更新代码..."
    cd "$PROJECT_DIR"

    # 保存本地修改（如果有）
    if ! git diff-index --quiet HEAD --; then
        print_warn "检测到本地修改，正在保存..."
        git stash save "Auto stash before update $(date +%Y%m%d_%H%M%S)"
    fi

    # 拉取最新代码
    git pull origin main

    if [ $? -eq 0 ]; then
        print_info "✓ 代码更新成功"
    else
        print_error "代码更新失败"
        exit 1
    fi
else
    print_info "克隆仓库..."
    git clone "$REPO_URL"

    if [ $? -eq 0 ]; then
        print_info "✓ 克隆成功"
        cd "$PROJECT_DIR"
    else
        print_error "克隆失败"

        # 提供备用方案
        print_warn "尝试使用备用方法..."
        print_info "下载ZIP压缩包..."

        wget -O main.zip "https://github.com/JoeyYu23/DQN_FruitMerger/archive/refs/heads/main.zip"
        unzip -q main.zip
        mv DQN_FruitMerger-main "$PROJECT_DIR"
        cd "$PROJECT_DIR"

        if [ $? -eq 0 ]; then
            print_info "✓ 通过ZIP下载成功"
        else
            print_error "所有下载方式都失败"
            exit 1
        fi
    fi
fi
echo ""

# 步骤3: 设置权限
print_step "3/4 设置脚本权限..."
chmod +x deploy_server.sh 2>/dev/null || true
chmod +x upload_to_server.sh 2>/dev/null || true
chmod +x train_cloud.sh 2>/dev/null || true
print_info "✓ 权限设置完成"
echo ""

# 步骤4: 运行部署脚本
print_step "4/4 运行环境部署脚本..."
echo ""

if [ -f "deploy_server.sh" ]; then
    ./deploy_server.sh
else
    print_error "未找到 deploy_server.sh 脚本"
    print_info "请手动运行部署:"
    echo "  cd /root/$PROJECT_DIR"
    echo "  ./deploy_server.sh"
    exit 1
fi

echo ""
print_info "=========================================="
print_info "✅ 部署完成！"
print_info "=========================================="
echo ""
echo "下一步:"
echo ""
echo "1. 激活虚拟环境:"
echo "   source venv/bin/activate"
echo ""
echo "2. 安装PaddlePaddle (如需训练DQN):"
echo "   pip install paddlepaddle-gpu==2.5.1.post118 \\"
echo "     -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"
echo ""
echo "3. 开始训练:"
echo "   python DQN.py"
echo ""
echo "4. 后台训练 (推荐):"
echo "   tmux new -s train"
echo "   source venv/bin/activate"
echo "   python DQN.py"
echo "   # 按 Ctrl+B, D 分离会话"
echo ""
echo "5. TensorBoard监控:"
echo "   tensorboard --logdir=./logs --host=0.0.0.0 --port=6006 &"
echo ""
print_info "=========================================="
