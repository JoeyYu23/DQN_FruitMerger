#!/bin/bash
# ========================================
# 一键推送到GitHub脚本
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

# 确保在正确的目录
cd "$(dirname "$0")"

print_info "=========================================="
print_info "推送 DQN_FruitMerger 到 GitHub"
print_info "=========================================="
echo ""

# 步骤1: 检查Git状态
print_step "1/5 检查Git状态..."
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "当前目录不是Git仓库"
    exit 1
fi

# 显示当前分支
CURRENT_BRANCH=$(git branch --show-current)
print_info "当前分支: $CURRENT_BRANCH"

# 显示远程仓库
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "未设置")
print_info "远程仓库: $REMOTE_URL"
echo ""

# 步骤2: 显示变更
print_step "2/5 检查变更文件..."
git status --short

CHANGED_FILES=$(git status --short | wc -l | tr -d ' ')
if [ "$CHANGED_FILES" -eq "0" ]; then
    print_warn "没有变更文件"
    read -p "是否继续推送? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi
else
    print_info "发现 $CHANGED_FILES 个变更文件"
fi
echo ""

# 步骤3: 添加文件
print_step "3/5 添加文件到暂存区..."

# 询问是否添加所有文件
read -p "是否添加所有文件? (y/n, 默认y): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_info "请手动添加文件: git add <file>"
    exit 0
else
    git add .
    print_info "✓ 已添加所有文件"
fi
echo ""

# 步骤4: 提交
print_step "4/5 创建提交..."

# 询问提交信息
echo "请输入提交信息 (留空使用默认信息):"
read -r COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    # 默认提交信息（带时间戳）
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    COMMIT_MSG="Update deployment scripts and guides - $TIMESTAMP

Changes:
- Update deployment configurations
- Add/update cloud deployment guides
- Optimize for PyTorch 2.1.2 + CUDA 11.8
- Server config: RTX 3080 Ti (12GB)"
fi

git commit -m "$COMMIT_MSG"

if [ $? -eq 0 ]; then
    print_info "✓ 提交成功"
else
    print_warn "没有需要提交的更改（可能已经提交过）"
fi
echo ""

# 步骤5: 推送
print_step "5/5 推送到GitHub..."

# 检查是否设置了upstream
if git rev-parse --abbrev-ref @{u} > /dev/null 2>&1; then
    git push
else
    print_warn "未设置upstream，使用 git push -u origin $CURRENT_BRANCH"
    git push -u origin "$CURRENT_BRANCH"
fi

if [ $? -eq 0 ]; then
    print_info "✓ 推送成功！"
else
    print_error "推送失败"
    exit 1
fi
echo ""

# 完成
print_info "=========================================="
print_info "✅ 推送完成！"
print_info "=========================================="
echo ""
echo "📍 GitHub仓库:"
echo "   $REMOTE_URL"
echo ""
echo "🚀 在AutoDL上部署:"
echo "   1. SSH登录AutoDL"
echo "   2. git clone $REMOTE_URL"
echo "   3. cd DQN_FruitMerger"
echo "   4. ./deploy_server.sh"
echo ""
print_info "=========================================="
