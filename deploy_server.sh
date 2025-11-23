#!/bin/bash
# ========================================
# DQN FruitMerger 云服务器自动部署脚本
# ========================================
# 服务器配置:
#   - GPU: RTX 3080 Ti (12GB) * 1
#   - CUDA: 11.8
#   - Python: 3.10
#   - PyTorch: 2.1.2
#   - 系统: Ubuntu 22.04
# ========================================

set -e  # 遇到错误立即退出

echo "======================================"
echo "DQN FruitMerger 云服务器部署开始"
echo "======================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 步骤1: 检查系统环境
print_info "步骤 1/6: 检查系统环境..."
echo "----------------------------------------"

# 检查Python版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Python版本: $PYTHON_VERSION"

# 检查CUDA版本
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_info "CUDA版本: $CUDA_VERSION"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    print_error "未检测到NVIDIA GPU或驱动未安装"
    exit 1
fi

# 检查磁盘空间
DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_info "可用磁盘空间: $DISK_SPACE"

echo ""

# 步骤2: 创建虚拟环境
print_info "步骤 2/6: 创建Python虚拟环境..."
echo "----------------------------------------"

if [ -d "venv" ]; then
    print_warn "虚拟环境已存在，跳过创建"
else
    python3 -m venv venv
    print_info "虚拟环境创建成功"
fi

# 激活虚拟环境
source venv/bin/activate
print_info "虚拟环境已激活"

echo ""

# 步骤3: 升级pip
print_info "步骤 3/6: 升级pip..."
echo "----------------------------------------"
pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

# 步骤4: 安装PyTorch (针对CUDA 11.8)
print_info "步骤 4/6: 安装PyTorch 2.1.2 (CUDA 11.8)..."
echo "----------------------------------------"

# 检查PyTorch是否已安装
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_warn "PyTorch已安装 (版本: $TORCH_VERSION)"
    read -p "是否重新安装? (y/n): " reinstall
    if [ "$reinstall" = "y" ]; then
        pip uninstall -y torch torchvision torchaudio
        pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
fi

# 验证PyTorch和CUDA
print_info "验证PyTorch安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"

if [ $? -eq 0 ]; then
    print_info "PyTorch安装验证成功 ✓"
else
    print_error "PyTorch安装验证失败"
    exit 1
fi

echo ""

# 步骤5: 安装其他依赖
print_info "步骤 5/6: 安装项目依赖..."
echo "----------------------------------------"

# 安装游戏引擎和工具
pip install pymunk>=6.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pygame>=2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install opencv-python>=4.5.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy>=1.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install tqdm>=4.62.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装可选的监控工具
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

print_info "依赖安装完成 ✓"
echo ""

# 步骤6: 测试环境
print_info "步骤 6/6: 测试环境..."
echo "----------------------------------------"

# 创建测试脚本
cat > test_env.py << 'EOF'
#!/usr/bin/env python3
"""环境测试脚本"""

def test_imports():
    """测试所有必要的包是否可以导入"""
    print("测试导入...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        import torchvision
        print(f"  ✓ torchvision {torchvision.__version__}")

        import pymunk
        print(f"  ✓ pymunk {pymunk.version}")

        import pygame
        print(f"  ✓ pygame {pygame.version.ver}")

        import cv2
        print(f"  ✓ opencv-python {cv2.__version__}")

        import numpy as np
        print(f"  ✓ numpy {np.__version__}")

        import tqdm
        print(f"  ✓ tqdm {tqdm.__version__}")

        return True
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        return False

def test_gpu():
    """测试GPU是否可用"""
    print("\n测试GPU...")

    import torch

    if not torch.cuda.is_available():
        print("  ✗ CUDA不可用")
        return False

    print(f"  ✓ CUDA可用")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 简单的GPU计算测试
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"  ✓ GPU计算测试通过")
        return True
    except Exception as e:
        print(f"  ✗ GPU计算失败: {e}")
        return False

def test_network():
    """测试网络架构"""
    print("\n测试网络架构...")

    try:
        import torch
        import torch.nn as nn

        # 简单的测试网络
        class TestNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.fc = nn.Linear(64 * 20 * 16, 16)

            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        net = TestNet()
        if torch.cuda.is_available():
            net = net.cuda()
            x = torch.randn(1, 13, 20, 16).cuda()
        else:
            x = torch.randn(1, 13, 20, 16)

        output = net(x)
        print(f"  ✓ 网络测试通过 (输出形状: {output.shape})")
        return True

    except Exception as e:
        print(f"  ✗ 网络测试失败: {e}")
        return False

if __name__ == '__main__':
    print("=" * 50)
    print("环境测试")
    print("=" * 50)

    all_passed = True
    all_passed &= test_imports()
    all_passed &= test_gpu()
    all_passed &= test_network()

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过！环境配置正确！")
        print("=" * 50)
        exit(0)
    else:
        print("✗ 部分测试失败，请检查配置")
        print("=" * 50)
        exit(1)
EOF

python test_env.py

if [ $? -eq 0 ]; then
    print_info "环境测试通过 ✓"
else
    print_error "环境测试失败"
    exit 1
fi

echo ""
echo "======================================"
echo "部署完成！"
echo "======================================"
echo ""
echo "下一步:"
echo "  1. 训练DQN模型:"
echo "     python DQN.py"
echo ""
echo "  2. 使用PyTorch版本 (开发中):"
echo "     python SuikaNet_torch.py"
echo ""
echo "  3. 启动TensorBoard监控 (端口6006):"
echo "     tensorboard --logdir=./logs --host=0.0.0.0 --port=6006"
echo ""
echo "  4. 退出虚拟环境:"
echo "     deactivate"
echo ""
echo "======================================"
