#!/usr/bin/env python3
"""
导出当前环境配置
生成多个版本的requirements文件供不同场景使用
"""

import subprocess
import sys
from datetime import datetime


def get_package_version(package_name):
    """获取包的版本"""
    try:
        result = subprocess.run(
            ['pip', 'show', package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        return None


def export_exact_versions():
    """导出精确版本"""
    print("🔍 检测本地环境版本...")
    print("="*70)

    # 关键包列表
    packages = [
        'paddlepaddle',
        'numpy',
        'opencv-python',
        'pymunk',
        'matplotlib',
        'tqdm',
        'psutil'
    ]

    versions = {}
    for pkg in packages:
        version = get_package_version(pkg)
        if version:
            versions[pkg] = version
            print(f"✓ {pkg:25} {version}")
        else:
            print(f"✗ {pkg:25} 未安装")

    print("="*70)

    # 获取Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    # 生成requirements_exact.txt
    content = f"""# 精确版本要求 - 与本地环境完全一致
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Python版本: {python_version}

# 核心依赖 (精确版本)
"""

    for pkg in ['paddlepaddle', 'numpy', 'opencv-python', 'pymunk']:
        if pkg in versions:
            content += f"{pkg}=={versions[pkg]}\n"

    content += "\n# 可视化和工具\n"
    for pkg in ['matplotlib', 'tqdm', 'psutil']:
        if pkg in versions:
            content += f"{pkg}=={versions[pkg]}\n"

    content += """
# 说明：
# - 云端如果使用GPU，将paddlepaddle改为paddlepaddle-gpu=={版本号}
# - 如遇兼容性问题，可参考requirements_flexible.txt
"""

    with open('requirements_exact.txt', 'w') as f:
        f.write(content)

    print("\n✅ 已生成: requirements_exact.txt")

    # 生成requirements_flexible.txt (兼容版本范围)
    content_flex = f"""# 灵活版本要求 - 兼容范围
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Python版本: {python_version}+

# 核心依赖 (兼容版本范围)
paddlepaddle>=3.0.0,<4.0.0  # 或paddlepaddle-gpu
numpy>=1.20.0,<2.0.0
opencv-python>=4.5.0
pymunk>=6.2.0

# 可视化和工具
matplotlib>=3.3.0
tqdm>=4.60.0
psutil>=5.8.0

# 说明：
# - 此文件使用版本范围，兼容性更好但可能有细微差异
# - 推荐使用requirements_exact.txt确保完全一致
"""

    with open('requirements_flexible.txt', 'w') as f:
        f.write(content_flex)

    print("✅ 已生成: requirements_flexible.txt")

    # 生成完整的pip freeze
    print("\n📦 生成完整环境快照...")
    try:
        result = subprocess.run(
            ['pip', 'freeze'],
            capture_output=True,
            text=True
        )
        with open('requirements_full.txt', 'w') as f:
            f.write(f"# 完整环境快照 (pip freeze)\n")
            f.write(f"# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Python版本: {python_version}\n\n")
            f.write(result.stdout)
        print("✅ 已生成: requirements_full.txt")
    except:
        print("⚠️  无法生成requirements_full.txt")

    return versions


def generate_cloud_setup():
    """生成云端安装脚本"""
    content = """#!/bin/bash
# 云端环境配置脚本
# 使用方法: bash setup_cloud.sh [cpu|gpu]

set -e  # 遇到错误立即退出

MODE=${1:-cpu}  # 默认CPU模式

echo "======================================"
echo "  云端环境配置"
echo "======================================"
echo "模式: $MODE"
echo ""

# 检测Python版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

# 升级pip
echo "📦 升级pip..."
pip install --upgrade pip

# 安装依赖
if [ "$MODE" == "gpu" ]; then
    echo "🚀 安装GPU版本..."
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "检测到CUDA，安装paddlepaddle-gpu..."
        pip install paddlepaddle-gpu==3.2.1 -i https://mirror.baidu.com/pypi/simple
    else
        echo "⚠️  未检测到CUDA，将使用CPU版本"
        pip install paddlepaddle==3.2.1 -i https://mirror.baidu.com/pypi/simple
    fi
else
    echo "💻 安装CPU版本..."
    pip install paddlepaddle==3.2.1 -i https://mirror.baidu.com/pypi/simple
fi

echo "📦 安装其他依赖..."
pip install -r requirements_exact.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "✅ 环境配置完成！"
echo ""
echo "验证安装:"
python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import pymunk; print('Pymunk: OK')"

echo ""
echo "======================================"
echo "  可以开始训练了！"
echo "======================================"
"""

    with open('setup_cloud.sh', 'w') as f:
        f.write(content)

    # 设置执行权限
    import os
    os.chmod('setup_cloud.sh', 0o755)

    print("✅ 已生成: setup_cloud.sh")


def main():
    print("\n" + "="*70)
    print("  环境导出工具")
    print("="*70)
    print()

    # 导出版本信息
    versions = export_exact_versions()

    # 生成云端配置脚本
    print()
    generate_cloud_setup()

    print("\n" + "="*70)
    print("  📋 生成的文件:")
    print("="*70)
    print("  1. requirements_exact.txt    - 精确版本 (推荐)")
    print("  2. requirements_flexible.txt - 兼容版本范围")
    print("  3. requirements_full.txt     - 完整环境快照")
    print("  4. setup_cloud.sh           - 云端安装脚本")
    print("="*70)

    print("\n💡 使用方法:")
    print()
    print("  本地测试:")
    print("    pip install -r requirements_exact.txt")
    print()
    print("  云端部署:")
    print("    # 上传项目到云端后:")
    print("    bash setup_cloud.sh cpu   # CPU版本")
    print("    bash setup_cloud.sh gpu   # GPU版本")
    print()
    print("  版本验证:")
    print("    python verify_env.py")
    print()


if __name__ == '__main__':
    main()
