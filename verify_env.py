#!/usr/bin/env python3
"""
验证环境配置是否正确
检查所有必需的包和版本
"""

import sys
from typing import Dict, Tuple


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 8:
        return True, f"✓ Python {version_str}"
    else:
        return False, f"✗ Python {version_str} (需要 >= 3.8)"


def check_package(package_name: str, expected_version: str = None) -> Tuple[bool, str]:
    """检查包是否安装及版本"""
    try:
        if package_name == 'opencv-python':
            import cv2 as module
            actual_name = 'cv2'
        elif package_name == 'paddlepaddle':
            import paddle as module
            actual_name = 'paddle'
        else:
            module = __import__(package_name)
            actual_name = package_name

        version = getattr(module, '__version__', 'unknown')

        if expected_version:
            if version == expected_version:
                return True, f"✓ {package_name}=={version}"
            else:
                return True, f"⚠ {package_name}=={version} (期望: {expected_version})"
        else:
            return True, f"✓ {package_name}=={version}"
    except ImportError:
        return False, f"✗ {package_name} 未安装"


def verify_environment():
    """验证整个环境"""
    print("\n" + "="*70)
    print("  环境验证")
    print("="*70)
    print()

    all_ok = True

    # 检查Python版本
    ok, msg = check_python_version()
    print(f"Python版本: {msg}")
    all_ok = all_ok and ok

    print("\n核心依赖:")
    print("-" * 70)

    # 关键包及期望版本
    packages = {
        'paddlepaddle': '3.2.1',
        'numpy': '1.26.4',
        'opencv-python': '4.11.0.86',
        'pymunk': '6.5.0',
    }

    for pkg, expected_ver in packages.items():
        ok, msg = check_package(pkg, expected_ver)
        print(msg)
        all_ok = all_ok and ok

    print("\n可选依赖:")
    print("-" * 70)

    optional_packages = {
        'matplotlib': '3.7.2',
        'tqdm': '4.67.1',
        'psutil': '5.9.0',
    }

    for pkg, expected_ver in optional_packages.items():
        ok, msg = check_package(pkg, expected_ver)
        print(msg)

    # 功能测试
    print("\n功能测试:")
    print("-" * 70)

    tests = []

    # 测试PaddlePaddle
    try:
        import paddle
        paddle.set_device('cpu')
        x = paddle.randn([2, 3])
        tests.append(("✓ PaddlePaddle CPU", True))

        # 检查GPU
        if paddle.is_compiled_with_cuda():
            tests.append(("✓ PaddlePaddle GPU支持", True))
        else:
            tests.append(("⚠ PaddlePaddle GPU不可用 (CPU版本)", True))
    except Exception as e:
        tests.append((f"✗ PaddlePaddle测试失败: {e}", False))
        all_ok = False

    # 测试Pymunk
    try:
        import pymunk
        space = pymunk.Space()
        tests.append(("✓ Pymunk物理引擎", True))
    except Exception as e:
        tests.append((f"✗ Pymunk测试失败: {e}", False))
        all_ok = False

    # 测试OpenCV
    try:
        import cv2
        import numpy as np
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _ = cv2.resize(img, (50, 50))
        tests.append(("✓ OpenCV图像处理", True))
    except Exception as e:
        tests.append((f"✗ OpenCV测试失败: {e}", False))
        all_ok = False

    # 测试项目核心模块
    try:
        from Game import GameCore
        from GameInterface import GameInterface
        tests.append(("✓ 游戏核心模块", True))
    except Exception as e:
        tests.append((f"✗ 游戏模块加载失败: {e}", False))
        all_ok = False

    try:
        from SuikaNet import SuikaNet
        from AlphaZeroMCTS import AlphaZeroMCTS
        tests.append(("✓ AlphaZero模块", True))
    except Exception as e:
        tests.append((f"✗ AlphaZero模块加载失败: {e}", False))
        all_ok = False

    for msg, ok in tests:
        print(msg)
        if not ok:
            all_ok = False

    # 总结
    print("\n" + "="*70)
    if all_ok:
        print("  ✅ 环境验证通过！可以开始训练")
    else:
        print("  ❌ 环境验证失败，请检查上述错误")
    print("="*70)
    print()

    return all_ok


if __name__ == '__main__':
    success = verify_environment()
    sys.exit(0 if success else 1)
