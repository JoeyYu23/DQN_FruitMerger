"""
测试随机种子设置的可重复性
运行此脚本多次，验证输出是否完全相同
"""
import random
import numpy as np
import paddle
from DQN import set_global_seed

def test_reproducibility():
    """测试各个随机数生成器的可重复性"""

    # 设置种子
    SEED = 42
    set_global_seed(SEED)

    print("=" * 60)
    print("测试随机种子可重复性")
    print("=" * 60)
    print(f"使用种子: {SEED}\n")

    # 测试 Python random 模块
    print("1. Python random 模块:")
    random_samples = random.sample(range(100), 5)
    random_uniform = random.uniform(0, 1)
    print(f"   random.sample(range(100), 5) = {random_samples}")
    print(f"   random.uniform(0, 1) = {random_uniform:.10f}")

    # 测试 NumPy random
    print("\n2. NumPy random:")
    np_randint = np.random.randint(0, 10, size=5)
    np_uniform = np.random.uniform(0, 1)
    np_normal = np.random.normal(0, 1, size=3)
    print(f"   np.random.randint(0, 10, size=5) = {np_randint}")
    print(f"   np.random.uniform(0, 1) = {np_uniform:.10f}")
    print(f"   np.random.normal(0, 1, size=3) = {np_normal}")

    # 测试 PaddlePaddle
    print("\n3. PaddlePaddle:")
    paddle_tensor = paddle.randn([2, 3])
    paddle_uniform = paddle.rand([2, 2])
    print(f"   paddle.randn([2, 3]) =\n{paddle_tensor.numpy()}")
    print(f"   paddle.rand([2, 2]) =\n{paddle_uniform.numpy()}")

    # 测试经验回放采样（模拟）
    print("\n4. 经验回放采样模拟:")
    memory = list(range(100))
    batch = random.sample(memory, 5)
    print(f"   random.sample(range(100), 5) = {batch}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("请多次运行此脚本，如果输出完全相同，则证明可重复性正确。")
    print("=" * 60)

if __name__ == "__main__":
    test_reproducibility()
