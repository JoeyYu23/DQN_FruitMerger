"""
测试新的综合奖励函数
"""
from GameInterface import GameInterface
import numpy as np

def test_reward_function():
    """测试新奖励函数的各个组成部分"""
    print("=" * 60)
    print("测试新的综合奖励函数")
    print("=" * 60)

    env = GameInterface()

    # 运行几步游戏
    test_steps = 10
    total_rewards = []

    for step in range(test_steps):
        action = np.random.randint(0, env.action_num)
        feature, reward, alive = env.next(action)

        total_rewards.append(reward)

        print(f"\n步骤 {step + 1}:")
        print(f"  动作: {action}")
        print(f"  奖励: {reward:.2f}")
        print(f"  得分: {env.game.score}")
        print(f"  存活: {alive}")
        print(f"  水果数量: {len(env.game.fruits)}")
        print(f"  连续合成: {env.consecutive_merges}")

        # 显示奖励组成（估算）
        space_util = env.calculate_space_utilization()
        height_ratio = env.calculate_max_height_ratio()
        print(f"  空间利用率: {space_util:.2%} (惩罚: {-space_util * 10:.2f})")
        print(f"  高度比例: {height_ratio:.2%} (惩罚: {-height_ratio * 5:.2f})")

        if not alive:
            print("\n游戏结束！")
            break

    print("\n" + "=" * 60)
    print("统计信息:")
    print(f"  总步数: {len(total_rewards)}")
    print(f"  总奖励: {sum(total_rewards):.2f}")
    print(f"  平均奖励: {np.mean(total_rewards):.2f}")
    print(f"  最大奖励: {max(total_rewards):.2f}")
    print(f"  最小奖励: {min(total_rewards):.2f}")
    print(f"  最终得分: {env.game.score}")
    print("=" * 60)

def test_merge_rewards():
    """测试不同合成类型的奖励"""
    print("\n" + "=" * 60)
    print("合成奖励表 (指数增长: 2^type)")
    print("=" * 60)

    for fruit_type in range(1, 12):
        if fruit_type < 11:
            reward = 2 ** fruit_type
        else:
            reward = 2 ** 11
        print(f"  类型 {fruit_type:2d} 合成: {reward:6d} 分")

    print("=" * 60)

if __name__ == "__main__":
    test_merge_rewards()
    test_reward_function()
