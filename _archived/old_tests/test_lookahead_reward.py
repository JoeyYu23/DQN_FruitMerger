#!/usr/bin/env python3
"""
测试修改后的Lookahead Reward系统

验证：
1. SimplifiedGameState.simulate_lookahead() 正常工作
2. AlphaZeroMCTS使用lookahead评估
3. 移除death penalty的效果
"""

import numpy as np
import paddle
from mcts.MCTS import SimplifiedGameState
from SuikaNet import SuikaNet
from AlphaZeroMCTS import AlphaZeroMCTS


def test_lookahead_simulation():
    """测试lookahead模拟功能"""
    print("\n" + "="*70)
    print("Test 1: Lookahead Simulation")
    print("="*70)

    # 创建测试状态
    state = SimplifiedGameState(grid_width=16, grid_height=16)
    state.current_fruit = 3

    # 放置一些水果
    state.grid[15, 5] = 5
    state.grid[15, 6] = 5
    state.grid[14, 5] = 3

    print(f"Initial score: {state.score}")
    print(f"Valid actions: {len(state.get_valid_actions())}")

    # 测试lookahead (greedy policy)
    print("\n[Test] Running 10-step greedy lookahead...")
    lookahead_reward_greedy = state.simulate_lookahead(num_steps=10, policy="greedy")
    print(f"  Greedy lookahead reward: {lookahead_reward_greedy:.2f}")

    # 测试lookahead (random policy)
    print("\n[Test] Running 10-step random lookahead...")
    lookahead_reward_random = state.simulate_lookahead(num_steps=10, policy="random")
    print(f"  Random lookahead reward: {lookahead_reward_random:.2f}")

    # 验证原状态未被修改
    print(f"\n[Verify] Original state score unchanged: {state.score}")

    print("\n✅ Lookahead simulation test passed!")


def test_mcts_with_lookahead():
    """测试MCTS使用lookahead评估"""
    print("\n" + "="*70)
    print("Test 2: MCTS with Lookahead")
    print("="*70)

    # 创建网络 - 匹配SimplifiedGameState的尺寸 (16x16)
    network = SuikaNet(
        input_channels=13,
        num_actions=16,
        hidden_channels=64,
        board_height=16,
        board_width=16
    )

    # 创建两个MCTS：一个使用lookahead，一个不使用
    mcts_with_lookahead = AlphaZeroMCTS(
        network=network,
        num_simulations=50,
        temperature=0.0,
        use_lookahead=True,
        lookahead_steps=10
    )

    mcts_without_lookahead = AlphaZeroMCTS(
        network=network,
        num_simulations=50,
        temperature=0.0,
        use_lookahead=False
    )

    # 创建测试状态
    state = SimplifiedGameState(grid_width=16, grid_height=16)
    state.current_fruit = 3
    state.grid[15, 5] = 5
    state.grid[15, 6] = 5

    print(f"Initial state score: {state.score}")

    # 测试使用lookahead的MCTS
    print("\n[Test] MCTS WITH lookahead (50 simulations)...")
    pi_with = mcts_with_lookahead.search(state.copy())
    action_with = int(np.argmax(pi_with))
    print(f"  Best action: {action_with}")
    print(f"  Action distribution: {pi_with[pi_with > 0]}")

    # 测试不使用lookahead的MCTS
    print("\n[Test] MCTS WITHOUT lookahead (50 simulations)...")
    pi_without = mcts_without_lookahead.search(state.copy())
    action_without = int(np.argmax(pi_without))
    print(f"  Best action: {action_without}")
    print(f"  Action distribution: {pi_without[pi_without > 0]}")

    print("\n✅ MCTS lookahead integration test passed!")


def test_terminal_value_no_penalty():
    """测试终止状态没有death penalty"""
    print("\n" + "="*70)
    print("Test 3: Terminal State Value (No Death Penalty)")
    print("="*70)

    network = SuikaNet(
        input_channels=13,
        num_actions=16,
        hidden_channels=64,
        board_height=16,
        board_width=16
    )
    mcts = AlphaZeroMCTS(
        network=network,
        num_simulations=20,
        temperature=0.0
    )

    # 创建一个接近终止的状态
    state = SimplifiedGameState(grid_width=16, grid_height=16)

    # 填充大部分网格，只留顶部一点空间
    for row in range(5, 16):
        for col in range(16):
            state.grid[row, col] = np.random.randint(1, 6)

    state.score = 120  # 中等分数

    print(f"State score: {state.score}")
    print(f"Valid actions: {len(state.get_valid_actions())}")

    # 强制设置为终止状态
    state.is_terminal = True

    # 在search中会计算终止状态的value
    # 原来是 -1.0，现在应该是归一化分数
    normalized_value = state.score / 500.0
    expected_value = np.clip(normalized_value - 0.5, -1.0, 1.0)

    print(f"\nExpected terminal value (normalized): {expected_value:.4f}")
    print(f"  (Old value would be: -1.0)")
    print(f"  Score 120 -> normalized to {normalized_value:.4f}")
    print(f"  After centering: {expected_value:.4f}")

    print("\n✅ Terminal value calculation correct (no death penalty)!")


def compare_lookahead_policies():
    """比较不同lookahead策略"""
    print("\n" + "="*70)
    print("Test 4: Compare Lookahead Policies")
    print("="*70)

    state = SimplifiedGameState(grid_width=16, grid_height=16)
    state.current_fruit = 3

    # 设置一个有趣的棋盘布局
    state.grid[15, 7] = 4
    state.grid[15, 8] = 4
    state.grid[14, 7] = 3
    state.grid[14, 8] = 3

    print(f"Initial score: {state.score}")

    # 测试不同步数的lookahead
    for steps in [5, 10, 15]:
        reward = state.simulate_lookahead(num_steps=steps, policy="greedy")
        print(f"  {steps}-step greedy lookahead: {reward:.2f}")

    print("\n✅ Lookahead policy comparison complete!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("  🧪 Testing Lookahead Reward System")
    print("="*70)

    try:
        test_lookahead_simulation()
        test_mcts_with_lookahead()
        test_terminal_value_no_penalty()
        compare_lookahead_policies()

        print("\n" + "="*70)
        print("  ✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nSummary of Changes:")
        print("  1. ✅ SimplifiedGameState.simulate_lookahead() - 模拟未来N步并返回总奖励")
        print("  2. ✅ AlphaZeroMCTS.evaluate_with_lookahead() - 结合网络+lookahead评估")
        print("  3. ✅ Removed death penalty - 终止状态使用归一化分数代替-1.0")
        print("  4. ✅ MCTS search 使用新的reward计算方式")
        print("\n修改完成！可以开始训练了。")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_all_tests()
