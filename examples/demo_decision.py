#!/usr/bin/env python3
"""
MCTS 决策过程可视化演示
展示MCTS如何思考和选择最佳动作
"""

import numpy as np
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent, FastGameState, FastMCTS
import time

def visualize_tree_stats(mcts: FastMCTS):
    """可视化搜索树统计信息"""
    root = mcts.root

    if not root or not root.children:
        print("  [搜索树为空]")
        return

    print("\n" + "="*70)
    print("🌳 搜索树分析")
    print("="*70)

    # 根节点信息
    print(f"\n根节点统计:")
    print(f"  总访问次数: {root.visit_count}")
    print(f"  总价值: {root.total_value:.1f}")
    print(f"  平均价值: {root.get_value():.1f}")
    print(f"  扩展的子节点数: {len(root.children)}")

    # 每个动作的详细信息
    print(f"\n各动作详细分析 (列 0-9):")
    print("-"*70)
    print(f"{'列':<4} {'访问':<8} {'平均Q值':<12} {'PUCT':<12} {'选择率':<10}")
    print("-"*70)

    # 按访问次数排序
    sorted_children = sorted(root.children.items(),
                            key=lambda x: x[1].visit_count,
                            reverse=True)

    total_visits = sum(child.visit_count for _, child in root.children.items())

    for action, child in sorted_children:
        visit_count = child.visit_count
        q_value = child.get_value()
        puct = child.get_puct()
        visit_rate = visit_count / total_visits * 100 if total_visits > 0 else 0

        # 标记最佳动作
        marker = "👉" if action == root.best_action() else "  "

        print(f"{marker} {action:<2} {visit_count:<8} {q_value:<12.2f} {puct:<12.2f} {visit_rate:<9.1f}%")

    print("-"*70)

    # 最佳动作
    best_action = root.best_action()
    best_child = root.children[best_action]

    print(f"\n✅ 最佳选择: 列 {best_action}")
    print(f"   原因分析:")
    print(f"   - 访问次数最多: {best_child.visit_count} 次")
    print(f"   - 平均收益: {best_child.get_value():.2f}")
    print(f"   - 被选择概率: {best_child.visit_count/total_visits*100:.1f}%")

def show_board_state(state: FastGameState):
    """显示棋盘状态"""
    print("\n📋 当前棋盘状态:")
    print("   ", end="")
    for col in range(state.width):
        print(f"{col:<2}", end=" ")
    print()

    # 只显示有水果的行
    first_fruit_row = state.height
    for row in range(state.height):
        if any(state.grid[row, col] != 0 for col in range(state.width)):
            first_fruit_row = row
            break

    # 显示从警戒线到底部
    start_row = min(state.warning_line, first_fruit_row)

    for row in range(start_row, state.height):
        if row == state.warning_line:
            print(f"⚠️ ", end="")  # 警戒线
        else:
            print(f"{row:2} ", end="")

        for col in range(state.width):
            fruit = state.grid[row, col]
            if fruit == 0:
                print("· ", end=" ")
            else:
                # 用不同符号表示不同水果
                symbols = [" ", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
                print(f"{symbols[min(fruit, 10)]:<2}", end=" ")
        print()

    print(f"\n   当前水果: {state.current_fruit} | 得分: {state.score}")

def demonstrate_decision_process(num_simulations=200, num_steps=5):
    """演示决策过程"""
    print("="*70)
    print("🎮 MCTS 决策过程演示")
    print("="*70)
    print(f"\n配置: 每步运行 {num_simulations} 次模拟")
    print(f"展示前 {num_steps} 步的详细决策过程\n")

    # 创建环境
    env = GameInterface()
    env.reset(seed=42)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0

    while alive and step < num_steps:
        step += 1

        print("\n" + "🔷"*35)
        print(f"第 {step} 步决策")
        print("🔷"*35)

        # 创建MCTS智能体
        agent = FastMCTSAgent(num_simulations=num_simulations)

        # 转换状态
        simple_state = agent._convert_state(env)

        # 显示当前状态
        show_board_state(simple_state)

        print(f"\n🤔 开始思考... (运行 {num_simulations} 次模拟)")

        # 运行MCTS搜索
        start_time = time.time()
        grid_action = agent.mcts.search(simple_state, num_simulations)
        elapsed = time.time() - start_time

        # 显示搜索统计
        visualize_tree_stats(agent.mcts)

        print(f"\n⏱️  思考用时: {elapsed:.3f}秒")
        print(f"   速度: {num_simulations/elapsed:.0f} rollouts/秒")

        # 转换为游戏动作
        game_action = int(grid_action * 16 / 10)
        game_action = min(15, max(0, game_action))

        print(f"\n💡 决策结果:")
        print(f"   选择在第 {grid_action} 列放置水果 (游戏坐标: action {game_action})")

        # 执行动作
        feature, reward, alive = env.next(game_action)

        print(f"\n📊 结果:")
        print(f"   即时奖励: {reward}")
        print(f"   当前得分: {env.game.score}")
        print(f"   游戏状态: {'继续' if alive else '结束'}")

        input("\n按回车继续下一步...")

    print("\n" + "="*70)
    print("演示结束")
    print("="*70)
    print(f"\n最终得分: {env.game.score}")
    print(f"总步数: {step}")

def quick_demo():
    """快速演示单步决策"""
    print("="*70)
    print("⚡ 快速演示：MCTS如何选择动作")
    print("="*70)

    # 创建简单测试状态
    state = FastGameState()

    # 手动设置一些水果
    state.grid[15, 5] = 1  # 底部中间放一个葡萄
    state.grid[15, 4] = 2  # 旁边放一个樱桃
    state.grid[15, 6] = 1  # 另一边也放一个葡萄
    state.grid[14, 5] = 2  # 上面放一个樱桃
    state.current_fruit = 1  # 当前是葡萄

    show_board_state(state)

    print("\n🤔 MCTS开始分析...")
    print("   当前要放置: 葡萄(①)")
    print("   可能的策略:")
    print("   1. 放在第4列 → 可能与底部的樱桃合并")
    print("   2. 放在第5列 → 可以与底部的葡萄合并 ✨")
    print("   3. 放在第6列 → 可以与底部的葡萄合并 ✨")

    # 运行MCTS
    mcts = FastMCTS()
    print(f"\n   运行 200 次模拟搜索...")
    start = time.time()
    best_action = mcts.search(state, num_simulations=200)
    elapsed = time.time() - start

    # 显示结果
    visualize_tree_stats(mcts)

    print(f"\n⏱️  思考用时: {elapsed:.3f}秒 ({200/elapsed:.0f} r/s)")
    print(f"\n✅ MCTS选择: 第 {best_action} 列")
    print(f"   这是最有利的选择，因为可以触发合并连锁反应！")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速演示
        quick_demo()
    else:
        # 完整演示
        print("\n选择演示模式:")
        print("1. 快速演示 (单步详细分析)")
        print("2. 完整演示 (多步游戏过程)")

        choice = input("\n请选择 [1/2] (默认1): ").strip() or "1"

        if choice == "1":
            quick_demo()
        else:
            num_steps = input("\n演示多少步? [默认3]: ").strip() or "3"
            num_sims = input("每步模拟次数? [默认200]: ").strip() or "200"

            demonstrate_decision_process(
                num_simulations=int(num_sims),
                num_steps=int(num_steps)
            )
