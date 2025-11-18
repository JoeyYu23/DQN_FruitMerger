#!/usr/bin/env python3
"""
简化版决策演示 - 直接展示3步
"""

import numpy as np
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent
import time

def show_decision_analysis(env, step_num):
    """展示单步决策分析"""
    print(f"\n{'='*70}")
    print(f"🎯 第 {step_num} 步 - MCTS决策分析")
    print(f"{'='*70}")

    # 创建MCTS
    agent = FastMCTSAgent(num_simulations=200)
    simple_state = agent._convert_state(env)

    print(f"\n当前状态:")
    print(f"  得分: {env.game.score}")
    print(f"  下一个水果: {env.game.current_fruit_type}")

    # 执行搜索
    print(f"\n🤔 MCTS思考中 (200次模拟)...")
    start = time.time()
    grid_action = agent.mcts.search(simple_state, num_simulations=200)
    elapsed = time.time() - start

    # 分析结果
    root = agent.mcts.root

    print(f"\n📊 搜索统计:")
    print(f"  总模拟次数: {root.visit_count}")
    print(f"  扩展节点数: {len(root.children)}")
    print(f"  思考时间: {elapsed:.3f}秒")
    print(f"  速度: {200/elapsed:.0f} rollouts/秒")

    # 显示前5个候选动作
    print(f"\n🏆 候选动作排名 (按访问次数):")
    print(f"  {'排名':<6} {'列':<6} {'访问次数':<10} {'平均价值':<12} {'选择概率'}")
    print(f"  {'-'*60}")

    sorted_children = sorted(root.children.items(),
                            key=lambda x: x[1].visit_count,
                            reverse=True)

    total_visits = sum(c.visit_count for _, c in root.children.items())

    for idx, (action, child) in enumerate(sorted_children[:5], 1):
        visit_rate = child.visit_count / total_visits * 100
        marker = "👉" if idx == 1 else "  "
        print(f"  {marker}{idx:<5} {action:<6} {child.visit_count:<10} "
              f"{child.get_value():<12.1f} {visit_rate:.1f}%")

    # 最佳选择
    best_action = root.best_action()
    best_child = root.children[best_action]

    print(f"\n✅ 最终决策: 在第 {best_action} 列放置水果")
    print(f"  理由: 该位置被模拟了 {best_child.visit_count} 次 ({best_child.visit_count/total_visits*100:.0f}%)")
    print(f"  预期价值: {best_child.get_value():.1f}")

    # 转换为游戏动作并执行
    game_action = int(grid_action * 16 / 10)
    game_action = min(15, max(0, game_action))

    return game_action

def main():
    """运行3步演示"""
    print("\n" + "🎮"*35)
    print("MCTS 决策过程实战演示")
    print("🎮"*35)

    env = GameInterface()
    env.reset(seed=123)

    # 第一步随机初始化
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    # 演示3步
    for step in range(1, 4):
        if not alive:
            break

        # 展示决策过程
        game_action = show_decision_analysis(env, step)

        # 执行动作
        feature, reward, alive = env.next(game_action)

        print(f"\n📈 执行结果:")
        print(f"  奖励: +{reward}")
        print(f"  新得分: {env.game.score}")

        if step < 3 and alive:
            print(f"\n{'·'*70}")

    print(f"\n\n{'='*70}")
    print(f"演示结束")
    print(f"{'='*70}")
    print(f"最终得分: {env.game.score}")

    print(f"\n💡 MCTS决策特点:")
    print(f"  ✓ 通过大量模拟探索可能性")
    print(f"  ✓ 访问次数多的动作 = 更可靠的选择")
    print(f"  ✓ 平衡探索(新动作)和利用(好动作)")
    print(f"  ✓ 自动考虑未来多步的影响")

if __name__ == "__main__":
    main()
