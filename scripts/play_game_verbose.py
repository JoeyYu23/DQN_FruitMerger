#!/usr/bin/env python3
"""
详细版游戏 - 展示关键决策
"""

import numpy as np
import time
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent

def play_with_commentary(num_simulations=200):
    """带解说的游戏"""
    print("="*70)
    print("🎮 MCTS 实战游戏 - 带详细解说")
    print("="*70)
    print(f"\n配置: 每步 {num_simulations} 次模拟")
    print("展示关键时刻的决策思路\n")

    env = GameInterface()
    agent = FastMCTSAgent(num_simulations=num_simulations)

    env.reset(seed=888)
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    total_time = 0
    milestones = [10, 20, 30, 40, 50]  # 展示这些步骤的详细分析

    print("游戏开始！\n")

    while alive:
        step += 1

        # 显示详细分析的时机
        show_detail = step in milestones or not alive

        if show_detail:
            print(f"\n{'='*70}")
            print(f"📍 第 {step} 步详细分析")
            print(f"{'='*70}")
            print(f"当前得分: {env.game.score}")
            print(f"下一个水果: {env.game.current_fruit_type}")

        # MCTS决策
        start = time.time()

        # 获取搜索树信息
        simple_state = agent._convert_state(env)
        grid_action = agent.mcts.search(simple_state, num_simulations)

        elapsed = time.time() - start
        total_time += elapsed

        # 分析搜索结果
        if show_detail:
            root = agent.mcts.root
            sorted_children = sorted(root.children.items(),
                                    key=lambda x: x[1].visit_count,
                                    reverse=True)

            print(f"\n🤔 MCTS思考结果:")
            print(f"  思考时间: {elapsed:.3f}秒")
            print(f"  速度: {num_simulations/elapsed:.0f} r/s")

            print(f"\n  前3候选:")
            for idx, (action, child) in enumerate(sorted_children[:3], 1):
                total_visits = sum(c.visit_count for _, c in root.children.items())
                visit_rate = child.visit_count / total_visits * 100
                marker = "👉" if idx == 1 else "  "
                print(f"  {marker}列{action}: {child.visit_count}次访问 "
                      f"({visit_rate:.0f}%), Q={child.get_value():.1f}")

            print(f"\n  决策: 选择列 {grid_action}")

        # 转换并执行动作
        game_action = int(grid_action * 16 / 10)
        game_action = min(15, max(0, game_action))

        feature, reward, alive = env.next(game_action)

        if show_detail:
            print(f"\n📊 执行结果:")
            print(f"  奖励: {reward:+d}")
            print(f"  新得分: {env.game.score}")

            if reward > 10:
                print(f"  💥 大合并！获得 {reward} 分")
            elif reward > 0:
                print(f"  ✓ 成功合并")

        elif step % 5 == 0:
            # 简要进度
            print(f"第 {step:3d} 步 | 得分: {env.game.score:4d} | "
                  f"{num_simulations/elapsed:.0f} r/s", end="\r")

    # 最终统计
    avg_time = total_time / step if step > 0 else 0

    print(f"\n\n{'='*70}")
    print(f"🏁 游戏结束！")
    print(f"{'='*70}")
    print(f"\n📊 最终统计:")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  总用时: {total_time:.1f}秒")
    print(f"  平均速度: {num_simulations/avg_time:.0f} rollouts/秒")
    print(f"  平均每步: {avg_time:.2f}秒")

    # 评估表现
    print(f"\n🎯 表现评估:")
    if env.game.score >= 300:
        print(f"  ⭐⭐⭐ 优秀！得分超过300")
    elif env.game.score >= 200:
        print(f"  ⭐⭐ 良好！得分超过200")
    elif env.game.score >= 150:
        print(f"  ⭐ 不错！得分超过150")
    else:
        print(f"  继续努力！")

    print(f"\n  效率评分:")
    print(f"  - 平均得分率: {env.game.score/step:.2f} 分/步")
    print(f"  - 生存步数: {step} 步")

if __name__ == "__main__":
    import sys

    num_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    play_with_commentary(num_sims)
