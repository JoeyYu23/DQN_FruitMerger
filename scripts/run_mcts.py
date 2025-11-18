#!/usr/bin/env python3
"""
简单运行脚本 - MCTS玩合成大西瓜
"""

import numpy as np
import time
from GameInterface import GameInterface
from MCTS import MCTSAgent

def play_one_game(num_simulations=100, show_steps=True):
    """运行一局游戏"""
    print("="*60)
    print(f"MCTS Agent 玩合成大西瓜 (每步 {num_simulations} 次模拟)")
    print("="*60)

    # 创建环境和智能体
    env = GameInterface()
    agent = MCTSAgent(num_simulations=num_simulations)

    # 重置环境
    env.reset(seed=12345)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    total_time = 0

    print("\n开始游戏...")

    while alive:
        step += 1

        # MCTS选择动作
        start = time.time()
        action = agent.predict(env)
        elapsed = time.time() - start
        total_time += elapsed

        # 执行动作
        feature, reward, alive = env.next(action[0])

        if show_steps and (step % 5 == 0 or not alive):
            print(f"第 {step:3d} 步 | 得分: {env.game.score:4d} | "
                  f"用时: {elapsed:.2f}秒 | 平均: {total_time/step:.2f}秒/步")

    avg_time = total_time / step if step > 0 else 0

    print("\n" + "="*60)
    print(f"游戏结束!")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  平均时间: {avg_time:.2f}秒/步")
    print(f"  总用时: {total_time:.1f}秒")
    print("="*60)

    return env.game.score, step, avg_time

if __name__ == "__main__":
    import sys

    # 解析参数
    if len(sys.argv) > 1:
        num_sims = int(sys.argv[1])
    else:
        num_sims = 100  # 默认100次模拟

    print(f"\n使用配置: {num_sims} 次模拟/步\n")

    score, steps, avg_time = play_one_game(num_simulations=num_sims)

    print(f"\n💡 提示:")
    print(f"   - 当前速度: ~{num_sims/avg_time:.0f} rollouts/秒")
    print(f"   - 减少模拟次数可加快速度 (但可能降低水平)")
    print(f"   - 运行方式: python3 run_mcts.py [模拟次数]")
    print(f"   - 例如: python3 run_mcts.py 50  (更快但较弱)")
    print(f"   - 例如: python3 run_mcts.py 200 (较慢但更强)")
