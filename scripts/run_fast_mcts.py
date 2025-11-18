#!/usr/bin/env python3
"""
运行优化版 MCTS
"""

import numpy as np
import time
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent

def play_fast_game(num_simulations=200):
    """运行优化版MCTS"""
    print("="*60)
    print(f"优化版 MCTS (每步 {num_simulations} 次模拟)")
    print("="*60)

    env = GameInterface()
    agent = FastMCTSAgent(num_simulations=num_simulations)

    env.reset(seed=12345)
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    total_time = 0

    print("\n开始游戏...\n")

    while alive:
        step += 1

        start = time.time()
        action = agent.predict(env)
        elapsed = time.time() - start
        total_time += elapsed

        feature, reward, alive = env.next(action[0])

        if step % 5 == 0 or not alive:
            rollouts_per_sec = num_simulations / elapsed if elapsed > 0 else 0
            print(f"第 {step:3d} 步 | 得分: {env.game.score:4d} | "
                  f"用时: {elapsed:.2f}秒 | {rollouts_per_sec:.0f} r/s")

    avg_time = total_time / step if step > 0 else 0

    print("\n" + "="*60)
    print(f"游戏结束!")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  平均速度: {num_simulations/avg_time:.0f} rollouts/秒")
    print(f"  平均用时: {avg_time:.2f}秒/步")
    print(f"  总用时: {total_time:.1f}秒")
    print("="*60)

if __name__ == "__main__":
    import sys

    num_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    play_fast_game(num_sims)
