#!/usr/bin/env python3
"""
评估 Real Physics MCTS
使用与 evaluate.py 相同的随机种子系统进行公平对比
"""

import os
import sys
import numpy as np
from GameInterface import GameInterface
from PRNG import PRNG
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent

# 使用与原始 evaluate.py 相同的随机种子
evaluate_random = PRNG()
evaluate_random.seed("RedContritio")


def evaluate_mcts(env, agent, seed, max_steps=200):
    """
    评估 MCTS agent

    Args:
        env: GameInterface 环境
        agent: MCTS Agent
        seed: 随机种子
        max_steps: 最大步数

    Returns:
        (score, total_reward): 最终得分和总奖励
    """
    env.reset(seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, reward, alive = env.next(action)

    total_reward = reward
    step = 0

    while alive and step < max_steps:
        step += 1

        # MCTS 决策
        action = agent.predict(env)[0]

        # 执行动作
        feature, reward, alive = env.next(action)
        total_reward += reward

    return env.game.score, total_reward


if __name__ == "__main__":
    EVALUATE_TIMES = 100
    NUM_SIMULATIONS = 200  # MCTS simulations per step
    MAX_STEPS = 200       # Max steps per episode

    print("="*70)
    print("🎮 Real Physics MCTS Evaluation")
    print("="*70)
    print(f"Configuration:")
    print(f"  Evaluate Times: {EVALUATE_TIMES}")
    print(f"  MCTS Simulations: {NUM_SIMULATIONS}")
    print(f"  Max Steps per Episode: {MAX_STEPS}")
    print("="*70)

    # 创建环境和智能体
    env = GameInterface()
    mcts_agent = RealPhysicsMCTSAgent(num_simulations=NUM_SIMULATIONS)

    scores = []
    rewards = []

    print(f"\n🚀 Starting evaluation...\n")

    for i in range(EVALUATE_TIMES):
        # 使用与 evaluate.py 相同的种子
        seed = evaluate_random.random()

        # 评估 MCTS
        score, reward = evaluate_mcts(env, mcts_agent, seed, max_steps=MAX_STEPS)
        scores.append(score)
        rewards.append(reward)
        print(f"  Episode {i+1:3d}/{EVALUATE_TIMES} | Score: {score:4d} | Reward: {reward:6.2f}")
        
        # 打印进度
        if (i + 1) % 10 == 0:
            avg_score = np.mean(scores[-10:])
            print(f"  Progress: {i+1:3d}/{EVALUATE_TIMES} | "
                  f"Last 10 avg score: {avg_score:6.1f} | "
                  f"Current: {score:4d}")

    # 统计结果
    print(f"\n{'='*70}")
    print("📊 Evaluation Results")
    print(f"{'='*70}")
    print(f"[Real Physics MCTS]:")
    print(f"  Mean Score:  {np.mean(scores):.2f}")
    print(f"  Mean Reward: {np.mean(rewards):.2f}")
    print(f"  Max Score:   {np.max(scores)}")
    print(f"  Max Reward:  {np.max(rewards):.2f}")
    print(f"  Min Score:   {np.min(scores)}")
    print(f"  Min Reward:  {np.min(rewards):.2f}")
    print(f"  Std Score:   {np.std(scores):.2f}")
    print(f"{'='*70}")

    # 保存结果
    results_file = "/Users/ycy/Downloads/DQN_FruitMerger 2/mcts_real_physics_evaluation.txt"
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Real Physics MCTS Evaluation Results\n")
        f.write("="*70 + "\n")
        f.write(f"Evaluate Times: {EVALUATE_TIMES}\n")
        f.write(f"MCTS Simulations: {NUM_SIMULATIONS}\n")
        f.write(f"Max Steps per Episode: {MAX_STEPS}\n")
        f.write("\n")
        f.write(f"Mean Score:  {np.mean(scores):.2f}\n")
        f.write(f"Mean Reward: {np.mean(rewards):.2f}\n")
        f.write(f"Max Score:   {np.max(scores)}\n")
        f.write(f"Max Reward:  {np.max(rewards):.2f}\n")
        f.write(f"Min Score:   {np.min(scores)}\n")
        f.write(f"Min Reward:  {np.min(rewards):.2f}\n")
        f.write(f"Std Score:   {np.std(scores):.2f}\n")
        f.write("="*70 + "\n")

        # 保存所有分数
        f.write("\nAll Scores:\n")
        f.write(str(scores) + "\n")

    print(f"\n✅ Results saved to: {results_file}")
    print(f"\n✅ Evaluation Complete!")
