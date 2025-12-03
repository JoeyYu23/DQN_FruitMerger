#!/usr/bin/env python3
"""
快速测试DQN模型性能
"""
import os
import numpy as np
import paddle
from DQN import Agent, RandomAgent, build_model, evaluate
from GameInterface import GameInterface
from PRNG import PRNG

evaluate_random = PRNG()
evaluate_random.seed("RedContritio")

if __name__ == "__main__":
    print("=" * 70)
    print("  🎮 Testing DQN Model Performance")
    print("=" * 70)

    EVALUATE_TIMES = 50  # 测试50局

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2
    e_greed = 0.0  # 测试时不探索
    e_greed_decrement = 0

    env = GameInterface()

    # 创建DQN agent
    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    # 加载最佳模型
    model_path = "weights/best_model.pdparams"
    if os.path.exists(model_path):
        agent.policy_net.set_state_dict(paddle.load(model_path))
        print(f"✅ Loaded model: {model_path}")
    else:
        print(f"❌ Model not found: {model_path}")
        exit(1)

    # 创建随机agent作为baseline
    random_agent = RandomAgent(GameInterface.ACTION_NUM)

    print(f"\n🎯 Running {EVALUATE_TIMES} evaluation games...")
    print("-" * 70)

    dqn_scores, dqn_rewards = [], []
    random_scores, random_rewards = [], []

    for i in range(EVALUATE_TIMES):
        seed = evaluate_random.random()

        # DQN agent
        score1, reward1 = evaluate(env, agent, seed)
        dqn_scores.append(score1)
        dqn_rewards.append(reward1)

        # Random agent (baseline)
        score2, reward2 = evaluate(env, random_agent, seed)
        random_scores.append(score2)
        random_rewards.append(reward2)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{EVALUATE_TIMES} games completed")

    print("\n" + "=" * 70)
    print("  📊 Evaluation Results")
    print("=" * 70)

    print("\n[DQN Agent]:")
    print(f"  Mean Score:   {np.mean(dqn_scores):.2f} ± {np.std(dqn_scores):.2f}")
    print(f"  Mean Reward:  {np.mean(dqn_rewards):.2f}")
    print(f"  Max Score:    {np.max(dqn_scores)}")
    print(f"  Min Score:    {np.min(dqn_scores)}")

    print("\n[Random Agent (Baseline)]:")
    print(f"  Mean Score:   {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
    print(f"  Mean Reward:  {np.mean(random_rewards):.2f}")
    print(f"  Max Score:    {np.max(random_scores)}")
    print(f"  Min Score:    {np.min(random_scores)}")

    # 计算提升
    improvement = (np.mean(dqn_scores) / np.mean(random_scores) - 1) * 100
    print("\n" + "=" * 70)
    print(f"  🚀 DQN vs Random: {improvement:+.1f}% improvement")
    print("=" * 70)

    # 分数分布
    print("\n[Score Distribution]:")
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    for i in range(len(bins) - 1):
        count_dqn = sum(bins[i] <= s < bins[i+1] for s in dqn_scores)
        count_rand = sum(bins[i] <= s < bins[i+1] for s in random_scores)
        print(f"  {bins[i]:4d}-{bins[i+1]:4d}: DQN={count_dqn:2d}, Random={count_rand:2d}")
