#!/usr/bin/env python3
"""
对比普通MCTS vs 智能MCTS的表现
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from GameInterface import GameInterface
from mcts.MCTS_optimized import FastMCTSAgent
from mcts.MCTS_advanced import SmartMCTSAgent


def play_game(agent, env, seed, agent_name="Agent"):
    """玩一局游戏并返回结果"""
    env.reset(seed=seed)

    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    total_time = 0

    while alive:
        step += 1

        start = time.time()
        action = agent.predict(env)
        elapsed = time.time() - start
        total_time += elapsed

        feature, reward, alive = env.next(action[0])

    avg_time = total_time / step if step > 0 else 0

    return {
        'agent': agent_name,
        'seed': seed,
        'score': env.game.score,
        'steps': step,
        'avg_time': avg_time,
        'total_time': total_time
    }


def compare_agents(num_games=5, num_simulations=100):
    """对比两个agent"""
    print("="*70)
    print("🥊 MCTS对比测试")
    print("="*70)

    env = GameInterface()

    # 创建两个agent
    print(f"\n创建智能体 (每步{num_simulations}次模拟)...")
    normal_agent = FastMCTSAgent(num_simulations=num_simulations)
    smart_agent = SmartMCTSAgent(num_simulations=num_simulations)

    # 测试种子
    seeds = [100, 200, 300, 400, 500][:num_games]

    print(f"\n将进行 {num_games} 局对比测试")
    print(f"种子: {seeds}\n")

    normal_results = []
    smart_results = []

    for i, seed in enumerate(seeds, 1):
        print(f"[{i}/{num_games}] Seed={seed}")

        # 普通MCTS
        print(f"  普通MCTS: ", end="")
        result1 = play_game(normal_agent, env, seed, "Normal")
        print(f"得分{result1['score']}, {result1['steps']}步, "
              f"{result1['avg_time']:.2f}秒/步")
        normal_results.append(result1)

        # 智能MCTS
        print(f"  智能MCTS: ", end="")
        result2 = play_game(smart_agent, env, seed, "Smart")
        print(f"得分{result2['score']}, {result2['steps']}步, "
              f"{result2['avg_time']:.2f}秒/步")
        smart_results.append(result2)

        # 对比
        score_diff = result2['score'] - result1['score']
        if score_diff > 0:
            print(f"  🏆 智能MCTS领先 {score_diff} 分")
        elif score_diff < 0:
            print(f"  📉 普通MCTS领先 {abs(score_diff)} 分")
        else:
            print(f"  🤝 平局")

        print()

    # 统计结果
    print("="*70)
    print("📊 统计结果")
    print("="*70)

    normal_scores = [r['score'] for r in normal_results]
    smart_scores = [r['score'] for r in smart_results]

    normal_times = [r['avg_time'] for r in normal_results]
    smart_times = [r['avg_time'] for r in smart_results]

    print(f"\n普通MCTS:")
    print(f"  平均得分: {np.mean(normal_scores):.1f} ± {np.std(normal_scores):.1f}")
    print(f"  最高得分: {np.max(normal_scores)}")
    print(f"  最低得分: {np.min(normal_scores)}")
    print(f"  平均用时: {np.mean(normal_times):.3f}秒/步")

    print(f"\n智能MCTS:")
    print(f"  平均得分: {np.mean(smart_scores):.1f} ± {np.std(smart_scores):.1f}")
    print(f"  最高得分: {np.max(smart_scores)}")
    print(f"  最低得分: {np.min(smart_scores)}")
    print(f"  平均用时: {np.mean(smart_times):.3f}秒/步")

    # 对比
    print(f"\n📈 对比:")
    score_improvement = np.mean(smart_scores) - np.mean(normal_scores)
    time_increase = np.mean(smart_times) / np.mean(normal_times)

    print(f"  得分提升: {score_improvement:+.1f} ({score_improvement/np.mean(normal_scores)*100:+.1f}%)")
    print(f"  时间增加: {time_increase:.2f}x")

    # 胜负统计
    wins = sum(1 for i in range(num_games) if smart_scores[i] > normal_scores[i])
    losses = sum(1 for i in range(num_games) if smart_scores[i] < normal_scores[i])
    draws = num_games - wins - losses

    print(f"\n🏆 胜负记录:")
    print(f"  智能MCTS: {wins}胜 {draws}平 {losses}负")
    print(f"  胜率: {wins/num_games*100:.0f}%")

    # 结论
    print(f"\n💡 结论:")
    if score_improvement > 10:
        print(f"  ✅ 智能MCTS显著优于普通MCTS")
        print(f"  虽然慢{time_increase:.1f}倍，但得分提升明显")
    elif score_improvement > 0:
        print(f"  ✅ 智能MCTS略优于普通MCTS")
        print(f"  得分稍高，但时间代价较大")
    else:
        print(f"  ⚠️  智能MCTS未体现优势")
        print(f"  可能需要更多模拟次数或调整参数")

    print("\n" + "="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        num_games = int(sys.argv[1])
    else:
        num_games = 5

    if len(sys.argv) > 2:
        num_sims = int(sys.argv[2])
    else:
        num_sims = 100  # 默认100次模拟（平衡速度和质量）

    print(f"\n配置: {num_games}局游戏, 每步{num_sims}次模拟\n")

    compare_agents(num_games, num_sims)
