"""
测试更强的MCTS（更多模拟次数）
"""
import numpy as np
import time
from GameInterface import GameInterface
from mcts.MCTS_optimized import FastMCTSAgent


def test_mcts_with_different_sims():
    """测试不同模拟次数的MCTS表现"""
    print("=" * 70)
    print("🎯 MCTS模拟次数对比测试")
    print("=" * 70)

    env = GameInterface()
    num_games = 3  # 每个配置测试3局

    sim_counts = [50, 100, 200, 500]

    results = {}

    for num_sims in sim_counts:
        print(f"\n[测试 {num_sims}次模拟/步]")
        print("-" * 70)

        agent = FastMCTSAgent(num_simulations=num_sims)
        scores = []
        times = []

        for game_idx in range(num_games):
            env.reset(seed=game_idx * 100)

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

                action_val = int(action[0]) if isinstance(action, np.ndarray) else int(action)
                feature, reward, alive = env.next(action_val)

            final_score = env.game.score
            avg_time = total_time / step if step > 0 else 0

            scores.append(final_score)
            times.append(avg_time)

            print(f"  游戏 {game_idx + 1}: 得分={final_score:4d}, {step:3d}步, {avg_time:.3f}s/步")

        mean_score = np.mean(scores)
        mean_time = np.mean(times)

        results[num_sims] = {
            'scores': scores,
            'mean': mean_score,
            'max': np.max(scores),
            'time': mean_time
        }

        print(f"  平均: {mean_score:.1f}分, {mean_time:.3f}秒/步")

    # 总结
    print("\n" + "=" * 70)
    print("📊 总结")
    print("=" * 70)
    print(f"\n{'模拟次数':<12} {'平均得分':<15} {'最高分':<10} {'时间(s/步)':<12}")
    print("-" * 70)

    for num_sims in sim_counts:
        r = results[num_sims]
        print(f"{num_sims:<12} {r['mean']:>8.1f}       {r['max']:>10}    {r['time']:>10.3f}")

    print("\n💡 建议:")
    best_sim = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"  最佳模拟次数: {best_sim[0]}次 (平均{best_sim[1]['mean']:.1f}分)")
    print(f"  随机基准: ~141分")

    if best_sim[1]['mean'] > 141:
        print(f"  ✅ MCTS已超越随机策略！")
    else:
        print(f"  ⚠️  建议增加模拟次数或优化评估函数")


if __name__ == "__main__":
    test_mcts_with_different_sims()
