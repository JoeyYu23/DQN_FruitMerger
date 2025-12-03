#!/usr/bin/env python3
"""
测试不同MERGE_BONUS值的效果
"""

from GameInterface import GameInterface
from mcts.MCTS_tuned import TunedMCTSAgent, TunedConfig
import numpy as np

def test_merge_bonus(bonus_value, test_seeds, num_sims=100):
    """测试特定MERGE_BONUS值"""
    # 临时修改配置
    original_bonus = TunedConfig.MERGE_BONUS
    TunedConfig.MERGE_BONUS = bonus_value

    agent = TunedMCTSAgent(num_sims)
    env = GameInterface()

    scores = []
    steps_list = []

    for seed in test_seeds:
        env.reset(seed=seed)
        env.next(np.random.randint(0, 16))

        steps = 0
        while env.game.alive and steps < 200:
            action = agent.predict(env)[0]
            env.next(action)
            steps += 1

        scores.append(env.game.score)
        steps_list.append(steps)

    # 恢复原值
    TunedConfig.MERGE_BONUS = original_bonus

    return {
        'bonus': bonus_value,
        'scores': scores,
        'steps': steps_list,
        'avg_score': np.mean(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'avg_steps': np.mean(steps_list)
    }


if __name__ == "__main__":
    print("="*70)
    print("🔬 测试不同MERGE_BONUS值的效果")
    print("="*70)

    test_seeds = [1000, 1001, 1002, 1003, 1004]
    bonus_values = [5,20,50]

    print(f"\n配置:")
    print(f"  测试Seeds: {test_seeds}")
    print(f"  MCTS Simulations: 100")
    print(f"  测试MERGE_BONUS值: {bonus_values}")
    print(f"\n开始测试...\n")

    results = []

    for bonus in bonus_values:
        print(f"{'─'*70}")
        print(f"测试 MERGE_BONUS = {bonus}")
        print(f"{'─'*70}")

        result = test_merge_bonus(bonus, test_seeds, num_sims=100)
        results.append(result)

        for seed, score, steps in zip(test_seeds, result['scores'], result['steps']):
            print(f"  Seed {seed}: {score:3d}分 ({steps}步)")

        print(f"  → 平均: {result['avg_score']:.1f}分")
        print()

    # 汇总对比
    print("="*70)
    print("📊 汇总对比")
    print("="*70)
    print(f"{'Bonus':>6} | {'平均得分':>8} | {'最高':>6} | {'最低':>6} | {'平均步数':>8}")
    print("─"*70)

    for r in results:
        print(f"{r['bonus']:6.1f} | {r['avg_score']:8.1f} | {r['max_score']:6d} | "
              f"{r['min_score']:6d} | {r['avg_steps']:8.1f}")

    print("="*70)

    # 找出最佳值
    best = max(results, key=lambda x: x['avg_score'])
    print(f"\n🏆 最佳配置: MERGE_BONUS = {best['bonus']}")
    print(f"   平均得分: {best['avg_score']:.1f}")
    print(f"   提升: {(best['avg_score']/results[0]['avg_score']-1)*100:+.1f}% (相比1.0)")

    # 可视化
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bonuses = [r['bonus'] for r in results]
    avg_scores = [r['avg_score'] for r in results]
    avg_steps = [r['avg_steps'] for r in results]

    # 平均得分
    ax1.plot(bonuses, avg_scores, 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('MERGE_BONUS', fontsize=12)
    ax1.set_ylabel('Average Score', fontsize=12)
    ax1.set_title('Score vs MERGE_BONUS', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 标注最佳值
    best_idx = bonuses.index(best['bonus'])
    ax1.plot(best['bonus'], best['avg_score'], 'r*', markersize=20,
             label=f'Best: {best["bonus"]}')
    ax1.legend()

    # 标注数值
    for b, s in zip(bonuses, avg_scores):
        ax1.text(b, s, f' {s:.1f}', fontsize=9, va='bottom')

    # 平均步数
    ax2.plot(bonuses, avg_steps, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('MERGE_BONUS', fontsize=12)
    ax2.set_ylabel('Average Steps', fontsize=12)
    ax2.set_title('Steps vs MERGE_BONUS', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 标注数值
    for b, s in zip(bonuses, avg_steps):
        ax2.text(b, s, f' {s:.1f}', fontsize=9, va='bottom')

    plt.tight_layout()
    plt.savefig('merge_bonus_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: merge_bonus_comparison.png")
    print("="*70)
