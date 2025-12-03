#!/usr/bin/env python3
"""
对比优化前后的MCTS性能
"""

from GameInterface import GameInterface
from mcts.MCTS_optimized import FastMCTSAgent as OldMCTS
from mcts.MCTS_tuned import TunedMCTSAgent as NewMCTS
import numpy as np
import matplotlib.pyplot as plt

def test_agent(agent, name, seeds, num_simulations=200):
    """测试智能体"""
    print(f"\n{'='*70}")
    print(f"测试 {name}")
    print(f"{'='*70}")

    env = GameInterface()
    scores = []
    steps_list = []

    for i, seed in enumerate(seeds, 1):
        env.reset(seed=seed)

        # 第一步随机
        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        steps = 0
        while alive and steps < 200:
            action = agent.predict(env)[0]
            feature, reward, alive = env.next(action)
            steps += 1

        scores.append(env.game.score)
        steps_list.append(steps)

        print(f"  [{i:2d}/{len(seeds)}] Seed={seed:4d}: Score={env.game.score:3d}, Steps={steps:2d}")

    print(f"\n{'='*70}")
    print(f"{name} 统计:")
    print(f"  平均得分: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  最高得分: {max(scores)}")
    print(f"  最低得分: {min(scores)}")
    print(f"  平均步数: {np.mean(steps_list):.1f}")
    print(f"{'='*70}")

    return scores, steps_list


def main():
    print("="*70)
    print("MCTS优化版本对比测试")
    print("="*70)

    # 测试种子（使用相同种子确保公平对比）
    test_seeds = [1000 + i for i in range(20)]  # 20局测试

    print(f"\n配置:")
    print(f"  测试局数: {len(test_seeds)}")
    print(f"  MCTS模拟次数: 200")
    print(f"  最大步数: 200")

    # 创建智能体
    old_agent = OldMCTS(num_simulations=200)
    new_agent = NewMCTS(num_simulations=200)

    # 测试
    old_scores, old_steps = test_agent(old_agent, "旧版MCTS (线性penalty)", test_seeds)
    new_scores, new_steps = test_agent(new_agent, "新版MCTS (指数penalty)", test_seeds)

    # 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 分数对比
    ax1 = axes[0, 0]
    x = np.arange(len(test_seeds))
    ax1.plot(x, old_scores, 'b-o', label='Old MCTS', alpha=0.7)
    ax1.plot(x, new_scores, 'r-s', label='New MCTS (Tuned)', alpha=0.7)
    ax1.set_xlabel('Game Index', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Score Comparison', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 分数分布
    ax2 = axes[0, 1]
    bins = np.arange(0, max(max(old_scores), max(new_scores)) + 20, 20)
    ax2.hist(old_scores, bins=bins, alpha=0.5, label='Old MCTS', color='blue')
    ax2.hist(new_scores, bins=bins, alpha=0.5, label='New MCTS', color='red')
    ax2.axvline(np.mean(old_scores), color='blue', linestyle='--', linewidth=2,
                label=f'Old Mean={np.mean(old_scores):.1f}')
    ax2.axvline(np.mean(new_scores), color='red', linestyle='--', linewidth=2,
                label=f'New Mean={np.mean(new_scores):.1f}')
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 步数对比
    ax3 = axes[1, 0]
    ax3.plot(x, old_steps, 'b-o', label='Old MCTS', alpha=0.7)
    ax3.plot(x, new_steps, 'r-s', label='New MCTS (Tuned)', alpha=0.7)
    ax3.set_xlabel('Game Index', fontsize=11)
    ax3.set_ylabel('Steps', fontsize=11)
    ax3.set_title('Steps Comparison', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 统计对比
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算改进
    score_improvement = (np.mean(new_scores) - np.mean(old_scores)) / np.mean(old_scores) * 100
    step_improvement = (np.mean(new_steps) - np.mean(old_steps)) / np.mean(old_steps) * 100

    stats_text = f"""
    📊 Statistical Comparison

    {'─'*45}
    Score Statistics:
    {'─'*45}
      Old MCTS:  {np.mean(old_scores):6.1f} ± {np.std(old_scores):5.1f}
      New MCTS:  {np.mean(new_scores):6.1f} ± {np.std(new_scores):5.1f}

      Improvement: {score_improvement:+.1f}%
      {'✅ Better!' if score_improvement > 0 else '❌ Worse'}

    {'─'*45}
    Step Statistics:
    {'─'*45}
      Old MCTS:  {np.mean(old_steps):6.1f} ± {np.std(old_steps):5.1f}
      New MCTS:  {np.mean(new_steps):6.1f} ± {np.std(new_steps):5.1f}

      Improvement: {step_improvement:+.1f}%
      {'✅ Longer!' if step_improvement > 0 else '❌ Shorter'}

    {'─'*45}
    Win Rate (New > Old):
    {'─'*45}
      Wins:  {sum(1 for n, o in zip(new_scores, old_scores) if n > o):2d} / {len(test_seeds)} = {sum(1 for n, o in zip(new_scores, old_scores) if n > o)/len(test_seeds)*100:.1f}%
      Ties:  {sum(1 for n, o in zip(new_scores, old_scores) if n == o):2d} / {len(test_seeds)}
      Loses: {sum(1 for n, o in zip(new_scores, old_scores) if n < o):2d} / {len(test_seeds)}
    """

    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('mcts_tuned_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 对比图已保存: mcts_tuned_comparison.png")

    # 总结
    print(f"\n{'='*70}")
    print("🎯 总结:")
    print(f"{'='*70}")
    print(f"  分数提升: {score_improvement:+.1f}%")
    print(f"  步数提升: {step_improvement:+.1f}%")
    print(f"  胜率: {sum(1 for n, o in zip(new_scores, old_scores) if n > o)/len(test_seeds)*100:.1f}%")

    if score_improvement > 5:
        print(f"\n  🎉 新版MCTS明显优于旧版！")
    elif score_improvement > 0:
        print(f"\n  ✅ 新版MCTS略优于旧版")
    elif score_improvement > -5:
        print(f"\n  ⚠️  新版与旧版性能接近")
    else:
        print(f"\n  ❌ 新版需要进一步调优")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
