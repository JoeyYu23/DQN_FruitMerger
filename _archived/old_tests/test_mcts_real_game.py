#!/usr/bin/env python3
"""
MCTS连接真实游戏环境测试
减少simulation步数以提高速度
"""

from GameInterface import GameInterface
from mcts.MCTS_tuned import TunedMCTSAgent
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def test_mcts_real_game(seed=888, num_simulations=50, show_game=True):
    """
    使用MCTS玩真实游戏

    Args:
        seed: 随机种子
        num_simulations: MCTS模拟次数（减少以提高速度）
        show_game: 是否显示游戏画面
    """
    print("="*70)
    print("🎮 MCTS + 真实游戏环境")
    print("="*70)
    print(f"配置:")
    print(f"  Seed: {seed}")
    print(f"  MCTS Simulations: {num_simulations}")
    print(f"  显示画面: {show_game}")
    print("="*70)

    # 创建真实游戏环境
    env = GameInterface()
    agent = TunedMCTSAgent(num_simulations=num_simulations)

    # 重置游戏
    env.reset(seed=seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    scores = [0]
    start_time = time.time()

    print(f"\n🚀 开始游戏...\n")

    while alive and step < 200:
        step += 1

        # MCTS决策
        step_start = time.time()
        action = agent.predict(env)[0]
        step_time = time.time() - step_start

        # 执行动作
        feature, reward, alive = env.next(action)
        scores.append(env.game.score)

        # 显示游戏画面（可选）
        if show_game and step % 5 == 0:  # 每5步显示一次
            screen = env.game.draw()
            cv2.imshow('MCTS Playing', screen)
            cv2.waitKey(1)

        # 打印进度
        if step % 10 == 0 or not alive:
            print(f"  Step {step:3d}: Score={env.game.score:3d}, "
                  f"Action={action:2d}, "
                  f"Time={step_time:.2f}s")

    total_time = time.time() - start_time

    if show_game:
        cv2.destroyAllWindows()

    print(f"\n{'='*70}")
    print("🏁 游戏结束!")
    print(f"{'='*70}")
    print(f"📊 统计:")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  平均每步得分: {env.game.score/step:.2f}")
    print(f"  总耗时: {total_time:.1f}秒")
    print(f"  平均每步耗时: {total_time/step:.2f}秒")
    print(f"{'='*70}")

    # 绘制得分曲线
    plt.figure(figsize=(10, 6))
    plt.plot(scores, linewidth=2, color='green', marker='o', markersize=3)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'MCTS Real Game (seed={seed}, sims={num_simulations})',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 标注最终得分
    plt.text(len(scores)-1, scores[-1], f' Final: {scores[-1]}',
            fontsize=11, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'mcts_real_game_seed{seed}_sims{num_simulations}.png',
                dpi=150, bbox_inches='tight')
    print(f"\n📈 得分曲线已保存: mcts_real_game_seed{seed}_sims{num_simulations}.png")

    return env.game.score, step


def compare_simulation_counts(seed=888):
    """
    对比不同simulation数量的效果
    """
    print("\n" + "="*70)
    print("🔬 对比不同Simulation数量")
    print("="*70)

    sim_counts = [20, 50, 100, 200]
    results = []

    for sims in sim_counts:
        print(f"\n{'─'*70}")
        print(f"测试 {sims} simulations...")
        print(f"{'─'*70}")

        score, steps = test_mcts_real_game(
            seed=seed,
            num_simulations=sims,
            show_game=False
        )

        results.append({
            'sims': sims,
            'score': score,
            'steps': steps
        })

    # 对比结果
    print("\n" + "="*70)
    print("📊 对比结果:")
    print("="*70)
    print(f"{'Sims':>6} | {'Score':>6} | {'Steps':>6} | {'Score/Step':>10}")
    print("─"*70)

    for r in results:
        print(f"{r['sims']:6d} | {r['score']:6d} | {r['steps']:6d} | "
              f"{r['score']/r['steps']:10.2f}")

    print("="*70)

    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sims = [r['sims'] for r in results]
    scores = [r['score'] for r in results]
    steps = [r['steps'] for r in results]

    # 得分对比
    ax1.plot(sims, scores, 'o-', linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Simulation Count', fontsize=12)
    ax1.set_ylabel('Final Score', fontsize=12)
    ax1.set_title('Score vs Simulation Count', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 标注数值
    for s, sc in zip(sims, scores):
        ax1.text(s, sc, f' {sc}', fontsize=10, va='center')

    # 步数对比
    ax2.plot(sims, steps, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.set_xlabel('Simulation Count', fontsize=12)
    ax2.set_ylabel('Steps', fontsize=12)
    ax2.set_title('Steps vs Simulation Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 标注数值
    for s, st in zip(sims, steps):
        ax2.text(s, st, f' {st}', fontsize=10, va='center')

    plt.tight_layout()
    plt.savefig('mcts_simulation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 对比图已保存: mcts_simulation_comparison.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MCTS真实游戏测试')
    parser.add_argument('--seed', type=int, default=888, help='随机种子')
    parser.add_argument('--sims', type=int, default=50,
                       help='MCTS模拟次数 (默认50)')
    parser.add_argument('--show', action='store_true',
                       help='显示游戏画面')
    parser.add_argument('--compare', action='store_true',
                       help='对比不同simulation数量')

    args = parser.parse_args()

    if args.compare:
        compare_simulation_counts(seed=args.seed)
    else:
        test_mcts_real_game(
            seed=args.seed,
            num_simulations=args.sims,
            show_game=args.show
        )

    print("\n✅ 完成!")
