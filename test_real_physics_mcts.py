#!/usr/bin/env python3
"""
测试 Real Physics MCTS
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent
import numpy as np
import time

def test_real_physics_mcts(seed=888, num_sims=50, max_steps=50):
    """测试Real Physics MCTS"""

    print("="*70)
    print("🎮 Real Physics MCTS 测试")
    print("="*70)
    print(f"配置:")
    print(f"  Seed: {seed}")
    print(f"  Simulations: {num_sims}")
    print(f"  Max Steps: {max_steps}")
    print("="*70)

    # 创建环境和智能体
    env = GameInterface()
    agent = RealPhysicsMCTSAgent(num_simulations=num_sims)

    # 重置游戏
    env.reset(seed=seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    scores = [0]
    start_time = time.time()

    print(f"\n🚀 开始游戏...\n")

    while alive and step < max_steps:
        step += 1

        # 打印进度
        if step % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:2d}: Score={env.game.score:3d}, "
                  f"Fruits={len(env.game.fruits):2d}, "
                  f"Time={elapsed:.1f}s")

        # MCTS决策（使用真实物理）
        step_start = time.time()
        action = agent.predict(env)[0]
        decision_time = time.time() - step_start

        # 执行动作
        feature, reward, alive = env.next(action)
        scores.append(env.game.score)

    total_time = time.time() - start_time

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

    return env.game.score, step


def compare_mcts_versions(seed=888):
    """对比不同MCTS版本"""

    from mcts.MCTS_optimized import FastMCTSAgent as OldMCTS
    from mcts.MCTS_tuned import TunedMCTSAgent as TunedMCTS

    print("\n" + "="*70)
    print("🔬 对比三种MCTS版本")
    print("="*70)

    seeds = [seed, seed+1, seed+2]
    results = {}

    # 测试Real Physics MCTS
    print(f"\n{'─'*70}")
    print("1️⃣ Real Physics MCTS (50 sims, 真实物理)")
    print(f"{'─'*70}")

    env = GameInterface()
    agent = RealPhysicsMCTSAgent(num_simulations=50)
    scores = []

    for s in seeds:
        env.reset(seed=s)
        env.next(np.random.randint(0, 16))

        step = 0
        while env.game.alive and step < 100:
            action = agent.predict(env)[0]
            env.next(action)
            step += 1

        scores.append(env.game.score)
        print(f"  Seed {s}: {env.game.score}")

    results['Real Physics'] = {
        'scores': scores,
        'avg': np.mean(scores),
        'sims': 50
    }
    print(f"  → 平均: {np.mean(scores):.1f}")

    # 测试Tuned MCTS
    print(f"\n{'─'*70}")
    print("2️⃣ Tuned MCTS (100 sims, 简化网格)")
    print(f"{'─'*70}")

    agent2 = TunedMCTS(num_simulations=100)
    scores2 = []

    for s in seeds:
        env.reset(seed=s)
        env.next(np.random.randint(0, 16))

        step = 0
        while env.game.alive and step < 100:
            action = agent2.predict(env)[0]
            env.next(action)
            step += 1

        scores2.append(env.game.score)
        print(f"  Seed {s}: {env.game.score}")

    results['Tuned'] = {
        'scores': scores2,
        'avg': np.mean(scores2),
        'sims': 100
    }
    print(f"  → 平均: {np.mean(scores2):.1f}")

    # 测试Old MCTS
    print(f"\n{'─'*70}")
    print("3️⃣ Optimized MCTS (200 sims, 简化网格)")
    print(f"{'─'*70}")

    agent3 = OldMCTS(num_simulations=200)
    scores3 = []

    for s in seeds:
        env.reset(seed=s)
        env.next(np.random.randint(0, 16))

        step = 0
        while env.game.alive and step < 100:
            action = agent3.predict(env)[0]
            env.next(action)
            step += 1

        scores3.append(env.game.score)
        print(f"  Seed {s}: {env.game.score}")

    results['Optimized'] = {
        'scores': scores3,
        'avg': np.mean(scores3),
        'sims': 200
    }
    print(f"  → 平均: {np.mean(scores3):.1f}")

    # 汇总对比
    print(f"\n{'='*70}")
    print("📊 对比结果")
    print(f"{'='*70}")
    print(f"{'版本':<20} | {'Sims':>6} | {'平均得分':>8} | {'速度':>10}")
    print("─"*70)

    for name, data in results.items():
        sims = data['sims']
        avg = data['avg']
        speed_ratio = sims / 50  # 相对于Real Physics
        print(f"{name:<20} | {sims:6d} | {avg:8.1f} | {speed_ratio:5.1f}x slower")

    print("="*70)

    # 找出最佳
    best = max(results.items(), key=lambda x: x[1]['avg'])
    print(f"\n🏆 最佳: {best[0]} - {best[1]['avg']:.1f}分")

    # 速度/质量比
    print(f"\n⚡ 速度/质量比:")
    for name, data in results.items():
        ratio = data['avg'] / data['sims']
        print(f"  {name}: {ratio:.2f} 分/sim")

    print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Real Physics MCTS测试')
    parser.add_argument('--seed', type=int, default=888, help='随机种子')
    parser.add_argument('--sims', type=int, default=50, help='模拟次数')
    parser.add_argument('--steps', type=int, default=50, help='最大步数')
    parser.add_argument('--compare', action='store_true', help='对比不同版本')

    args = parser.parse_args()

    if args.compare:
        compare_mcts_versions(seed=args.seed)
    else:
        test_real_physics_mcts(
            seed=args.seed,
            num_sims=args.sims,
            max_steps=args.steps
        )

    print("\n✅ 完成!")
