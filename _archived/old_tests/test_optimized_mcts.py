"""
Test Optimized MCTS on Test Set
完整日志版本 - 与CNN-DQN使用相同测试集
"""
import sys
import numpy as np
import time
from datetime import datetime
sys.path.insert(0, '/Users/ycy/Downloads/DQN_FruitMerger')

from GameInterface import GameInterface
from mcts.MCTS_optimized import FastMCTSAgent

def test_mcts(env, agent, num_episodes=100, start_seed=1000):
    """
    测试集评估 - 与CNN-DQN使用相同的测试集
    """
    scores = []
    times = []

    print(f"Testing Optimized MCTS on {num_episodes} games (Test Set, seed {start_seed}-{start_seed+num_episodes-1})...")
    print()

    for episode in range(num_episodes):
        game_start = time.time()

        env.reset(seed=start_seed + episode)

        # Initial action
        action = np.random.randint(0, 16)
        state, _, alive = env.next(action)

        steps = 0
        while alive:
            # MCTS predict
            action = agent.predict(env)[0]
            state, _, alive = env.next(action)
            steps += 1

        score = env.game.score
        game_time = time.time() - game_start

        scores.append(score)
        times.append(game_time)

        print(f"Game {episode+1:3d}/100 | Score: {score:3d} | Steps: {steps:2d} | Time: {game_time:5.1f}s | Avg: {np.mean(scores):6.1f}", flush=True)

    return scores, times

if __name__ == "__main__":
    start_time = time.time()

    print("="*70)
    print("Optimized MCTS Test Set Evaluation")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize
    env = GameInterface()
    agent = FastMCTSAgent(num_simulations=2000)

    print(f"Configuration:")
    print(f"  Algorithm: Optimized MCTS (Fast version)")
    print(f"  Simulations per move: 2000")
    print(f"  Test set: seeds 1000-1099 (100 games)")
    print(f"  Merge rule: Simplified (single-pass)")
    print("="*70)
    print()

    # Test
    scores, times = test_mcts(env, agent, num_episodes=100, start_seed=1000)
    total_time = time.time() - start_time

    # Results
    print()
    print("="*70)
    print("🎯 TEST SET Results (100 games):")
    print("="*70)
    print(f"  Mean Score:      {np.mean(scores):6.1f} ± {np.std(scores):.1f}")
    print(f"  Max Score:       {max(scores):6d}")
    print(f"  Min Score:       {min(scores):6d}")
    print(f"  Median Score:    {np.median(scores):6.1f}")
    print()
    print(f"  Total Time:      {total_time:6.1f}s")
    print(f"  Avg Time/Game:   {total_time/100:6.1f}s")
    print(f"  Avg Time/Step:   ~{np.mean(times)/50:5.2f}s (estimated)")
    print("="*70)
    print()

    print("Comparison with other algorithms:")
    print(f"  Optimized MCTS: {np.mean(scores):.1f} (this test)")
    print(f"  CNN-DQN:        205.7")
    print(f"  DQN (MLP):      183.9")
    print(f"  Random:         133.5")
    print()

    # Save results
    output_file = 'optimized_mcts_test_results.txt'
    with open(output_file, 'w') as f:
        f.write(f"Optimized MCTS Test Set Results\n")
        f.write(f"="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Set: seeds 1000-1099 (100 games)\n")
        f.write(f"Simulations: 2000 per move\n")
        f.write(f"Merge Rule: Simplified (single-pass)\n")
        f.write(f"\n")
        f.write(f"Results Summary:\n")
        f.write(f"  Mean:   {np.mean(scores):.1f} ± {np.std(scores):.1f}\n")
        f.write(f"  Max:    {max(scores)}\n")
        f.write(f"  Min:    {min(scores)}\n")
        f.write(f"  Median: {np.median(scores):.1f}\n")
        f.write(f"\n")
        f.write(f"Performance:\n")
        f.write(f"  Total Time:    {total_time:.1f}s\n")
        f.write(f"  Avg Time/Game: {total_time/100:.1f}s\n")
        f.write(f"\n")
        f.write(f"All Scores:\n")
        f.write(str(scores) + "\n")

    print(f"✅ Results saved to: {output_file}")
