"""
Test Basic MCTS on Test Set - Fast Version
先测20局，看速度和效果
"""
import sys
import numpy as np
import time
sys.path.insert(0, '/Users/ycy/Downloads/DQN_FruitMerger')

from GameInterface import GameInterface
from mcts.MCTS import MCTSAgent

def test_mcts(env, agent, num_episodes=20, start_seed=1000):
    """
    测试集评估 - 快速版本
    """
    scores = []
    print(f"Testing Basic MCTS on {num_episodes} games (Test Set, seed {start_seed}-{start_seed+num_episodes-1})...")
    print()

    for episode in range(num_episodes):
        episode_start = time.time()

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
        scores.append(score)

        episode_time = time.time() - episode_start

        print(f"  Game {episode+1}/{num_episodes} | Score: {score:3d} | Steps: {steps:2d} | Time: {episode_time:.1f}s | Avg: {np.mean(scores):.1f}", flush=True)

    return scores

if __name__ == "__main__":
    print("="*60)
    print("Basic MCTS Test Set Evaluation (Fast)")
    print("="*60)
    print()

    # Initialize
    env = GameInterface()
    agent = MCTSAgent(num_simulations=500)  # 降低到500

    print(f"Configuration:")
    print(f"  Simulations per move: 500 (fast)")
    print(f"  Test set: seeds 1000-1019 (20 games)")
    print("="*60)
    print()

    # Test
    start = time.time()
    scores = test_mcts(env, agent, num_episodes=20, start_seed=1000)
    total_time = time.time() - start

    # Results
    print()
    print("="*60)
    print("🎯 TEST SET Results (20 games):")
    print("="*60)
    print(f"  Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  Max Score:  {max(scores)}")
    print(f"  Min Score:  {min(scores)}")
    print(f"  Total Time: {total_time:.1f}s ({total_time/20:.1f}s per game)")
    print("="*60)
    print()

    print("Comparison with other algorithms:")
    print(f"  Optimized MCTS: 255.0 (reported)")
    print(f"  CNN-DQN:        205.7")
    print(f"  DQN (MLP):      183.9")
    print(f"  Basic MCTS:     {np.mean(scores):.1f} (this test, 500 sims)")
    print()

    # Save results
    with open('mcts_basic_fast_results.txt', 'w') as f:
        f.write(f"Basic MCTS Test Results (500 simulations)\n")
        f.write(f"="*60 + "\n")
        f.write(f"Test Set: seeds 1000-1019 (20 games)\n")
        f.write(f"Mean: {np.mean(scores):.1f} ± {np.std(scores):.1f}\n")
        f.write(f"Max: {max(scores)}\n")
        f.write(f"Min: {min(scores)}\n")
        f.write(f"Total Time: {total_time:.1f}s\n")
        f.write(f"\nAll Scores:\n")
        f.write(str(scores) + "\n")

    print("✅ Results saved to: mcts_basic_fast_results.txt")
