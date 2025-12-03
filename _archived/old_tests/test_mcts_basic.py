"""
Test Basic MCTS on Test Set
Same test set as CNN-DQN for fair comparison
"""
import sys
import numpy as np
sys.path.insert(0, '/Users/ycy/Downloads/DQN_FruitMerger')

from GameInterface import GameInterface
from mcts.MCTS import MCTSAgent

def test_mcts(env, agent, num_episodes=100, start_seed=1000):
    """
    测试集评估 - 与CNN-DQN使用相同的测试集
    """
    scores = []

    print(f"Testing Basic MCTS on {num_episodes} games (Test Set, seed {start_seed}-{start_seed+num_episodes-1})...")
    print()

    for episode in range(num_episodes):
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

        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode+1}/{num_episodes} games | Latest: {score} | Avg so far: {np.mean(scores):.1f}", flush=True)

    return scores

if __name__ == "__main__":
    print("="*60)
    print("Basic MCTS Test Set Evaluation")
    print("="*60)
    print()

    # Initialize
    env = GameInterface()
    agent = MCTSAgent(num_simulations=2000)

    print(f"Configuration:")
    print(f"  Simulations per move: 2000")
    print(f"  Test set: seeds 1000-1099 (100 games)")
    print("="*60)
    print()

    # Test
    scores = test_mcts(env, agent, num_episodes=100, start_seed=1000)

    # Results
    print()
    print("="*60)
    print("🎯 TEST SET Results (100 games):")
    print("="*60)
    print(f"  Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  Max Score:  {max(scores)}")
    print(f"  Min Score:  {min(scores)}")
    print("="*60)
    print()

    print("Comparison with other algorithms:")
    print(f"  Optimized MCTS: 255.0 (reported)")
    print(f"  CNN-DQN:        205.7")
    print(f"  DQN (MLP):      183.9")
    print(f"  Basic MCTS:     {np.mean(scores):.1f} (this test)")
