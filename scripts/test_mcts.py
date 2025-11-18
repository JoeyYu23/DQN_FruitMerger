"""
Test script for MCTS Agent
Runs the MCTS agent on the Suika game and evaluates performance.
"""

import time
import numpy as np
from GameInterface import GameInterface
from MCTS import MCTSAgent, MCTSConfig
from typing import List, Tuple


def evaluate_agent(agent, env: GameInterface, num_games: int = 10,
                   verbose: bool = True) -> Tuple[List[int], List[float]]:
    """
    Evaluate MCTS agent over multiple games.

    Args:
        agent: MCTS agent
        env: Game environment
        num_games: Number of games to play
        verbose: Print progress

    Returns:
        (scores, times): Lists of final scores and time per move
    """
    scores = []
    times_per_move = []

    for game_idx in range(num_games):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Game {game_idx + 1}/{num_games}")
            print(f"{'='*60}")

        # Reset environment
        seed = game_idx * 1000  # Deterministic for reproducibility
        env.reset(seed)

        # First random action to initialize
        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        step = 0
        total_time = 0
        game_times = []

        while alive:
            step += 1

            # Run MCTS to select action
            start_time = time.time()
            action = agent.predict(env)
            elapsed = time.time() - start_time
            game_times.append(elapsed)
            total_time += elapsed

            # Execute action
            feature, reward, alive = env.next(action[0])

            if verbose and step % 10 == 0:
                print(f"  Step {step:3d} | Score: {env.game.score:4d} | "
                      f"Time: {elapsed:.3f}s | Avg: {np.mean(game_times):.3f}s")

        final_score = env.game.score
        avg_time = total_time / step if step > 0 else 0

        scores.append(final_score)
        times_per_move.append(avg_time)

        if verbose:
            print(f"\n  Final Score: {final_score}")
            print(f"  Steps: {step}")
            print(f"  Avg time/move: {avg_time:.3f}s")

    return scores, times_per_move


def compare_agents(num_games: int = 5):
    """Compare MCTS with different simulation counts"""
    env = GameInterface()

    configs = [
        (500, "MCTS-500"),
        (1000, "MCTS-1000"),
        (2000, "MCTS-2000"),
    ]

    results = {}

    for num_sims, name in configs:
        print(f"\n{'#'*60}")
        print(f"Testing {name} ({num_sims} simulations)")
        print(f"{'#'*60}")

        agent = MCTSAgent(num_simulations=num_sims)
        scores, times = evaluate_agent(agent, env, num_games, verbose=False)

        results[name] = {
            'scores': scores,
            'times': times,
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_time': np.mean(times),
        }

        print(f"\nResults for {name}:")
        print(f"  Mean Score: {results[name]['mean_score']:.1f} ± {results[name]['std_score']:.1f}")
        print(f"  Max Score: {np.max(scores)}")
        print(f"  Min Score: {np.min(scores)}")
        print(f"  Avg Time/Move: {results[name]['mean_time']:.3f}s")
        print(f"  Rollouts/sec: {num_sims / results[name]['mean_time']:.0f}")

    return results


def profile_mcts_performance():
    """Profile MCTS to measure rollouts per second"""
    from MCTS import SimplifiedGameState, MCTS

    print("\n" + "="*60)
    print("MCTS Performance Profiling")
    print("="*60)

    state = SimplifiedGameState()
    mcts = MCTS()

    simulation_counts = [100, 500, 1000, 2000, 5000]

    for num_sims in simulation_counts:
        start = time.time()
        best_action = mcts.search(state, num_simulations=num_sims)
        elapsed = time.time() - start

        rollouts_per_sec = num_sims / elapsed
        stats = mcts.get_stats()

        print(f"\n{num_sims} simulations:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Rollouts/sec: {rollouts_per_sec:.0f}")
        print(f"  Best action: {best_action}")
        print(f"  Tree size: {stats['num_children']} children")


def demonstrate_mcts():
    """Simple demonstration of MCTS agent"""
    print("\n" + "="*60)
    print("MCTS Agent Demonstration")
    print("="*60)

    # Create environment and agent
    env = GameInterface()
    agent = MCTSAgent(num_simulations=1000)

    print("\nConfiguration:")
    print(f"  Simulations per move: 1000")
    print(f"  Grid size: {MCTSConfig.GRID_WIDTH}x{MCTSConfig.GRID_HEIGHT}")
    print(f"  Max simulation depth: {MCTSConfig.MAX_SIMULATION_DEPTH}")
    print(f"  C_PUCT: {MCTSConfig.C_PUCT}")

    # Run single game
    scores, times = evaluate_agent(agent, env, num_games=1, verbose=True)

    print(f"\n{'='*60}")
    print("Demonstration Complete!")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "demo"

    if mode == "demo":
        # Simple demonstration
        demonstrate_mcts()

    elif mode == "profile":
        # Profile performance
        profile_mcts_performance()

    elif mode == "compare":
        # Compare different configurations
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        compare_agents(num_games)

    elif mode == "eval":
        # Full evaluation
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        env = GameInterface()
        agent = MCTSAgent(num_simulations=2000)

        print("\n" + "="*60)
        print(f"Evaluating MCTS Agent ({num_games} games)")
        print("="*60)

        scores, times = evaluate_agent(agent, env, num_games, verbose=True)

        print(f"\n{'='*60}")
        print("Final Results:")
        print(f"  Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"  Max Score: {np.max(scores)}")
        print(f"  Min Score: {np.min(scores)}")
        print(f"  Avg Time/Move: {np.mean(times):.3f}s")

    else:
        print("Usage:")
        print("  python test_mcts.py demo      - Run demonstration")
        print("  python test_mcts.py profile   - Profile performance")
        print("  python test_mcts.py compare [N] - Compare configurations (N games)")
        print("  python test_mcts.py eval [N]  - Full evaluation (N games)")
