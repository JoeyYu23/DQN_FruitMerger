#!/usr/bin/env python3
"""
Unified Benchmark Script for All Agents
Compare DQN, MCTS variants, and Random baseline
"""

import numpy as np
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
import json

# Import agents
from GameInterface import GameInterface
from DQN import Agent, RandomAgent, build_model
import paddle

# Import MCTS variants
sys.path.insert(0, 'mcts')
try:
    from mcts.MCTS_optimized import FastMCTSAgent
    from mcts.MCTS_advanced import SmartMCTSAgent
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    print("⚠️  MCTS modules not found in mcts/ folder")


class BenchmarkRunner:
    """Unified benchmark runner for all agents"""

    def __init__(self, num_episodes: int = 100, seeds: List[int] = None):
        self.num_episodes = num_episodes
        if seeds is None:
            self.seeds = list(range(num_episodes))
        else:
            self.seeds = seeds

        self.env = GameInterface()
        self.results = {}

    def evaluate_agent(self, agent, agent_name: str, use_env: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single agent

        Args:
            agent: The agent to evaluate
            agent_name: Name for display
            use_env: If True, pass env to predict(); else pass feature

        Returns:
            Dictionary of performance metrics
        """
        print(f"\n{'='*70}")
        print(f"🤖 Evaluating: {agent_name}")
        print(f"{'='*70}")

        scores = []
        rewards = []
        steps = []
        times = []

        start_time = time.time()

        for i, seed in enumerate(self.seeds, 1):
            self.env.reset(seed=seed)

            step_count = 0
            reward_sum = 0
            episode_time = 0

            # Random first action
            action = np.random.randint(0, self.env.action_num)
            feature, _, alive = self.env.next(action)

            while alive:
                step_count += 1

                # Predict action
                step_start = time.time()
                if use_env:
                    # For MCTS agents that need full environment
                    action = agent.predict(self.env)
                    if isinstance(action, (list, np.ndarray)):
                        action = action[0]
                else:
                    # For DQN/Random agents that use features
                    action = agent.predict(feature)
                    if isinstance(action, np.ndarray):
                        action = action.item()

                step_time = time.time() - step_start
                episode_time += step_time

                # Execute action
                feature, reward, alive = self.env.next(action)
                reward_sum += np.sum(reward)

            # Record results
            scores.append(self.env.game.score)
            rewards.append(reward_sum)
            steps.append(step_count)
            times.append(episode_time / step_count if step_count > 0 else 0)

            # Progress update
            if i % 10 == 0 or i == len(self.seeds):
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = avg_time * (len(self.seeds) - i)
                print(f"  Progress: {i}/{len(self.seeds)} | "
                      f"Last score: {self.env.game.score} | "
                      f"ETA: {remaining:.1f}s")

        total_time = time.time() - start_time

        # Compile statistics
        results = {
            'agent_name': agent_name,
            'scores': np.array(scores),
            'rewards': np.array(rewards),
            'steps': np.array(steps),
            'times': np.array(times),
            'total_time': total_time,
            'num_episodes': len(self.seeds)
        }

        self._print_statistics(results)
        return results

    def _print_statistics(self, results: Dict[str, Any]):
        """Print detailed statistics for an agent"""
        scores = results['scores']
        rewards = results['rewards']
        steps = results['steps']
        times = results['times']

        print(f"\n📊 Statistics:")
        print(f"  Score   - Mean: {np.mean(scores):6.1f} ± {np.std(scores):5.1f} | "
              f"Max: {np.max(scores):4.0f} | Min: {np.min(scores):4.0f} | "
              f"Median: {np.median(scores):6.1f}")
        print(f"  Reward  - Mean: {np.mean(rewards):6.1f} ± {np.std(rewards):5.1f} | "
              f"Max: {np.max(rewards):4.0f} | Min: {np.min(rewards):4.0f}")
        print(f"  Steps   - Mean: {np.mean(steps):6.1f} ± {np.std(steps):5.1f} | "
              f"Max: {np.max(steps):4.0f} | Min: {np.min(steps):4.0f}")
        print(f"  Time    - Avg: {np.mean(times):.4f}s/step | "
              f"Total: {results['total_time']:.1f}s")

    def compare_results(self):
        """Compare all evaluated agents"""
        if len(self.results) < 2:
            print("\n⚠️  Need at least 2 agents to compare")
            return

        print(f"\n{'='*70}")
        print(f"📊 COMPARATIVE ANALYSIS")
        print(f"{'='*70}")

        # Sort by mean score
        sorted_agents = sorted(
            self.results.items(),
            key=lambda x: np.mean(x[1]['scores']),
            reverse=True
        )

        # Summary table
        print(f"\n{'Agent':<20} {'Mean Score':<12} {'Std Dev':<10} {'Max':<8} {'Time/Step':<12}")
        print(f"{'-'*70}")
        for name, data in sorted_agents:
            mean_score = np.mean(data['scores'])
            std_score = np.std(data['scores'])
            max_score = np.max(data['scores'])
            avg_time = np.mean(data['times'])
            print(f"{name:<20} {mean_score:>6.1f} ± {std_score:<4.1f} "
                  f"{std_score:<10.1f} {max_score:<8.0f} {avg_time:<12.6f}s")

        # Head-to-head win rates
        print(f"\n🏆 Win Rate Matrix (row vs column):")
        agents = list(self.results.keys())
        print(f"{'':20}", end='')
        for name in agents:
            print(f"{name[:12]:>12}", end='')
        print()

        for name1 in agents:
            print(f"{name1[:20]:<20}", end='')
            scores1 = self.results[name1]['scores']
            for name2 in agents:
                if name1 == name2:
                    print(f"{'---':>12}", end='')
                else:
                    scores2 = self.results[name2]['scores']
                    wins = np.sum(scores1 > scores2)
                    win_rate = wins / len(scores1) * 100
                    print(f"{win_rate:>11.1f}%", end='')
            print()

        # Statistical comparison to baseline (first agent or Random)
        baseline_name = 'Random' if 'Random' in self.results else sorted_agents[-1][0]
        baseline_scores = self.results[baseline_name]['scores']

        print(f"\n📈 Improvement over {baseline_name}:")
        for name, data in sorted_agents:
            if name == baseline_name:
                continue
            scores = data['scores']
            improvement = (np.mean(scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100
            wins = np.sum(scores > baseline_scores)
            win_rate = wins / len(scores) * 100

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(scores) + np.var(baseline_scores)) / 2)
            cohens_d = (np.mean(scores) - np.mean(baseline_scores)) / pooled_std

            print(f"  {name:<20}: {improvement:>+6.1f}% | "
                  f"Win rate: {win_rate:>5.1f}% | "
                  f"Cohen's d: {cohens_d:>5.2f}")

    def save_results(self, filename: str = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for name, data in self.results.items():
            save_data[name] = {
                'agent_name': data['agent_name'],
                'scores': data['scores'].tolist(),
                'rewards': data['rewards'].tolist(),
                'steps': data['steps'].tolist(),
                'times': data['times'].tolist(),
                'total_time': data['total_time'],
                'num_episodes': data['num_episodes'],
                'statistics': {
                    'mean_score': float(np.mean(data['scores'])),
                    'std_score': float(np.std(data['scores'])),
                    'max_score': float(np.max(data['scores'])),
                    'min_score': float(np.min(data['scores'])),
                    'median_score': float(np.median(data['scores'])),
                }
            }

        save_data['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'num_episodes': self.num_episodes,
            'seeds': self.seeds,
        }

        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    def export_latex_table(self, filename: str = "benchmark_table.tex"):
        """Export results as LaTeX table"""
        with open(filename, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{@{}lccccc@{}}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Agent} & \\textbf{Mean Score} & \\textbf{Std Dev} & "
                   "\\textbf{Max} & \\textbf{Median} & \\textbf{Time/Step} \\\\\n")
            f.write("\\midrule\n")

            sorted_agents = sorted(
                self.results.items(),
                key=lambda x: np.mean(x[1]['scores']),
                reverse=True
            )

            for name, data in sorted_agents:
                mean = np.mean(data['scores'])
                std = np.std(data['scores'])
                max_score = np.max(data['scores'])
                median = np.median(data['scores'])
                avg_time = np.mean(data['times'])

                f.write(f"{name} & ${mean:.1f} \\pm {std:.1f}$ & {std:.1f} & "
                       f"{max_score:.0f} & {median:.1f} & {avg_time:.4f}s \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Benchmark results ({self.num_episodes} episodes)}}\n")
            f.write("\\end{table}\n")

        print(f"📄 LaTeX table exported to: {filename}")


def main():
    """Main benchmark execution"""
    print("="*70)
    print("🎮 UNIFIED FRUIT MERGER BENCHMARK")
    print("="*70)

    # Configuration
    NUM_EPISODES = 100  # Change this for more/fewer episodes

    print(f"\nConfiguration:")
    print(f"  Episodes: {NUM_EPISODES}")
    print(f"  Seeds: 0 to {NUM_EPISODES-1}")

    # Initialize benchmark
    benchmark = BenchmarkRunner(num_episodes=NUM_EPISODES)

    # Initialize environment dimensions
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    # 1. Random Agent (Baseline)
    print("\n" + "="*70)
    print("1️⃣  Loading Random Agent (Baseline)")
    print("="*70)
    random_agent = RandomAgent(action_dim)
    benchmark.results['Random'] = benchmark.evaluate_agent(
        random_agent, 'Random', use_env=False
    )

    # 2. DQN Agent
    print("\n" + "="*70)
    print("2️⃣  Loading DQN Agent")
    print("="*70)

    if os.path.exists("final.pdparams"):
        dqn_agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
        dqn_agent.policy_net.set_state_dict(paddle.load("final.pdparams"))
        print("✅ DQN model loaded from final.pdparams")

        benchmark.results['DQN'] = benchmark.evaluate_agent(
            dqn_agent, 'DQN', use_env=False
        )
    else:
        print("⚠️  final.pdparams not found, skipping DQN")

    # 3. MCTS Agents
    if MCTS_AVAILABLE:
        # Fast MCTS
        print("\n" + "="*70)
        print("3️⃣  Loading Fast MCTS Agent")
        print("="*70)
        fast_mcts = FastMCTSAgent(num_simulations=100)
        benchmark.results['Fast MCTS'] = benchmark.evaluate_agent(
            fast_mcts, 'Fast MCTS', use_env=True
        )

        # Smart MCTS
        print("\n" + "="*70)
        print("4️⃣  Loading Smart MCTS Agent")
        print("="*70)
        smart_mcts = SmartMCTSAgent(num_simulations=100)
        benchmark.results['Smart MCTS'] = benchmark.evaluate_agent(
            smart_mcts, 'Smart MCTS', use_env=True
        )
    else:
        print("\n⚠️  MCTS agents not available")

    # Compare all results
    benchmark.compare_results()

    # Save results
    benchmark.save_results()
    benchmark.export_latex_table()

    print("\n" + "="*70)
    print("✅ BENCHMARK COMPLETE")
    print("="*70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Allow specifying number of episodes from command line
        NUM_EPISODES = int(sys.argv[1])
        print(f"\nCustom configuration: {NUM_EPISODES} episodes")

    main()
