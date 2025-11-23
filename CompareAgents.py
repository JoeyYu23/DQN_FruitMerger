"""
CompareAgents.py - 评估和对比不同agents

支持的agents:
1. Random - 随机策略
2. DQN - Deep Q-Network
3. AlphaZero - MCTS + 神经网络
4. MCTS (原始) - 启发式MCTS
"""

import numpy as np
import paddle
import time
from typing import List, Dict
import json
import os

from GameInterface import GameInterface
from DQN import Agent as DQNAgent
from SuikaNet import SuikaNet
from AlphaZeroMCTS import AlphaZeroMCTS
from StateConverter import StateConverter
from mcts.MCTS import MCTS, MCTSConfig


class AgentEvaluator:
    """Agent评估器"""

    def __init__(self, num_games: int = 50, seed_start: int = 10000):
        """
        Args:
            num_games: 评估游戏局数
            seed_start: 起始随机种子
        """
        self.num_games = num_games
        self.seed_start = seed_start
        self.converter = StateConverter()

    def evaluate_random(self, verbose: bool = True) -> Dict:
        """评估随机agent"""
        if verbose:
            print("\n[Evaluating Random Agent]")

        scores = []
        steps = []

        for game_idx in range(self.num_games):
            game = GameInterface()
            game.reset(seed=self.seed_start + game_idx)

            step_count = 0
            while game.game.alive:
                action = np.random.randint(0, 16)
                _, _, alive = game.next(action)
                step_count += 1

            scores.append(game.game.score)
            steps.append(step_count)

            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  {game_idx + 1}/{self.num_games} games completed...")

        return self._compute_metrics(scores, steps, "Random")

    def evaluate_dqn(self, model_path: str, verbose: bool = True) -> Dict:
        """评估DQN agent"""
        if verbose:
            print(f"\n[Evaluating DQN Agent]")
            print(f"  Loading model from: {model_path}")

        # 加载DQN模型
        agent = DQNAgent(640, 16)  # feature_size=640, action_num=16
        if os.path.exists(model_path):
            agent.load_model(model_path)
        else:
            print(f"  Warning: Model not found, using random DQN")

        scores = []
        steps = []

        for game_idx in range(self.num_games):
            game = GameInterface()
            game.reset(seed=self.seed_start + game_idx)

            step_count = 0
            while game.game.alive:
                # 获取特征
                features = game.game.get_features(16, 20)
                feature_flat = features.flatten().astype(np.float32)

                # DQN选择动作
                action = agent.predict(feature_flat)
                if hasattr(action, '__len__'):
                    action = action[0]

                _, _, alive = game.next(action)
                step_count += 1

            scores.append(game.game.score)
            steps.append(step_count)

            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  {game_idx + 1}/{self.num_games} games completed...")

        return self._compute_metrics(scores, steps, "DQN")

    def evaluate_alphazero(self,
                          model_path: str,
                          num_simulations: int = 200,
                          verbose: bool = True) -> Dict:
        """评估AlphaZero agent"""
        if verbose:
            print(f"\n[Evaluating AlphaZero Agent]")
            print(f"  Loading model from: {model_path}")
            print(f"  MCTS simulations: {num_simulations}")

        # 加载网络
        network = SuikaNet(input_channels=13, num_actions=16)
        if os.path.exists(model_path):
            network.set_state_dict(paddle.load(model_path))
        else:
            print(f"  Warning: Model not found, using random network")

        # 创建MCTS
        mcts = AlphaZeroMCTS(
            network=network,
            num_simulations=num_simulations,
            temperature=0.0,  # 确定性选择
            add_dirichlet_noise=False
        )

        scores = []
        steps = []

        for game_idx in range(self.num_games):
            game = GameInterface()
            game.reset(seed=self.seed_start + game_idx)

            step_count = 0
            while game.game.alive:
                # 转换状态
                simplified_state = self.converter.game_to_simplified(game)

                # AlphaZero选择动作
                action = mcts.get_action(simplified_state)

                _, _, alive = game.next(action)
                step_count += 1

            scores.append(game.game.score)
            steps.append(step_count)

            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  {game_idx + 1}/{self.num_games} games completed...")

        return self._compute_metrics(scores, steps, f"AlphaZero (sim={num_simulations})")

    def evaluate_mcts_baseline(self,
                               num_simulations: int = 200,
                               verbose: bool = True) -> Dict:
        """评估原始启发式MCTS"""
        if verbose:
            print(f"\n[Evaluating Baseline MCTS]")
            print(f"  MCTS simulations: {num_simulations}")

        from mcts.MCTS import MCTSAgent

        # 创建MCTS agent
        config = MCTSConfig()
        config.NUM_SIMULATIONS = num_simulations
        mcts_agent = MCTSAgent(config)

        scores = []
        steps = []

        for game_idx in range(self.num_games):
            game = GameInterface()
            game.reset(seed=self.seed_start + game_idx)

            step_count = 0
            while game.game.alive:
                # 获取特征
                features = game.game.get_features(config.GRID_WIDTH, config.GRID_HEIGHT)

                # MCTS选择动作
                grid_action = mcts_agent.get_action(
                    features,
                    game.game.current_fruit_type,
                    game.game.score
                )

                # 映射到game action
                action = int(grid_action * 16 / config.GRID_WIDTH)

                _, _, alive = game.next(action)
                step_count += 1

            scores.append(game.game.score)
            steps.append(step_count)

            if verbose and (game_idx + 1) % 10 == 0:
                print(f"  {game_idx + 1}/{self.num_games} games completed...")

        return self._compute_metrics(scores, steps, f"MCTS Baseline (sim={num_simulations})")

    def _compute_metrics(self, scores: List[float], steps: List[int], name: str) -> Dict:
        """计算评估指标"""
        metrics = {
            'name': name,
            'num_games': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': float(np.max(scores)),
            'min_score': float(np.min(scores)),
            'median_score': float(np.median(scores)),
            'mean_steps': float(np.mean(steps)),
            'std_steps': float(np.std(steps)),
            'scores': scores,
            'steps': steps
        }
        return metrics


def compare_agents(num_games: int = 50,
                  dqn_model: str = None,
                  alphazero_model: str = None,
                  alphazero_sims: int = 200,
                  mcts_sims: int = 200,
                  output_file: str = "evaluation_results.json"):
    """
    对比所有agents

    Args:
        num_games: 每个agent测试的游戏局数
        dqn_model: DQN模型路径
        alphazero_model: AlphaZero模型路径
        alphazero_sims: AlphaZero的MCTS模拟次数
        mcts_sims: 基线MCTS的模拟次数
        output_file: 结果保存文件
    """
    print("="*70)
    print("Agent Performance Comparison")
    print("="*70)
    print(f"Configuration:")
    print(f"  Number of games: {num_games}")
    print(f"  AlphaZero simulations: {alphazero_sims}")
    print(f"  Baseline MCTS simulations: {mcts_sims}")
    print("="*70)

    evaluator = AgentEvaluator(num_games=num_games)
    results = {}

    # 1. Random
    results['random'] = evaluator.evaluate_random(verbose=True)

    # 2. DQN (如果提供)
    if dqn_model:
        results['dqn'] = evaluator.evaluate_dqn(dqn_model, verbose=True)

    # 3. AlphaZero (如果提供)
    if alphazero_model:
        results['alphazero'] = evaluator.evaluate_alphazero(
            alphazero_model,
            num_simulations=alphazero_sims,
            verbose=True
        )

    # 4. Baseline MCTS
    try:
        results['mcts_baseline'] = evaluator.evaluate_mcts_baseline(
            num_simulations=mcts_sims,
            verbose=True
        )
    except Exception as e:
        print(f"  Error evaluating baseline MCTS: {e}")

    # 打印结果
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"{'Agent':<25} {'Mean Score':<15} {'Max Score':<15} {'Mean Steps':<15}")
    print("-"*70)

    for agent_name, metrics in results.items():
        print(f"{metrics['name']:<25} "
              f"{metrics['mean_score']:<15.1f} "
              f"{metrics['max_score']:<15.1f} "
              f"{metrics['mean_steps']:<15.1f}")

    print("="*70)

    # 保存结果
    # 移除scores和steps列表（太大）用于json保存
    results_summary = {}
    for agent_name, metrics in results.items():
        summary = {k: v for k, v in metrics.items() if k not in ['scores', 'steps']}
        results_summary[agent_name] = summary

    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


# ============================================
# 入口
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate and compare agents')
    parser.add_argument('--num-games', type=int, default=50, help='Number of games per agent')
    parser.add_argument('--dqn-model', type=str, default=None, help='Path to DQN model')
    parser.add_argument('--alphazero-model', type=str, default=None, help='Path to AlphaZero model')
    parser.add_argument('--alphazero-sims', type=int, default=200, help='AlphaZero MCTS simulations')
    parser.add_argument('--mcts-sims', type=int, default=200, help='Baseline MCTS simulations')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file')

    args = parser.parse_args()

    # 自动查找模型
    if args.alphazero_model is None:
        # 尝试找最新的AlphaZero模型
        az_dir = "weights/alphazero"
        if os.path.exists(az_dir):
            models = [f for f in os.listdir(az_dir) if f.endswith('.pdparams')]
            if models:
                # 按迭代数排序，取最大的
                models_sorted = sorted(models, key=lambda x: int(x.split('_')[1].split('.')[0]))
                args.alphazero_model = os.path.join(az_dir, models_sorted[-1])
                print(f"Auto-detected AlphaZero model: {args.alphazero_model}")

    if args.dqn_model is None:
        # 尝试找DQN模型
        dqn_paths = ["weights/final.pdparams", "final.pdparams"]
        for path in dqn_paths:
            if os.path.exists(path):
                args.dqn_model = path
                print(f"Auto-detected DQN model: {args.dqn_model}")
                break

    compare_agents(
        num_games=args.num_games,
        dqn_model=args.dqn_model,
        alphazero_model=args.alphazero_model,
        alphazero_sims=args.alphazero_sims,
        mcts_sims=args.mcts_sims,
        output_file=args.output
    )
