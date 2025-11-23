"""
TrainAlphaZero.py - AlphaZero训练主循环

训练流程：
1. Self-Play: 收集游戏数据
2. Train: 训练神经网络
3. Evaluate: 评估新网络
4. Update: 如果更好，替换旧网络
5. Repeat
"""

import os
import time
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from typing import List, Tuple
import json

from SuikaNet import SuikaNet
from SelfPlay import SelfPlayCollector
from GameInterface import GameInterface
from AlphaZeroMCTS import AlphaZeroMCTS
from StateConverter import StateConverter


# ============================================
# 数据集类
# ============================================

class SuikaDataset(Dataset):
    """AlphaZero训练数据集"""

    def __init__(self, data: List[Tuple]):
        """
        Args:
            data: [(state, pi, z), ...]
                state: [13, H, W] numpy数组
                pi: [16] 概率分布
                z: float 标量
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, pi, z = self.data[idx]

        # 转为tensor
        state_tensor = paddle.to_tensor(state, dtype='float32')
        pi_tensor = paddle.to_tensor(pi, dtype='float32')
        z_tensor = paddle.to_tensor([z], dtype='float32')

        return state_tensor, pi_tensor, z_tensor


# ============================================
# 训练器类
# ============================================

class AlphaZeroTrainer:
    """AlphaZero训练器"""

    def __init__(self,
                 network: SuikaNet,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 batch_size: int = 32,
                 epochs_per_iteration: int = 5,
                 checkpoint_dir: str = "weights/alphazero"):
        """
        Args:
            network: SuikaNet神经网络
            learning_rate: 学习率
            weight_decay: L2正则化系数
            batch_size: 批量大小
            epochs_per_iteration: 每轮迭代训练几个epoch
            checkpoint_dir: 检查点保存目录
        """
        self.network = network
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.checkpoint_dir = checkpoint_dir

        # 创建优化器
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=learning_rate,
            parameters=network.parameters(),
            weight_decay=weight_decay
        )

        # 确保目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 训练历史
        self.history = {
            'iterations': [],
            'train_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'eval_scores': []
        }

    def train_on_data(self, data: List[Tuple], verbose: bool = True) -> dict:
        """
        在数据上训练网络

        Args:
            data: [(state, pi, z), ...]
            verbose: 是否打印训练信息

        Returns:
            metrics: {'loss': ..., 'policy_loss': ..., 'value_loss': ...}
        """
        dataset = SuikaDataset(data)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False
        )

        self.network.train()

        epoch_metrics = []

        for epoch in range(self.epochs_per_iteration):
            total_loss = 0.0
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0

            for states, pis, zs in dataloader:
                # Forward
                pred_policy, pred_value = self.network(states)

                # Loss计算
                # 1. Value loss: MSE
                value_loss = F.mse_loss(pred_value.squeeze(-1), zs.squeeze(-1))

                # 2. Policy loss: Cross-entropy
                # pi * log(P)
                policy_loss = -(pis * paddle.log(pred_policy + 1e-8)).sum(axis=1).mean()

                # 3. Total loss
                loss = value_loss + policy_loss

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                # 统计
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                num_batches += 1

            # Epoch统计
            avg_loss = total_loss / num_batches
            avg_v_loss = total_value_loss / num_batches
            avg_p_loss = total_policy_loss / num_batches

            epoch_metrics.append({
                'loss': avg_loss,
                'policy_loss': avg_p_loss,
                'value_loss': avg_v_loss
            })

            if verbose:
                print(f"  Epoch {epoch+1}/{self.epochs_per_iteration}: "
                      f"Loss={avg_loss:.4f} "
                      f"(Policy={avg_p_loss:.4f}, Value={avg_v_loss:.4f})")

        # 返回最后一个epoch的metrics
        return epoch_metrics[-1]

    def save_checkpoint(self, iteration: int, metrics: dict = None):
        """保存检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"iter_{iteration}.pdparams")
        paddle.save(self.network.state_dict(), checkpoint_path)

        # 保存训练历史
        if metrics:
            self.history['iterations'].append(iteration)
            self.history['train_losses'].append(metrics.get('loss', 0))
            self.history['policy_losses'].append(metrics.get('policy_loss', 0))
            self.history['value_losses'].append(metrics.get('value_loss', 0))

        history_path = os.path.join(self.checkpoint_dir, "history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, iteration: int):
        """加载检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"iter_{iteration}.pdparams")
        self.network.set_state_dict(paddle.load(checkpoint_path))
        print(f"  Checkpoint loaded: {checkpoint_path}")


# ============================================
# 评估器
# ============================================

def evaluate_network(network: SuikaNet,
                    num_games: int = 10,
                    num_simulations: int = 200,
                    verbose: bool = True) -> dict:
    """
    评估网络性能

    Args:
        network: 要评估的网络
        num_games: 测试游戏局数
        num_simulations: MCTS模拟次数
        verbose: 是否打印信息

    Returns:
        metrics: {'mean_score': ..., 'max_score': ..., 'std_score': ...}
    """
    if verbose:
        print(f"  Evaluating network on {num_games} games...")

    mcts = AlphaZeroMCTS(
        network=network,
        num_simulations=num_simulations,
        temperature=0.0,  # 确定性选择
        add_dirichlet_noise=False
    )

    # 使用网络的尺寸创建converter
    converter = StateConverter(
        grid_height=network.board_h,
        grid_width=network.board_w,
        feature_height=network.board_h,
        feature_width=network.board_w
    )
    scores = []

    for game_idx in range(num_games):
        game = GameInterface()
        game.reset(seed=1000 + game_idx)  # 固定种子用于评估

        while game.game.alive:
            simplified_state = converter.game_to_simplified(game)
            action = mcts.get_action(simplified_state)
            _, _, alive = game.next(action)

        scores.append(game.game.score)

    metrics = {
        'mean_score': np.mean(scores),
        'max_score': np.max(scores),
        'std_score': np.std(scores)
    }

    if verbose:
        print(f"  Evaluation: Mean={metrics['mean_score']:.1f} ± {metrics['std_score']:.1f}, "
              f"Max={metrics['max_score']:.1f}")

    return metrics


# ============================================
# 主训练循环
# ============================================

def train_alphazero(num_iterations: int = 20,
                   games_per_iteration: int = 50,
                   mcts_simulations: int = 200,
                   batch_size: int = 32,
                   epochs_per_iteration: int = 5,
                   eval_games: int = 10,
                   checkpoint_dir: str = "weights/alphazero",
                   resume_from: int = None):
    """
    AlphaZero主训练循环

    Args:
        num_iterations: 总迭代次数
        games_per_iteration: 每轮迭代收集的游戏局数
        mcts_simulations: MCTS每步模拟次数
        batch_size: 训练批量大小
        epochs_per_iteration: 每轮迭代训练几个epoch
        eval_games: 评估游戏局数
        checkpoint_dir: 检查点保存目录
        resume_from: 从哪个迭代恢复 (None表示从头开始)
    """
    print("="*60)
    print("AlphaZero Training for Suika Game")
    print("="*60)
    print(f"Configuration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Games per iteration: {games_per_iteration}")
    print(f"  MCTS simulations: {mcts_simulations}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per iteration: {epochs_per_iteration}")
    print("="*60)

    # 创建网络
    network = SuikaNet(input_channels=13, num_actions=16, hidden_channels=64)

    # 创建训练器
    trainer = AlphaZeroTrainer(
        network=network,
        learning_rate=0.001,
        weight_decay=1e-4,
        batch_size=batch_size,
        epochs_per_iteration=epochs_per_iteration,
        checkpoint_dir=checkpoint_dir
    )

    # 恢复训练
    start_iteration = 0
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
        start_iteration = resume_from

    # 训练循环
    for iteration in range(start_iteration, num_iterations):
        iter_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # ========================================
        # 1. Self-Play: 收集数据
        # ========================================
        print(f"\n[1/3] Self-Play: Collecting {games_per_iteration} games...")
        collector = SelfPlayCollector(
            network=network,
            num_simulations=mcts_simulations,
            temperature=1.0,
            temperature_threshold=30
        )

        data = collector.collect_episodes(
            num_episodes=games_per_iteration,
            start_seed=iteration * 1000,
            verbose=True
        )

        print(f"  Collected {len(data)} training samples")

        # ========================================
        # 2. Train: 训练网络
        # ========================================
        print(f"\n[2/3] Training network...")
        metrics = trainer.train_on_data(data, verbose=True)

        # ========================================
        # 3. Evaluate: 评估网络
        # ========================================
        print(f"\n[3/3] Evaluating network...")
        eval_metrics = evaluate_network(
            network=network,
            num_games=eval_games,
            num_simulations=mcts_simulations,
            verbose=True
        )

        # 保存评估结果到历史
        trainer.history['eval_scores'].append(eval_metrics['mean_score'])

        # ========================================
        # 4. Save: 保存检查点
        # ========================================
        trainer.save_checkpoint(iteration + 1, metrics)

        # 打印迭代总结
        iter_time = time.time() - iter_start_time
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1} Summary:")
        print(f"  Training Loss: {metrics['loss']:.4f}")
        print(f"  Eval Score: {eval_metrics['mean_score']:.1f} ± {eval_metrics['std_score']:.1f}")
        print(f"  Time: {iter_time:.1f}s")
        print(f"{'='*60}")

    print("\n" + "="*60)
    print("Training Completed!")
    print("="*60)


# ============================================
# 入口
# ============================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train AlphaZero for Suika Game')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations')
    parser.add_argument('--games', type=int, default=50, help='Games per iteration')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs per iteration')
    parser.add_argument('--eval-games', type=int, default=10, help='Evaluation games')
    parser.add_argument('--checkpoint-dir', type=str, default='weights/alphazero', help='Checkpoint directory')
    parser.add_argument('--resume', type=int, default=None, help='Resume from iteration')

    args = parser.parse_args()

    train_alphazero(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume
    )
