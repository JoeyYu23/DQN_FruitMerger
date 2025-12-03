#!/usr/bin/env python3
"""
统一的训练脚本 - 支持本地和云端训练
可以选择训练AlphaZero或查看训练结果
"""

import argparse
import os
import sys


def run_alphazero_training(args):
    """运行AlphaZero训练"""
    print("\n" + "="*70)
    print("  开始 AlphaZero 训练")
    print("="*70)

    from TrainAlphaZero import train_alphazero

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


def run_visualization(args):
    """可视化训练结果"""
    print("\n" + "="*70)
    print("  可视化训练结果")
    print("="*70)

    from visualize_results import visualize_training_history

    visualize_training_history(
        checkpoint_dir=args.checkpoint_dir,
        save_path=args.save_path
    )


def run_evaluation(args):
    """评估模型性能"""
    print("\n" + "="*70)
    print("  评估模型性能")
    print("="*70)

    from evaluate_model import evaluate_alphazero

    evaluate_alphazero(
        model_path=args.model_path,
        num_games=args.num_games,
        simulations=args.simulations,
        visualize=args.visualize
    )


def main():
    parser = argparse.ArgumentParser(
        description='合成大西瓜游戏 - AlphaZero训练和评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 快速测试训练 (本地):
   python run_training.py train --quick

2. 完整训练 (云端推荐):
   python run_training.py train --iterations 20 --games 50 --simulations 200

3. 可视化训练历史:
   python run_training.py visualize

4. 评估模型性能:
   python run_training.py evaluate --model-path weights/alphazero/iter_20.pdparams

5. 评估并可视化游戏过程:
   python run_training.py evaluate --model-path weights/alphazero/iter_20.pdparams --visualize
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='命令')

    # ==================== 训练命令 ====================
    train_parser = subparsers.add_parser('train', help='训练AlphaZero模型')
    train_parser.add_argument('--quick', action='store_true',
                             help='快速测试模式 (2轮迭代, 适合本地测试)')
    train_parser.add_argument('--iterations', type=int, default=20,
                             help='训练迭代次数 (默认: 20)')
    train_parser.add_argument('--games', type=int, default=50,
                             help='每轮迭代的游戏局数 (默认: 50)')
    train_parser.add_argument('--simulations', type=int, default=200,
                             help='MCTS每步模拟次数 (默认: 200)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='训练批量大小 (默认: 32)')
    train_parser.add_argument('--epochs', type=int, default=5,
                             help='每轮迭代训练epoch数 (默认: 5)')
    train_parser.add_argument('--eval-games', type=int, default=10,
                             help='评估游戏局数 (默认: 10)')
    train_parser.add_argument('--checkpoint-dir', type=str,
                             default='weights/alphazero',
                             help='检查点保存目录')
    train_parser.add_argument('--resume', type=int, default=None,
                             help='从指定迭代恢复训练')

    # ==================== 可视化命令 ====================
    viz_parser = subparsers.add_parser('visualize', help='可视化训练历史')
    viz_parser.add_argument('--checkpoint-dir', type=str,
                           default='weights/alphazero',
                           help='检查点目录')
    viz_parser.add_argument('--save-path', type=str,
                           default='training_visualization.png',
                           help='保存图片路径')

    # ==================== 评估命令 ====================
    eval_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    eval_parser.add_argument('--model-path', type=str, required=True,
                            help='模型权重路径')
    eval_parser.add_argument('--num-games', type=int, default=20,
                            help='评估游戏局数')
    eval_parser.add_argument('--simulations', type=int, default=200,
                            help='MCTS模拟次数')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='可视化游戏过程')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # 处理快速模式
    if args.command == 'train' and args.quick:
        print("\n⚡ 快速测试模式启动")
        args.iterations = 2
        args.games = 10
        args.simulations = 50
        args.batch_size = 16
        args.epochs = 3
        args.eval_games = 5

    # 执行命令
    if args.command == 'train':
        run_alphazero_training(args)
    elif args.command == 'visualize':
        run_visualization(args)
    elif args.command == 'evaluate':
        run_evaluation(args)


if __name__ == '__main__':
    main()
