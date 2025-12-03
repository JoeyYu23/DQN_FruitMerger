#!/usr/bin/env python3
"""
可视化AlphaZero训练结果
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def visualize_training_history(checkpoint_dir: str = "weights/alphazero",
                               save_path: Optional[str] = None,
                               show: bool = True):
    """
    可视化训练历史

    Args:
        checkpoint_dir: 检查点目录
        save_path: 保存图片路径 (None则不保存)
        show: 是否显示图片
    """
    history_path = os.path.join(checkpoint_dir, "history.json")

    if not os.path.exists(history_path):
        print(f"❌ 找不到历史文件: {history_path}")
        print(f"   请先运行训练: python run_training.py train")
        return

    # 读取历史数据
    with open(history_path, 'r') as f:
        history = json.load(f)

    # 检查数据
    if not history.get('iterations'):
        print("❌ 历史数据为空")
        return

    iterations = history['iterations']
    train_losses = history.get('train_losses', [])
    policy_losses = history.get('policy_losses', [])
    value_losses = history.get('value_losses', [])
    eval_scores = history.get('eval_scores', [])

    print(f"\n📊 训练历史统计:")
    print(f"   总迭代次数: {len(iterations)}")
    print(f"   最终训练Loss: {train_losses[-1]:.4f}" if train_losses else "   无训练Loss数据")
    print(f"   最终评估分数: {eval_scores[-1]:.1f}" if eval_scores else "   无评估分数数据")

    if eval_scores:
        print(f"   最高评估分数: {max(eval_scores):.1f}")
        print(f"   平均评估分数: {np.mean(eval_scores):.1f}")

    # 创建图表
    fig = plt.figure(figsize=(15, 10))

    # 2行2列布局
    # 1. 训练Loss
    ax1 = plt.subplot(2, 2, 1)
    if train_losses:
        ax1.plot(iterations, train_losses, 'b-', linewidth=2, label='Total Loss')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # 2. Policy和Value Loss对比
    ax2 = plt.subplot(2, 2, 2)
    if policy_losses and value_losses:
        ax2.plot(iterations, policy_losses, 'r-', linewidth=2, label='Policy Loss')
        ax2.plot(iterations, value_losses, 'g-', linewidth=2, label='Value Loss')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Policy vs Value Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    # 3. 评估分数
    ax3 = plt.subplot(2, 2, 3)
    if eval_scores:
        ax3.plot(iterations, eval_scores, 'purple', linewidth=2, marker='o',
                markersize=6, label='Eval Score')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Evaluation Score Progress', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # 添加最高分标记
        max_idx = np.argmax(eval_scores)
        ax3.annotate(f'Max: {eval_scores[max_idx]:.1f}',
                    xy=(iterations[max_idx], eval_scores[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # 4. 分数变化率
    ax4 = plt.subplot(2, 2, 4)
    if eval_scores and len(eval_scores) > 1:
        score_diff = np.diff(eval_scores)
        ax4.bar(iterations[1:], score_diff, color=['green' if x > 0 else 'red' for x in score_diff],
               alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Score Change', fontsize=12)
        ax4.set_title('Score Improvement per Iteration', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ 图表已保存到: {save_path}")

    # 显示
    if show:
        plt.show()

    return fig


def print_training_summary(checkpoint_dir: str = "weights/alphazero"):
    """打印训练摘要"""
    history_path = os.path.join(checkpoint_dir, "history.json")

    if not os.path.exists(history_path):
        print(f"❌ 找不到历史文件: {history_path}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    iterations = history.get('iterations', [])
    eval_scores = history.get('eval_scores', [])

    if not iterations:
        print("❌ 无训练数据")
        return

    print("\n" + "="*60)
    print("  训练摘要")
    print("="*60)
    print(f"迭代次数: {len(iterations)}")

    if eval_scores:
        print(f"\n评估分数:")
        print(f"  初始: {eval_scores[0]:.1f}")
        print(f"  最终: {eval_scores[-1]:.1f}")
        print(f"  最高: {max(eval_scores):.1f}")
        print(f"  平均: {np.mean(eval_scores):.1f}")
        print(f"  提升: {eval_scores[-1] - eval_scores[0]:.1f} ({((eval_scores[-1]/eval_scores[0]-1)*100):.1f}%)")

    print("="*60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='可视化AlphaZero训练结果')
    parser.add_argument('--checkpoint-dir', type=str, default='weights/alphazero',
                       help='检查点目录')
    parser.add_argument('--save-path', type=str, default='training_visualization.png',
                       help='保存图片路径')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图片')
    parser.add_argument('--summary-only', action='store_true',
                       help='只打印摘要')

    args = parser.parse_args()

    if args.summary_only:
        print_training_summary(args.checkpoint_dir)
    else:
        visualize_training_history(
            checkpoint_dir=args.checkpoint_dir,
            save_path=args.save_path,
            show=not args.no_show
        )
        print_training_summary(args.checkpoint_dir)
