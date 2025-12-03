#!/usr/bin/env python3
"""
评估AlphaZero模型性能
可以可视化游戏过程
"""

import numpy as np
import paddle
import cv2
import matplotlib.pyplot as plt
from typing import Optional
import os

from SuikaNet import SuikaNet
from AlphaZeroMCTS import AlphaZeroMCTS
from GameInterface import GameInterface
from StateConverter import StateConverter


def evaluate_alphazero(model_path: str,
                       num_games: int = 20,
                       simulations: int = 200,
                       visualize: bool = False,
                       save_video: bool = False):
    """
    评估AlphaZero模型

    Args:
        model_path: 模型权重路径
        num_games: 评估游戏局数
        simulations: MCTS模拟次数
        visualize: 是否可视化游戏过程
        save_video: 是否保存视频
    """
    print("\n" + "="*70)
    print("  AlphaZero 模型评估")
    print("="*70)
    print(f"模型路径: {model_path}")
    print(f"游戏局数: {num_games}")
    print(f"MCTS模拟: {simulations}")
    print("="*70)

    # 加载模型
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return

    network = SuikaNet(input_channels=13, num_actions=16, hidden_channels=64)
    network.set_state_dict(paddle.load(model_path))
    network.eval()

    print("✅ 模型加载成功")

    # 创建MCTS
    mcts = AlphaZeroMCTS(
        network=network,
        num_simulations=simulations,
        temperature=0.0,  # 确定性选择
        add_dirichlet_noise=False
    )

    # 创建状态转换器
    converter = StateConverter(
        grid_height=network.board_h,
        grid_width=network.board_w,
        feature_height=network.board_h,
        feature_width=network.board_w
    )

    # 评估
    scores = []
    steps_list = []

    for game_idx in range(num_games):
        game = GameInterface()
        game.reset(seed=2000 + game_idx)

        steps = 0
        frames = [] if (visualize or save_video) and game_idx == 0 else None

        while game.game.alive:
            simplified_state = converter.game_to_simplified(game)
            action = mcts.get_action(simplified_state)
            _, _, alive = game.next(action)

            steps += 1

            # 记录帧
            if frames is not None:
                game.game.draw()
                frame = game.game.screen.copy()
                frames.append(frame)

        scores.append(game.game.score)
        steps_list.append(steps)

        print(f"  游戏 {game_idx+1}/{num_games}: 得分={game.game.score}, 步数={steps}")

        # 可视化第一局
        if frames and game_idx == 0:
            if visualize:
                visualize_game(frames, game.game.score, steps)
            if save_video:
                save_game_video(frames, f"game_eval_{game.game.score}.mp4")

    # 统计
    print("\n" + "="*70)
    print("  评估结果")
    print("="*70)
    print(f"平均得分: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"最高得分: {max(scores)}")
    print(f"最低得分: {min(scores)}")
    print(f"平均步数: {np.mean(steps_list):.1f}")
    print("="*70)

    # 绘制得分分布
    plot_score_distribution(scores)

    return {
        'scores': scores,
        'steps': steps_list,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores)
    }


def visualize_game(frames, score, steps):
    """可视化游戏过程"""
    print(f"\n🎮 显示游戏回放 (得分: {score}, 步数: {steps})")

    # 选择关键帧显示
    num_display = min(12, len(frames))
    indices = np.linspace(0, len(frames)-1, num_display, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        frame = frames[idx]
        # 转换颜色
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        axes[i].imshow(frame_rgb)
        axes[i].set_title(f'Step {idx}/{len(frames)}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Game Replay (Score: {score}, Steps: {steps})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_game_video(frames, output_path: str, fps: int = 10):
    """保存游戏视频"""
    if not frames:
        return

    print(f"\n💾 保存视频到: {output_path}")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # 转换BGRA到BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✅ 视频保存成功 (共{len(frames)}帧)")


def plot_score_distribution(scores):
    """绘制得分分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图
    ax1.bar(range(len(scores)), scores, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    ax1.set_xlabel('Game Index', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Score per Game', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 直方图
    ax2.hist(scores, bins=15, color='green', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(scores), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    ax2.set_xlabel('Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('score_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 得分分布图已保存: score_distribution.png")
    plt.show()


def compare_models(model_paths: list, num_games: int = 20, simulations: int = 200):
    """
    比较多个模型的性能

    Args:
        model_paths: 模型路径列表
        num_games: 每个模型评估的游戏局数
        simulations: MCTS模拟次数
    """
    print("\n" + "="*70)
    print("  模型对比评估")
    print("="*70)

    results = {}

    for model_path in model_paths:
        model_name = os.path.basename(model_path).replace('.pdparams', '')
        print(f"\n评估模型: {model_name}")

        result = evaluate_alphazero(
            model_path=model_path,
            num_games=num_games,
            simulations=simulations,
            visualize=False
        )

        results[model_name] = result

    # 对比图
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(results.keys())
    mean_scores = [results[name]['mean_score'] for name in model_names]
    std_scores = [results[name]['std_score'] for name in model_names]

    x = np.arange(len(model_names))
    ax.bar(x, mean_scores, yerr=std_scores, capsize=5,
          color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 模型对比图已保存: model_comparison.png")
    plt.show()

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='评估AlphaZero模型')
    parser.add_argument('--model-path', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--num-games', type=int, default=20,
                       help='评估游戏局数')
    parser.add_argument('--simulations', type=int, default=200,
                       help='MCTS模拟次数')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化游戏过程')
    parser.add_argument('--save-video', action='store_true',
                       help='保存视频')
    parser.add_argument('--compare', nargs='+',
                       help='比较多个模型 (提供多个模型路径)')

    args = parser.parse_args()

    if args.compare:
        compare_models(args.compare, args.num_games, args.simulations)
    else:
        evaluate_alphazero(
            model_path=args.model_path,
            num_games=args.num_games,
            simulations=args.simulations,
            visualize=args.visualize,
            save_video=args.save_video
        )
