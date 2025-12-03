#!/usr/bin/env python3
"""
测试并可视化训练好的AlphaZero模型
包含：录制视频、展示游戏过程、分析决策
"""

import numpy as np
import paddle
import cv2
import matplotlib.pyplot as plt
import os
from typing import List

from SuikaNet import SuikaNet
from AlphaZeroMCTS import AlphaZeroMCTS
from GameInterface import GameInterface
from StateConverter import StateConverter


def test_model_with_visualization(model_path: str,
                                  num_games: int = 5,
                                  simulations: int = 100,
                                  save_video: bool = True,
                                  show_frames: bool = True):
    """
    测试模型并可视化游戏过程

    Args:
        model_path: 模型权重路径
        num_games: 测试游戏局数
        simulations: MCTS模拟次数
        save_video: 是否保存视频
        show_frames: 是否显示关键帧
    """
    print("\n" + "="*70)
    print(f"  测试模型: {os.path.basename(model_path)}")
    print("="*70)

    # 加载模型
    network = SuikaNet(input_channels=13, num_actions=16, hidden_channels=64)
    network.set_state_dict(paddle.load(model_path))
    network.eval()

    # 创建MCTS和状态转换器
    mcts = AlphaZeroMCTS(
        network=network,
        num_simulations=simulations,
        temperature=0.0,
        add_dirichlet_noise=False
    )

    converter = StateConverter(
        grid_height=network.board_h,
        grid_width=network.board_w,
        feature_height=network.board_h,
        feature_width=network.board_w
    )

    # 测试游戏
    all_scores = []
    all_steps = []

    for game_idx in range(num_games):
        print(f"\n🎮 游戏 {game_idx+1}/{num_games}")
        print("-" * 70)

        game = GameInterface()
        game.reset(seed=3000 + game_idx)

        steps = 0
        frames = []
        actions = []
        scores_history = []

        while game.game.alive and steps < 100:
            # 获取状态并决策
            simplified_state = converter.game_to_simplified(game)
            action = mcts.get_action(simplified_state)

            # 记录决策
            actions.append(action)
            scores_history.append(game.game.score)

            # 执行动作
            _, _, alive = game.next(action)
            steps += 1

            # 记录帧
            game.game.draw()
            frame = game.game.screen.copy()
            frames.append(frame)

            # 打印进度
            if steps % 10 == 0:
                print(f"  步数: {steps}, 得分: {game.game.score}, 动作: {action}")

        final_score = game.game.score
        all_scores.append(final_score)
        all_steps.append(steps)

        print(f"\n✅ 游戏 {game_idx+1} 完成:")
        print(f"   最终得分: {final_score}")
        print(f"   总步数: {steps}")
        print(f"   平均每步得分: {final_score/steps:.2f}")

        # 第一局游戏：展示详细信息
        if game_idx == 0:
            # 保存视频
            if save_video and frames:
                video_path = f"game_test_{final_score}.mp4"
                save_game_video(frames, video_path, fps=5)
                print(f"   📹 视频已保存: {video_path}")

            # 显示关键帧
            if show_frames and frames:
                show_game_frames(frames, final_score, steps)

            # 分析动作分布
            analyze_actions(actions, scores_history)

    # 统计结果
    print("\n" + "="*70)
    print("  📊 测试结果汇总")
    print("="*70)
    print(f"平均得分: {np.mean(all_scores):.1f} ± {np.std(all_scores):.1f}")
    print(f"最高得分: {max(all_scores)}")
    print(f"最低得分: {min(all_scores)}")
    print(f"平均步数: {np.mean(all_steps):.1f}")
    print("="*70)

    return all_scores, all_steps


def save_game_video(frames: List, output_path: str, fps: int = 10):
    """保存游戏视频"""
    if not frames:
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # 转换BGRA到BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        out.write(frame_bgr)

    out.release()


def show_game_frames(frames: List, score: int, steps: int):
    """显示关键帧"""
    num_display = min(12, len(frames))
    indices = np.linspace(0, len(frames)-1, num_display, dtype=int)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        frame = frames[idx]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        axes[i].imshow(frame_rgb)
        axes[i].set_title(f'Step {idx}/{len(frames)}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'Game Replay (Score: {score}, Steps: {steps})',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('game_frames.png', dpi=150, bbox_inches='tight')
    print(f"   🖼️  关键帧已保存: game_frames.png")
    plt.close()


def analyze_actions(actions: List[int], scores: List[int]):
    """分析动作分布"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 动作分布
    action_counts = np.bincount(actions, minlength=16)
    ax1.bar(range(16), action_counts, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Action (Position)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Action Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(range(0, 16, 2))

    # 得分进度
    ax2.plot(scores, linewidth=2, color='green')
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Score Progress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('action_analysis.png', dpi=150, bbox_inches='tight')
    print(f"   📈 动作分析已保存: action_analysis.png")
    plt.close()


def compare_all_models(num_games: int = 10, simulations: int = 100):
    """比较所有训练的模型"""
    print("\n" + "="*70)
    print("  🔍 比较所有模型")
    print("="*70)

    model_dir = "weights/alphazero"
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pdparams')])

    if not model_files:
        print("❌ 未找到模型文件")
        return

    results = {}

    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace('.pdparams', '')

        print(f"\n测试模型: {model_name}")

        # 加载模型
        network = SuikaNet(input_channels=13, num_actions=16, hidden_channels=64)
        network.set_state_dict(paddle.load(model_path))
        network.eval()

        mcts = AlphaZeroMCTS(network=network, num_simulations=simulations, temperature=0.0)
        converter = StateConverter(20, 16, 20, 16)

        scores = []
        for i in range(num_games):
            game = GameInterface()
            game.reset(seed=5000 + i)

            while game.game.alive:
                state = converter.game_to_simplified(game)
                action = mcts.get_action(state)
                _, _, _ = game.next(action)

            scores.append(game.game.score)

        results[model_name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'max': max(scores),
            'min': min(scores)
        }

        print(f"  平均: {results[model_name]['mean']:.1f} ± {results[model_name]['std']:.1f}")
        print(f"  最高: {results[model_name]['max']}")

    # 绘制对比图
    plot_model_comparison(results)

    return results


def plot_model_comparison(results: dict):
    """绘制模型对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(results.keys())
    means = [results[m]['mean'] for m in models]
    stds = [results[m]['std'] for m in models]

    x = np.arange(len(models))
    ax.bar(x, means, yerr=stds, capsize=5, color='skyblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 模型对比图已保存: model_comparison.png")
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试并可视化AlphaZero模型')
    parser.add_argument('--model', type=str, default='weights/alphazero/iter_7.pdparams',
                       help='模型路径')
    parser.add_argument('--games', type=int, default=5,
                       help='测试游戏局数')
    parser.add_argument('--simulations', type=int, default=100,
                       help='MCTS模拟次数')
    parser.add_argument('--no-video', action='store_true',
                       help='不保存视频')
    parser.add_argument('--no-frames', action='store_true',
                       help='不显示关键帧')
    parser.add_argument('--compare-all', action='store_true',
                       help='比较所有模型')

    args = parser.parse_args()

    if args.compare_all:
        compare_all_models(num_games=10, simulations=args.simulations)
    else:
        test_model_with_visualization(
            model_path=args.model,
            num_games=args.games,
            simulations=args.simulations,
            save_video=not args.no_video,
            show_frames=not args.no_frames
        )
