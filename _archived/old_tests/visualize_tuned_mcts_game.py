#!/usr/bin/env python3
"""
可视化MCTS的单局游戏，展示详细的搜索树信息
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from GameInterface import GameInterface
from mcts.MCTS_optimized import FastMCTSAgent
from render_utils import putText2


def visualize_mcts_step(env, agent, step_num, save_dir='mcts_viz'):
    """
    可视化MCTS单步的决策过程

    Args:
        env: 游戏环境
        agent: MCTS智能体
        step_num: 当前步数
        save_dir: 保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 执行MCTS搜索
    grid_state = agent._convert_state(env)
    action = agent.mcts.search(grid_state, agent.num_simulations)

    # 获取搜索树信息
    root = agent.mcts.root

    # 创建可视化画布
    fig = plt.figure(figsize=(18, 10))

    # 1. 游戏画面 (左上)
    ax1 = plt.subplot(2, 3, 1)
    screen = env.game.draw()
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
    ax1.imshow(screen_rgb)
    ax1.set_title(f'Step {step_num} - Game State\nScore: {env.game.score}',
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 标记选择的列
    unit_w = env.game.width / 10
    rect = Rectangle((action * unit_w, 0), unit_w, env.game.height,
                     linewidth=3, edgecolor='lime', facecolor='none')
    ax1.add_patch(rect)

    # 2. 访问次数分布 (右上)
    ax2 = plt.subplot(2, 3, 2)
    if root and root.children:
        visits = [root.children.get(a, type('', (), {'visit_count': 0})).visit_count
                  for a in range(10)]
        colors = ['lime' if a == action else 'steelblue' for a in range(10)]
        bars = ax2.bar(range(10), visits, color=colors, alpha=0.7, edgecolor='black')

        # 标注百分比
        total_visits = sum(visits)
        for i, (bar, v) in enumerate(zip(bars, visits)):
            if v > 0:
                pct = v / total_visits * 100
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(visits)*0.02,
                        f'{pct:.0f}%', ha='center', va='bottom', fontsize=9)

        ax2.set_xlabel('Action (Column)', fontsize=12)
        ax2.set_ylabel('Visit Count', fontsize=12)
        ax2.set_title(f'MCTS Visit Distribution\nTotal Simulations: {root.visit_count}',
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Q值分布 (右中)
    ax3 = plt.subplot(2, 3, 3)
    if root and root.children:
        q_values = []
        actions_list = []
        for a in range(10):
            child = root.children.get(a)
            if child and child.visit_count > 0:
                q_values.append(child.get_value())
                actions_list.append(a)

        if q_values:
            colors = ['lime' if a == action else 'orange' for a in actions_list]
            bars = ax3.bar(actions_list, q_values, color=colors, alpha=0.7, edgecolor='black')

            # 标注数值
            for bar, q in zip(bars, q_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(q_values) - min(q_values))*0.05,
                        f'{q:.0f}', ha='center', va='bottom', fontsize=9)

            ax3.set_xlabel('Action (Column)', fontsize=12)
            ax3.set_ylabel('Q Value', fontsize=12)
            ax3.set_title('Action Q-Values', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')

    # 4. Top 5候选动作详情 (左下)
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')

    if root and root.children:
        sorted_children = sorted(root.children.items(),
                                key=lambda x: x[1].visit_count,
                                reverse=True)

        text_lines = ['🎯 Top 5 Actions:\n']
        text_lines.append('-' * 40)

        total_visits = sum(c.visit_count for _, c in root.children.items())

        for idx, (act, child) in enumerate(sorted_children[:5], 1):
            visit_pct = child.visit_count / total_visits * 100 if total_visits > 0 else 0
            q_val = child.get_value()

            marker = '👉' if act == action else '  '
            text_lines.append(f'\n{marker} #{idx}  Column {act}:')
            text_lines.append(f'    Visits: {child.visit_count:4d} ({visit_pct:5.1f}%)')
            text_lines.append(f'    Q-Value: {q_val:6.1f}')

        ax4.text(0.05, 0.95, '\n'.join(text_lines),
                transform=ax4.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # 5. 搜索统计信息 (中下)
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    if root:
        stats_lines = ['📊 Search Statistics:\n']
        stats_lines.append('-' * 40)
        stats_lines.append(f'\n🔢 Total Simulations: {root.visit_count}')
        stats_lines.append(f'🌲 Expanded Actions: {len(root.children)}')

        if root.children:
            avg_visits = np.mean([c.visit_count for c in root.children.values()])
            stats_lines.append(f'📈 Avg Visits/Action: {avg_visits:.1f}')

            max_q = max(c.get_value() for c in root.children.values())
            min_q = min(c.get_value() for c in root.children.values())
            stats_lines.append(f'🎚️  Q Range: [{min_q:.0f}, {max_q:.0f}]')

        stats_lines.append(f'\n✅ Selected Action: Column {action}')

        ax5.text(0.05, 0.95, '\n'.join(stats_lines),
                transform=ax5.transAxes,
                fontsize=11,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 6. 游戏信息 (右下)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    game_info = ['🎮 Game Information:\n']
    game_info.append('-' * 40)
    game_info.append(f'\n🏆 Current Score: {env.game.score}')
    game_info.append(f'🎯 Current Fruit: Type {env.game.current_fruit_type}')
    game_info.append(f'🍇 Max Fruit: Type {env.game.largest_fruit_type}')
    game_info.append(f'🎲 Fruits on Board: {len(env.game.fruits)}')
    game_info.append(f'⏱️  Step Number: {step_num}')

    ax6.text(0.05, 0.95, '\n'.join(game_info),
            transform=ax6.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()

    # 保存图片
    output_path = os.path.join(save_dir, f'mcts_step_{step_num:03d}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  💾 Step {step_num} 可视化已保存: {output_path}")

    return action


def play_mcts_game_with_viz(seed=1234, max_steps=30, num_simulations=200):
    """
    运行一局MCTS游戏并可视化每一步

    Args:
        seed: 随机种子
        max_steps: 最大步数
        num_simulations: 每步的MCTS模拟次数
    """
    print("="*70)
    print(f"🎮 MCTS游戏可视化")
    print("="*70)
    print(f"⚙️  配置:")
    print(f"   Seed: {seed}")
    print(f"   Max Steps: {max_steps}")
    print(f"   MCTS Simulations: {num_simulations}")
    print("="*70)

    # 创建环境和智能体
    env = GameInterface()
    agent = FastMCTSAgent(num_simulations=num_simulations)

    # 重置游戏
    env.reset(seed=seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    game_action = int(action * 10 / 16)  # 转换到10列
    feature, _, alive = env.next(action)

    step = 0
    scores = [0]

    print(f"\n🚀 开始游戏 (Seed={seed})...\n")

    # 游戏循环
    while alive and step < max_steps:
        step += 1

        print(f"📍 Step {step}/{max_steps}, Score: {env.game.score}", end=' ')

        # 可视化当前步
        grid_action = visualize_mcts_step(env, agent, step)

        # 转换动作并执行
        game_action = int(grid_action * 16 / 10)
        game_action = min(15, max(0, game_action))

        feature, reward, alive = env.next(game_action)
        scores.append(env.game.score)

        print(f"  ✅ Action: Col {grid_action} → Score: {env.game.score}")

    print(f"\n{'='*70}")
    print(f"🏁 游戏结束!")
    print(f"{'='*70}")
    print(f"📊 最终统计:")
    print(f"   最终得分: {env.game.score}")
    print(f"   总步数: {step}")
    print(f"   平均每步得分: {env.game.score/step:.2f}")
    print(f"   可视化文件数: {step}")
    print(f"\n💾 所有可视化已保存到: mcts_viz/")
    print(f"{'='*70}")

    # 创建得分曲线图
    create_score_plot(scores, seed)

    return env.game.score, step


def create_score_plot(scores, seed):
    """创建得分进度图"""
    plt.figure(figsize=(10, 6))
    plt.plot(scores, linewidth=2, marker='o', markersize=4, color='green')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'MCTS Score Progress (Seed={seed})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 标注最终得分
    plt.text(len(scores)-1, scores[-1], f' Final: {scores[-1]}',
            fontsize=11, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig('mcts_viz/score_progress.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  📈 得分曲线已保存: mcts_viz/score_progress.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MCTS游戏可视化')
    parser.add_argument('--seed', type=int, default=1234, help='随机种子')
    parser.add_argument('--steps', type=int, default=30, help='最大步数')
    parser.add_argument('--simulations', type=int, default=200, help='MCTS模拟次数')

    args = parser.parse_args()

    play_mcts_game_with_viz(
        seed=args.seed,
        max_steps=args.steps,
        num_simulations=args.simulations
    )

    print(f"\n✨ 完成! 可以查看 mcts_viz/ 目录中的可视化文件")
    print(f"   例如: open mcts_viz/mcts_step_001.png")
