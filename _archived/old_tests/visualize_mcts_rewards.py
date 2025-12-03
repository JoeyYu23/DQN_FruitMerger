#!/usr/bin/env python3
"""
可视化每一步MCTS所有位置的reward/Q值
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent, RealPhysicsNode
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def visualize_step_rewards(
    env,
    agent,
    step,
    output_dir
):
    """
    可视化当前步骤所有位置的reward

    Args:
        env: 游戏环境
        agent: MCTS智能体
        step: 当前步数
        output_dir: 输出目录
    """
    # 保存状态
    original_state = agent.mcts._save_state(env)

    # 直接计算每个action的两步前瞻reward
    action_rewards = {}
    action_segment_len = env.game.width / 16

    for action1 in range(16):
        # 恢复到初始状态
        agent.mcts._restore_state(env, original_state)

        # 记录初始状态
        score_before = env.game.score
        fruits_before = agent.mcts._get_fruits_info(env)

        # 执行第一步
        agent.mcts._apply_action(env, action1)

        # 计算第一步reward
        reward1 = agent.mcts._calculate_reward(env, score_before, fruits_before)

        # 记录第一步后的状态
        state_after_step1 = agent.mcts._save_state(env)
        score_after1 = env.game.score
        fruits_after1 = agent.mcts._get_fruits_info(env)

        # 计算所有第二步的reward，找最大值
        max_reward2 = float('-inf')

        for action2 in range(16):
            # 恢复到第一步后的状态
            agent.mcts._restore_state(env, state_after_step1)

            # 执行第二步
            agent.mcts._apply_action(env, action2)

            # 计算第二步reward
            reward2 = agent.mcts._calculate_reward(env, score_after1, fruits_after1)

            # 更新最大值
            if reward2 > max_reward2:
                max_reward2 = reward2

        # 总reward = 第一步reward + 第二步最大reward
        action_rewards[action1] = reward1 + max_reward2

    # 恢复状态
    agent.mcts._restore_state(env, original_state)

    # 收集每个动作的统计信息
    action_stats = []
    for action in range(16):
        q_value = action_rewards.get(action, 0.0)

        action_stats.append({
            'action': action,
            'q_value': q_value,
            'visits': 1,  # 每个都计算了1次
            'puct': q_value  # PUCT就用Q值
        })

    # 获取游戏画面（先渲染）
    env.game.draw()  # 渲染当前状态
    game_frame = env.game.screen.copy()

    # 创建可视化
    fig = plt.figure(figsize=(16, 10))

    # 1. 游戏画面 + 位置标记
    ax1 = plt.subplot(2, 3, (1, 4))
    game_rgb = cv2.cvtColor(game_frame, cv2.COLOR_RGBA2RGB)
    ax1.imshow(game_rgb)
    ax1.set_title(f'Step {step} - Game State', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 在游戏画面上标记每个位置的Q值
    game_width = env.game.width
    action_segment_len = game_width / 16

    # 找出Q值范围用于归一化颜色
    q_values = [s['q_value'] for s in action_stats]
    if max(q_values) > min(q_values):
        q_min, q_max = min(q_values), max(q_values)
    else:
        q_min, q_max = 0, 1

    # 在每个位置画圆圈和数值
    game_rgb_marked = game_rgb.copy()  # 创建一次副本

    for stat in action_stats:
        action = stat['action']
        q_value = stat['q_value']
        visits = stat['visits']

        x = int((action + 0.5) * action_segment_len)
        y = 30  # 顶部位置

        # 颜色映射：Q值越高越绿，越低越红
        if q_max > q_min:
            normalized_q = (q_value - q_min) / (q_max - q_min)
        else:
            normalized_q = 0.5

        color_r = int(255 * (1 - normalized_q))
        color_g = int(255 * normalized_q)

        # 画圆圈和访问次数
        cv2.circle(game_rgb_marked, (x, y), 12, (color_r, color_g, 0), -1)
        cv2.circle(game_rgb_marked, (x, y), 12, (255, 255, 255), 1)  # 白色边框

        # 画访问次数
        cv2.putText(game_rgb_marked, f'{visits}', (x-6, y+4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    ax1.imshow(game_rgb_marked)

    # 2. Q值柱状图
    ax2 = plt.subplot(2, 3, 2)
    actions = [s['action'] for s in action_stats]
    q_values = [s['q_value'] for s in action_stats]
    colors = plt.cm.RdYlGn([(q - q_min) / (q_max - q_min + 1e-6) for q in q_values])

    bars = ax2.bar(actions, q_values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Action (Position)', fontsize=11)
    ax2.set_ylabel('Q-Value', fontsize=11)
    ax2.set_title('Q-Value per Action', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 16, 2))

    # 标记最佳动作（选Q值最大的）
    best_action = max(action_stats, key=lambda x: x['q_value'])['action']
    ax2.axvline(best_action, color='red', linestyle='--', linewidth=2,
                label=f'Best: {best_action}')
    ax2.legend()

    # 3. 访问次数柱状图
    ax3 = plt.subplot(2, 3, 3)
    visits = [s['visits'] for s in action_stats]
    ax3.bar(actions, visits, color='skyblue', alpha=0.8, edgecolor='black')
    ax3.set_xlabel('Action (Position)', fontsize=11)
    ax3.set_ylabel('Visit Count', fontsize=11)
    ax3.set_title('Visit Count per Action', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(0, 16, 2))
    ax3.axvline(best_action, color='red', linestyle='--', linewidth=2)

    # 4. PUCT值柱状图
    ax4 = plt.subplot(2, 3, 5)
    puct_values = [s['puct'] for s in action_stats]
    ax4.bar(actions, puct_values, color='orange', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Action (Position)', fontsize=11)
    ax4.set_ylabel('PUCT Value', fontsize=11)
    ax4.set_title('PUCT Value per Action', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(range(0, 16, 2))
    ax4.axvline(best_action, color='red', linestyle='--', linewidth=2)

    # 5. 统计表格
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('off')

    # 选择top 5动作
    top_actions = sorted(action_stats, key=lambda x: x['visits'], reverse=True)[:5]

    table_data = []
    for stat in top_actions:
        table_data.append([
            f"{stat['action']}",
            f"{stat['q_value']:.2f}",
            f"{stat['visits']}",
            f"{stat['puct']:.2f}"
        ])

    table = ax5.table(
        cellText=table_data,
        colLabels=['Action', 'Q-Value', 'Visits', 'PUCT'],
        cellLoc='center',
        loc='center',
        bbox=[0, 0.3, 1, 0.6]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 标题
    ax5.text(0.5, 0.95, 'Top 5 Actions',
            ha='center', va='top', fontsize=12, fontweight='bold')

    # 游戏信息
    info_text = f"Score: {env.game.score}\n"
    info_text += f"Fruits: {len(env.game.fruits)}\n"
    info_text += f"Simulations: {agent.num_simulations}"
    ax5.text(0.5, 0.15, info_text,
            ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # 保存
    output_path = os.path.join(output_dir, f'step_{step:03d}_rewards.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 返回最佳动作
    return best_action


def run_game_with_visualization(
    seed=888,
    num_sims=50,
    max_steps=50,
    output_dir='mcts_rewards_visualization'
):
    """运行游戏并可视化每一步的reward"""

    print("="*70)
    print("🎨 MCTS Rewards 可视化")
    print("="*70)
    print(f"配置:")
    print(f"  Seed: {seed}")
    print(f"  Simulations: {num_sims}")
    print(f"  Max Steps: {max_steps}")
    print(f"  输出目录: {output_dir}")
    print("="*70)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✅ 创建目录: {output_dir}")

    # 创建环境和智能体
    env = GameInterface()
    agent = RealPhysicsMCTSAgent(num_simulations=num_sims)

    # 重置游戏
    env.reset(seed=seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0

    print(f"\n🎬 开始可视化...\n")

    while alive and step < max_steps:
        step += 1

        print(f"  Processing Step {step:3d}...", end='')

        # 可视化并获取最佳动作
        action = visualize_step_rewards(env, agent, step, output_dir)

        # 执行动作
        feature, reward, alive = env.next(action)

        print(f" Score={env.game.score:4d}, Fruits={len(env.game.fruits):2d}, Action={action:2d}")

    print(f"\n{'='*70}")
    print("✅ 可视化完成!")
    print(f"{'='*70}")
    print(f"📊 统计:")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  输出文件: {step} 张图片")
    print(f"\n📁 所有可视化已保存到: {output_dir}/")
    print("="*70)

    # 创建汇总视频
    create_summary_video(output_dir, step)

    return env.game.score, step


def create_summary_video(output_dir, total_steps):
    """将所有可视化图片合成视频"""

    print(f"\n🎥 正在创建汇总视频...")

    # 读取第一张图片获取尺寸
    first_img_path = os.path.join(output_dir, 'step_001_rewards.png')
    if not os.path.exists(first_img_path):
        print("⚠️  找不到图片文件")
        return

    first_img = cv2.imread(first_img_path)
    height, width = first_img.shape[:2]

    # 创建视频（使用更通用的avc1编码器）
    video_path = os.path.join(output_dir, 'rewards_summary.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))

    if not video_writer.isOpened():
        print("⚠️  无法创建视频，尝试使用mp4v编码器")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))

    for step in range(1, total_steps + 1):
        img_path = os.path.join(output_dir, f'step_{step:03d}_rewards.png')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            video_writer.write(img)

    video_writer.release()
    print(f"✅ 视频已保存: {video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='可视化MCTS每一步的reward')
    parser.add_argument('--seed', type=int, default=888, help='随机种子')
    parser.add_argument('--sims', type=int, default=50, help='MCTS模拟次数')
    parser.add_argument('--steps', type=int, default=30, help='最大步数')
    parser.add_argument('--output', type=str, default='mcts_rewards_viz',
                       help='输出目录')

    args = parser.parse_args()

    run_game_with_visualization(
        seed=args.seed,
        num_sims=args.sims,
        max_steps=args.steps,
        output_dir=args.output
    )

    print("\n✅ 完成!")
