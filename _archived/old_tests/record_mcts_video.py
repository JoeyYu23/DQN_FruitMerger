#!/usr/bin/env python3
"""
录制MCTS玩游戏的视频
"""

from GameInterface import GameInterface
from mcts.MCTS_tuned import TunedMCTSAgent
import numpy as np
import cv2
import time

def record_mcts_game_video(seed=888, num_simulations=50, output_file='mcts_gameplay.mp4'):
    """
    录制MCTS玩游戏的视频

    Args:
        seed: 随机种子
        num_simulations: MCTS模拟次数
        output_file: 输出视频文件名
    """
    print("="*70)
    print("🎬 录制MCTS游戏视频")
    print("="*70)
    print(f"配置:")
    print(f"  Seed: {seed}")
    print(f"  MCTS Simulations: {num_simulations}")
    print(f"  输出文件: {output_file}")
    print("="*70)

    # 创建游戏环境
    env = GameInterface()
    agent = TunedMCTSAgent(num_simulations=num_simulations)

    # 重置游戏
    env.reset(seed=seed)

    # 获取初始画面来确定视频尺寸
    initial_screen = env.game.draw()
    height, width = initial_screen.shape[:2]

    # 增加信息显示区域的高度
    info_height = 80
    video_height = height + info_height
    video_width = width

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5  # 每秒5帧（慢速播放，便于观察）
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

    if not video_writer.isOpened():
        print("❌ 无法创建视频文件！")
        return

    print(f"\n🎥 开始录制...\n")

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    start_time = time.time()

    while alive and step < 200:
        step += 1

        # MCTS决策
        step_start = time.time()
        action = agent.predict(env)[0]
        decision_time = time.time() - step_start

        # 获取游戏画面
        screen = env.game.draw()
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

        # 创建带信息栏的画布
        canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        canvas[:height, :] = screen_rgb

        # 在底部添加信息栏（黑色背景）
        info_bg = canvas[height:, :]
        info_bg[:] = (30, 30, 30)  # 深灰色背景

        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # 白色

        # 第一行信息
        y1 = height + 25
        cv2.putText(canvas, f'Step: {step}', (10, y1), font, font_scale, color, thickness)
        cv2.putText(canvas, f'Score: {env.game.score}', (150, y1), font, font_scale, (0, 255, 0), thickness)
        cv2.putText(canvas, f'Action: Col {action}', (300, y1), font, font_scale, (255, 200, 0), thickness)

        # 第二行信息
        y2 = height + 55
        cv2.putText(canvas, f'MCTS: {num_simulations} sims', (10, y2), font, font_scale, color, thickness)
        cv2.putText(canvas, f'Time: {decision_time:.2f}s', (220, y2), font, font_scale, (100, 200, 255), thickness)

        # 写入视频帧
        video_writer.write(canvas)

        # 执行动作
        feature, reward, alive = env.next(action)

        # 打印进度
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  录制第 {step:3d} 步: Score={env.game.score:3d}, "
                  f"用时 {elapsed:.1f}s")

    # 添加结束画面（停留2秒）
    final_screen = env.game.draw()
    final_screen_rgb = cv2.cvtColor(final_screen, cv2.COLOR_BGRA2RGB)
    final_canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
    final_canvas[:height, :] = final_screen_rgb
    final_canvas[height:, :] = (30, 30, 30)

    # 添加最终信息
    cv2.putText(final_canvas, 'GAME OVER', (width//2 - 80, y1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.putText(final_canvas, f'Final Score: {env.game.score}', (10, y2),
                font, font_scale, (0, 255, 0), thickness)
    cv2.putText(final_canvas, f'Total Steps: {step}', (220, y2),
                font, font_scale, (255, 255, 255), thickness)

    # 写入结束画面（2秒 = fps * 2 帧）
    for _ in range(fps * 2):
        video_writer.write(final_canvas)

    # 释放资源
    video_writer.release()

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("🏁 录制完成!")
    print(f"{'='*70}")
    print(f"📊 统计:")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  平均每步得分: {env.game.score/step:.2f}")
    print(f"  录制耗时: {total_time:.1f}秒")
    print(f"  视频时长: {(step + fps*2) / fps:.1f}秒")
    print(f"  视频分辨率: {video_width}x{video_height}")
    print(f"  帧率: {fps} FPS")
    print(f"\n🎬 视频已保存: {output_file}")
    print(f"{'='*70}")

    return env.game.score, step


def record_comparison_video(seed=888):
    """
    录制不同simulation数量的对比视频（并排显示）
    """
    print("\n" + "="*70)
    print("🎬 录制对比视频（20 vs 100 simulations）")
    print("="*70)

    sim_configs = [
        {'sims': 20, 'label': '20 Sims (Fast)', 'color': (0, 255, 255)},
        {'sims': 100, 'label': '100 Sims (Best)', 'color': (0, 255, 0)}
    ]

    # 创建两个环境
    envs = []
    agents = []

    for config in sim_configs:
        env = GameInterface()
        env.reset(seed=seed)
        agent = TunedMCTSAgent(num_simulations=config['sims'])

        # 第一步随机
        action = np.random.randint(0, env.action_num)
        env.next(action)

        envs.append(env)
        agents.append(agent)

    # 获取画面尺寸
    screen = envs[0].game.draw()
    height, width = screen.shape[:2]
    info_height = 60

    # 创建并排视频（两个画面+中间间隔）
    gap = 20
    video_width = width * 2 + gap
    video_height = height + info_height

    # 创建视频写入器
    output_file = 'mcts_comparison_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

    print(f"\n🎥 开始录制对比视频...\n")

    step = 0
    max_steps = 200

    while step < max_steps:
        step += 1

        # 创建画布
        canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        # 为每个配置生成画面
        for i, (env, agent, config) in enumerate(zip(envs, agents, sim_configs)):
            if not env.game.alive:
                # 游戏结束，显示最后画面
                screen = env.game.draw()
            else:
                # 执行MCTS决策
                action = agent.predict(env)[0]
                screen = env.game.draw()
                env.next(action)

            screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)

            # 放置画面
            x_offset = i * (width + gap)
            canvas[:height, x_offset:x_offset+width] = screen_rgb

            # 添加标签
            label_y = height + 25
            cv2.putText(canvas, config['label'], (x_offset + 10, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, config['color'], 2)
            cv2.putText(canvas, f"Score: {env.game.score}", (x_offset + 10, label_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 添加总体信息
        cv2.putText(canvas, f'Step: {step}', (video_width//2 - 40, height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 写入帧
        video_writer.write(canvas)

        # 检查是否两个游戏都结束
        if not envs[0].game.alive and not envs[1].game.alive:
            break

        if step % 10 == 0:
            scores = [env.game.score for env in envs]
            print(f"  Step {step:3d}: Scores = {scores}")

    # 添加结束画面
    for _ in range(fps * 2):
        video_writer.write(canvas)

    video_writer.release()

    print(f"\n{'='*70}")
    print("✅ 对比视频录制完成!")
    print(f"  视频已保存: {output_file}")
    print(f"  20 Sims 得分: {envs[0].game.score}")
    print(f"  100 Sims 得分: {envs[1].game.score}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='录制MCTS游戏视频')
    parser.add_argument('--seed', type=int, default=888, help='随机种子')
    parser.add_argument('--sims', type=int, default=50, help='MCTS模拟次数')
    parser.add_argument('--output', type=str, default='mcts_gameplay.mp4',
                       help='输出视频文件名')
    parser.add_argument('--compare', action='store_true',
                       help='录制对比视频（20 vs 100 sims）')

    args = parser.parse_args()

    if args.compare:
        record_comparison_video(seed=args.seed)
    else:
        record_mcts_game_video(
            seed=args.seed,
            num_simulations=args.sims,
            output_file=args.output
        )

    print("\n🎉 完成! 可以使用视频播放器打开观看")
    print(f"   macOS: open {args.output if not args.compare else 'mcts_comparison_video.mp4'}")
