#!/usr/bin/env python3
"""
录制 Real Physics MCTS 游戏视频
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent
import numpy as np
import cv2
import time

def record_real_physics_mcts(
    seed=888,
    num_sims=50,
    max_steps=100,
    output_path='real_physics_mcts_video.mp4'
):
    """录制Real Physics MCTS游戏视频"""

    print("="*70)
    print("🎥 录制 Real Physics MCTS 游戏视频")
    print("="*70)
    print(f"配置:")
    print(f"  Seed: {seed}")
    print(f"  Simulations: {num_sims}")
    print(f"  Max Steps: {max_steps}")
    print(f"  输出路径: {output_path}")
    print("="*70)

    # 创建环境和智能体
    env = GameInterface()
    agent = RealPhysicsMCTSAgent(num_simulations=num_sims)

    # 视频设置
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (300, 400))

    # 重置游戏
    env.reset(seed=seed)

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0
    start_time = time.time()

    print(f"\n🎬 开始录制...\n")

    while alive and step < max_steps:
        step += 1

        # 打印进度
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:3d}: Score={env.game.score:4d}, "
                  f"Fruits={len(env.game.fruits):2d}, "
                  f"Time={elapsed:.1f}s")

        # MCTS决策
        step_start = time.time()
        action = agent.predict(env)[0]
        decision_time = time.time() - step_start

        # 执行动作
        feature, reward, alive = env.next(action)

        # 录制帧
        frame = env.game.screen

        # 添加信息叠加
        frame_with_info = frame.copy()

        # 添加半透明黑色背景框
        overlay = frame_with_info.copy()
        cv2.rectangle(overlay, (5, 5), (295, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame_with_info, 0.7, 0, frame_with_info)

        # 显示信息
        info_texts = [
            f"Real Physics MCTS",
            f"Sims: {num_sims}",
            f"Step: {step}",
            f"Score: {env.game.score}",
            f"Decision: {decision_time:.2f}s"
        ]

        y_offset = 20
        for text in info_texts:
            cv2.putText(frame_with_info, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 12

        # 写入视频
        video_writer.write(cv2.cvtColor(frame_with_info, cv2.COLOR_RGBA2BGR))

    # 游戏结束，添加结束画面
    for _ in range(fps * 2):  # 2秒结束画面
        frame = env.game.screen
        frame_with_info = frame.copy()

        # 添加半透明黑色背景
        overlay = frame_with_info.copy()
        cv2.rectangle(overlay, (50, 150), (250, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame_with_info, 0.5, 0, frame_with_info)

        # 显示最终得分
        cv2.putText(frame_with_info, "GAME OVER", (70, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_with_info, f"Final Score: {env.game.score}", (70, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame_with_info, f"Steps: {step}", (70, 235),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        video_writer.write(cv2.cvtColor(frame_with_info, cv2.COLOR_RGBA2BGR))

    video_writer.release()

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print("✅ 录制完成!")
    print(f"{'='*70}")
    print(f"📊 统计:")
    print(f"  最终得分: {env.game.score}")
    print(f"  总步数: {step}")
    print(f"  总耗时: {total_time:.1f}秒")
    print(f"  平均每步: {total_time/step:.2f}秒")
    print(f"\n📹 视频已保存: {output_path}")
    print("="*70)

    return env.game.score, step


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='录制Real Physics MCTS游戏视频')
    parser.add_argument('--seed', type=int, default=888, help='随机种子')
    parser.add_argument('--sims', type=int, default=50, help='MCTS模拟次数')
    parser.add_argument('--steps', type=int, default=100, help='最大步数')
    parser.add_argument('--output', type=str, default='real_physics_mcts_video.mp4',
                       help='输出视频路径')

    args = parser.parse_args()

    record_real_physics_mcts(
        seed=args.seed,
        num_sims=args.sims,
        max_steps=args.steps,
        output_path=args.output
    )

    print("\n✅ 完成!")
