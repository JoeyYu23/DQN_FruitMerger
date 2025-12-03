#!/usr/bin/env python3
"""
录制不同MERGE_BONUS的对比视频
"""

from GameInterface import GameInterface
from mcts.MCTS_tuned import TunedMCTSAgent, TunedConfig
import numpy as np
import cv2

def record_bonus_comparison(seed=1004):
    """录制MERGE_BONUS对比视频（1.0 vs 3.0）"""

    print("="*70)
    print("🎬 录制MERGE_BONUS对比视频")
    print("="*70)

    configs = [
        {'bonus': 1.0, 'label': 'BONUS=1.0 (Default)', 'color': (255, 200, 0)},
        {'bonus': 3.0, 'label': 'BONUS=3.0 (Best)', 'color': (0, 255, 0)}
    ]

    # 创建两个环境
    envs = []
    agents = []

    for config in configs:
        # 设置bonus
        TunedConfig.MERGE_BONUS = config['bonus']

        env = GameInterface()
        env.reset(seed=seed)
        agent = TunedMCTSAgent(num_simulations=100)

        # 第一步随机
        action = np.random.randint(0, env.action_num)
        env.next(action)

        envs.append(env)
        agents.append(agent)

    # 获取画面尺寸
    screen = envs[0].game.draw()
    height, width = screen.shape[:2]
    info_height = 80

    # 创建并排视频
    gap = 20
    video_width = width * 2 + gap
    video_height = height + info_height

    # 创建视频写入器
    output_file = 'mcts_bonus_comparison.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 5
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (video_width, video_height))

    print(f"\n🎥 开始录制...\n")

    step = 0
    max_steps = 200

    while step < max_steps:
        step += 1

        # 创建画布
        canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)

        # 为每个配置生成画面
        for i, (env, agent, config) in enumerate(zip(envs, agents, configs)):
            if not env.game.alive:
                screen = env.game.draw()
            else:
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
            cv2.putText(canvas, f"Score: {env.game.score}",
                       (x_offset + 10, label_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, f"Steps: {step if env.game.alive else 'END'}",
                       (x_offset + 10, label_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 添加总体信息
        cv2.putText(canvas, f'Step: {step}', (video_width//2 - 50, height + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 写入帧
        video_writer.write(canvas)

        # 检查是否两个游戏都结束
        if not envs[0].game.alive and not envs[1].game.alive:
            break

        if step % 10 == 0:
            scores = [env.game.score for env in envs]
            print(f"  Step {step:3d}: BONUS 1.0={scores[0]:3d}, "
                  f"BONUS 3.0={scores[1]:3d}, "
                  f"差距={scores[1]-scores[0]:+3d}")

    # 添加结束画面
    for _ in range(fps * 3):
        video_writer.write(canvas)

    video_writer.release()

    print(f"\n{'='*70}")
    print("✅ 对比视频录制完成!")
    print(f"{'='*70}")
    print(f"  视频已保存: {output_file}")
    print(f"  BONUS 1.0 得分: {envs[0].game.score}")
    print(f"  BONUS 3.0 得分: {envs[1].game.score}")
    print(f"  差距: {envs[1].game.score - envs[0].game.score:+d} "
          f"({(envs[1].game.score/envs[0].game.score-1)*100:+.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1004)
    args = parser.parse_args()

    record_bonus_comparison(seed=args.seed)

    print(f"\n🎉 完成!")
    print(f"   播放: open mcts_bonus_comparison.mp4")
