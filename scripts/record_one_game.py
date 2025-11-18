#!/usr/bin/env python3
"""
快速录制一局MCTS游戏
"""

import numpy as np
import cv2
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent
from render_utils import cover
import os
import imageio
import time

print("="*70)
print("🎥 录制MCTS游戏视频")
print("="*70)

# 创建目录
if not os.path.exists('videos'):
    os.makedirs('videos')

# 创建智能体和环境
print("\n初始化MCTS智能体...")
agent = FastMCTSAgent(num_simulations=200)
env = GameInterface()

seed = 888
output_path = f"videos/mcts_game.mp4"
fps = 12

print(f"开始录制 (Seed={seed})...")
env.reset(seed=seed)

frames = []
step_count = 0
reward_sum = 0

# 第一步随机
action = np.random.randint(0, env.action_num)
feature, _, alive = env.next(action)

print("\n游戏进行中...")

while alive:
    step_count += 1

    # MCTS决策
    start_time = time.time()
    simple_state = agent._convert_state(env)
    grid_action = agent.mcts.search(simple_state, agent.num_simulations)
    think_time = time.time() - start_time

    # 绘制游戏画面
    screen = env.game.draw()

    # 标记选择的列
    unit_w = env.game.width / 10
    highlight = np.zeros_like(screen, dtype=np.uint8)
    cv2.rectangle(highlight,
                 (int(grid_action * unit_w), 0),
                 (int((grid_action + 1) * unit_w), env.game.height),
                 (0, 255, 0, 180), -1)
    cover(screen, highlight, 0.3)

    # 边框高亮
    cv2.rectangle(screen,
                 (int(grid_action * unit_w), 0),
                 (int((grid_action + 1) * unit_w), env.game.height),
                 (0, 255, 0, 255), 3)

    # 添加信息
    info = f"Step:{step_count:3d} Score:{env.game.score:4d} Col:{grid_action}"
    cv2.putText(screen, info, (5, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 2)

    # 显示MCTS统计
    root = agent.mcts.root
    if root and root.children:
        best_child = root.children.get(grid_action)
        if best_child:
            stats = f"Visits:{best_child.visit_count} Q:{best_child.get_value():.0f}"
            cv2.putText(screen, stats, (5, env.game.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200, 255), 1)

    # 转换为RGB
    screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
    frames.append(screen_rgb)

    # 执行动作
    game_action = int(grid_action * 16 / 10)
    game_action = min(15, max(0, game_action))

    next_feature, reward, alive = env.next(game_action)
    reward_sum += np.sum(reward)
    feature = next_feature

    if step_count % 5 == 0:
        print(f"  第{step_count}步, 得分{env.game.score}, 帧数{len(frames)}", end='\r')

print(f"\n\n游戏结束! 得分: {env.game.score}, 步数: {step_count}")

# 最后一帧
final_screen = env.game.draw()
cv2.putText(final_screen, "GAME OVER",
           (env.game.width // 2 - 70, env.game.height // 2 - 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 3)
cv2.putText(final_screen, f"Score: {env.game.score}",
           (env.game.width // 2 - 60, env.game.height // 2 + 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0, 255), 2)

final_rgb = cv2.cvtColor(final_screen, cv2.COLOR_BGRA2RGB)

# 结束画面保持3秒
for _ in range(fps * 3):
    frames.append(final_rgb)

print(f"\n正在保存视频...")

# 保存视频
imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

print(f"\n{'='*70}")
print(f"✅ 视频录制完成!")
print(f"{'='*70}")
print(f"\n📊 视频信息:")
print(f"  文件: {output_path}")
print(f"  得分: {env.game.score}")
print(f"  步数: {step_count}")
print(f"  时长: {len(frames)/fps:.1f}秒")
print(f"  帧数: {len(frames)}")
print(f"  大小: {file_size_mb:.2f}MB")
print(f"\n🎬 播放命令:")
print(f"  open {output_path}")
print(f"  或者")
print(f"  vlc {output_path}")
