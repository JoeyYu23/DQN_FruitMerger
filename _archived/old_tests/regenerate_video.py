#!/usr/bin/env python3
"""
重新生成汇总视频
"""

import cv2
import os

output_dir = 'mcts_rewards_viz'

print("="*70)
print("🎥 重新生成汇总视频")
print("="*70)

# 统计图片数量
img_files = sorted([f for f in os.listdir(output_dir) if f.endswith('_rewards.png')])
total_imgs = len(img_files)
print(f"\n找到 {total_imgs} 张图片")

if total_imgs == 0:
    print("⚠️  没有找到图片文件")
    exit(1)

# 读取第一张图片获取尺寸
first_img_path = os.path.join(output_dir, img_files[0])
first_img = cv2.imread(first_img_path)

if first_img is None:
    print(f"⚠️  无法读取图片: {first_img_path}")
    exit(1)

height, width = first_img.shape[:2]
print(f"图片尺寸: {width}x{height}")

# 尝试不同的编码器
encoders = [
    ('avc1', 'H.264 (推荐)'),
    ('mp4v', 'MPEG-4'),
    ('XVID', 'Xvid'),
]

video_path = os.path.join(output_dir, 'rewards_summary.mp4')

for fourcc_str, desc in encoders:
    print(f"\n尝试编码器: {fourcc_str} ({desc})")
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    video_writer = cv2.VideoWriter(video_path, fourcc, 2, (width, height))

    if not video_writer.isOpened():
        print(f"  ✗ 无法创建视频写入器")
        continue

    print(f"  ✓ 成功创建视频写入器")
    print(f"  正在写入 {total_imgs} 帧...")

    frames_written = 0
    for img_file in img_files:
        img_path = os.path.join(output_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ⚠️  无法读取: {img_file}")
            continue

        # 确保尺寸匹配
        if img.shape[:2] != (height, width):
            print(f"  ⚠️  尺寸不匹配: {img_file}")
            img = cv2.resize(img, (width, height))

        video_writer.write(img)
        frames_written += 1

        if frames_written % 10 == 0:
            print(f"    已写入 {frames_written}/{total_imgs} 帧")

    video_writer.release()

    # 检查视频文件
    if os.path.exists(video_path):
        file_size = os.path.getsize(video_path) / 1024 / 1024
        print(f"\n✅ 视频创建成功!")
        print(f"  路径: {video_path}")
        print(f"  大小: {file_size:.2f} MB")
        print(f"  帧数: {frames_written}")
        print(f"  帧率: 2 fps")
        print(f"  时长: {frames_written/2:.1f} 秒")
        break
    else:
        print(f"  ✗ 视频文件未创建")

print("\n" + "="*70)
