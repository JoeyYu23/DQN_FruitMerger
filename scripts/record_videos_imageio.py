"""
使用imageio录制高分局视频（更可靠）
"""
import numpy as np
import paddle
import cv2
from DQN import Agent, build_model
from GameInterface import GameInterface
from Game import visualize_feature
from render_utils import cover
import os
import imageio

def record_game_with_imageio(agent, env, seed, output_path, fps=12):
    """使用imageio录制游戏视频"""
    print(f"\n🎬 开始录制 Seed={seed} 的游戏...")

    env.reset(seed=seed)

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM

    frames = []
    step_count = 0
    reward_sum = 0

    # 第一步
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    while alive:
        step_count += 1

        # 获取Q值
        with paddle.no_grad():
            q_values = agent.policy_net(paddle.to_tensor(feature)).numpy().flatten()

        # 选择动作
        action = agent.predict(feature)
        if isinstance(action, np.ndarray):
            action = action.item()

        # 绘制游戏画面
        screen = env.game.draw()

        # Q值热力图
        unit_w = 1.0 * env.game.width / action_dim
        q_min, q_max = q_values.min(), q_values.max()
        if q_max > q_min:
            q_norm = (q_values - q_min) / (q_max - q_min)
        else:
            q_norm = np.zeros_like(q_values)

        for i in range(action_dim):
            rect = np.zeros_like(screen, dtype=np.uint8)
            if q_norm[i] < 0.5:
                b, g, r = int(255 * (1 - q_norm[i] * 2)), int(255 * q_norm[i] * 2), 0
            else:
                b, g, r = 0, int(255 * (1 - (q_norm[i] - 0.5) * 2)), int(255 * (q_norm[i] - 0.5) * 2)
            color = (b, g, r, 60)
            cv2.rectangle(rect, (int(i * unit_w), 0), (int((i + 1) * unit_w), env.game.height), color, -1)
            cover(screen, rect, 0.4)

        # 标记选择的动作
        best_rect = np.zeros_like(screen, dtype=np.uint8)
        cv2.rectangle(best_rect, (int(action * unit_w), 0), (int((action + 1) * unit_w), env.game.height),
                     (0, 0, 255, 150), 3)
        cover(screen, best_rect, 1)

        # 添加信息
        info = f"Step:{step_count:3d} Score:{env.game.score:3d} Reward:{int(reward_sum):4d}"
        cv2.putText(screen, info, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info2 = f"Seed:{seed} Q:{q_values[action]:.1f}"
        cv2.putText(screen, info2, (5, env.game.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 特征图
        reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
        feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(np.uint8)

        # 合并
        screen_rgb = cv2.cvtColor(screen, cv2.COLOR_BGRA2RGB)
        feature_rgb = cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB)
        combined = np.hstack([screen_rgb, feature_rgb])

        frames.append(combined)

        # 执行动作
        next_feature, reward, alive = env.next(action)
        reward_sum += np.sum(reward)
        feature = next_feature

        if step_count % 10 == 0:
            print(f"  步骤 {step_count}, 帧数 {len(frames)}", end='\r')

    # 最后一帧（游戏结束）
    final_screen = env.game.draw()
    cv2.putText(final_screen, "GAME OVER", (env.game.width // 2 - 60, env.game.height // 2 - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(final_screen, f"Score: {env.game.score}", (env.game.width // 2 - 50, env.game.height // 2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    final_rgb = cv2.cvtColor(final_screen, cv2.COLOR_BGRA2RGB)
    final_feature_rgb = cv2.cvtColor(feature_rgb, cv2.COLOR_BGR2RGB)
    final_combined = np.hstack([final_rgb, final_feature_rgb])

    # 结束画面保持3秒
    for _ in range(fps * 3):
        frames.append(final_combined)

    print(f"\n  正在保存视频到 {output_path}...")

    # 使用imageio保存
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"✅ 完成! 分数={env.game.score}, 步数={step_count}, "
          f"时长={len(frames)/fps:.1f}s, 大小={file_size_mb:.2f}MB")

    return {
        'seed': seed,
        'score': env.game.score,
        'reward': reward_sum,
        'steps': step_count,
        'frames': len(frames),
        'file': output_path,
        'size_mb': file_size_mb
    }

if __name__ == "__main__":
    print("=" * 70)
    print("🎥 录制Top 5高分局视频")
    print("=" * 70)

    # 初始化
    feature_dim = GameInterface.FEATURE_MAP_HEIGHT * GameInterface.FEATURE_MAP_WIDTH * 2
    action_dim = GameInterface.ACTION_NUM

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    agent.policy_net.set_state_dict(paddle.load("final.pdparams"))

    # 创建目录
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # Top 5 seeds（从之前的分析）
    top_games = [
        {'seed': 34, 'score': 368},
        {'seed': 14, 'score': 326},
        {'seed': 74, 'score': 319},
        {'seed': 33, 'score': 299},
        {'seed': 97, 'score': 279},
    ]

    print(f"\n将录制以下高分局:")
    for i, game in enumerate(top_games, 1):
        print(f"  #{i}: Seed={game['seed']}, 分数={game['score']}")

    videos = []
    for i, game in enumerate(top_games, 1):
        print(f"\n[{i}/{len(top_games)}] 录制第{i}名...")
        output_path = f"videos/top{i}_seed{game['seed']}_score{game['score']}.mp4"

        result = record_game_with_imageio(agent, env, game['seed'], output_path, fps=15)
        if result:
            videos.append(result)

    print(f"\n{'='*70}")
    print("✅ 所有视频录制完成!")
    print(f"{'='*70}")

    total_size = sum([v['size_mb'] for v in videos])
    print(f"\n📊 统计:")
    print(f"  视频数量: {len(videos)}")
    print(f"  总大小: {total_size:.2f}MB")
    print(f"  保存位置: {os.path.abspath('videos')}/\n")

    for v in videos:
        print(f"  {os.path.basename(v['file'])}: {v['size_mb']:.2f}MB")

    print(f"\n🎊 完成! 可以播放videos目录中的视频查看AI表现")
