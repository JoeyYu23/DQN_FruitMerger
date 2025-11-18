"""
录制高分局视频 - 保存Top游戏的完整视频
"""
import numpy as np
import paddle
import cv2
from DQN import Agent, build_model
from GameInterface import GameInterface
from Game import visualize_feature
from render_utils import cover
import os

def record_game_video(agent, env, seed, output_path, fps=10, show_q_values=True):
    """
    录制一局游戏并保存为视频

    参数:
        agent: DQN智能体
        env: 游戏环境
        seed: 随机种子
        output_path: 输出视频路径
        fps: 视频帧率
        show_q_values: 是否显示Q值热力图
    """
    print(f"\n🎬 开始录制 Seed={seed} 的游戏视频...")

    env.reset(seed=seed)

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM

    # 初始化视频写入器
    # 合并游戏画面和特征图
    frame_width = env.game.width * 2  # 游戏画面 + 特征图
    frame_height = env.game.height

    # 尝试不同的编码器
    # macOS上使用avc1 (H.264)更可靠
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 如果avc1失败，尝试mp4v
    if not video_writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        print(f"❌ 无法创建视频文件: {output_path}")
        return None

    # 游戏数据
    step_count = 0
    reward_sum = 0
    frames_recorded = 0

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

        if show_q_values:
            # 绘制Q值热力图
            unit_w = 1.0 * env.game.width / action_dim

            # 归一化Q值
            q_min, q_max = q_values.min(), q_values.max()
            if q_max > q_min:
                q_norm = (q_values - q_min) / (q_max - q_min)
            else:
                q_norm = np.zeros_like(q_values)

            # 为每个位置绘制颜色条
            for i in range(action_dim):
                rect = np.zeros_like(screen, dtype=np.uint8)

                # 颜色映射：蓝色(低) -> 绿色(中) -> 红色(高)
                if q_norm[i] < 0.5:
                    # 蓝色到绿色
                    b = int(255 * (1 - q_norm[i] * 2))
                    g = int(255 * q_norm[i] * 2)
                    r = 0
                else:
                    # 绿色到红色
                    b = 0
                    g = int(255 * (1 - (q_norm[i] - 0.5) * 2))
                    r = int(255 * (q_norm[i] - 0.5) * 2)

                color = (b, g, r, 60)

                cv2.rectangle(rect,
                             (int(i * unit_w), 0),
                             (int((i + 1) * unit_w), env.game.height),
                             color, -1)
                cover(screen, rect, 0.4)

            # 标记选择的动作
            best_rect = np.zeros_like(screen, dtype=np.uint8)
            cv2.rectangle(best_rect,
                         (int(action * unit_w), 0),
                         (int((action + 1) * unit_w), env.game.height),
                         (0, 0, 255, 150), 3)
            cover(screen, best_rect, 1)

        # 添加信息文本
        info = f"Step:{step_count:3d} Score:{env.game.score:3d} Reward:{int(reward_sum):4d}"
        cv2.putText(screen, info, (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info2 = f"Seed:{seed} Q:{q_values[action]:.1f}"
        cv2.putText(screen, info2, (5, env.game.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 绘制特征图
        reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
        feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(np.uint8)

        # 转换为BGR格式
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        # 合并画面
        combined_frame = np.hstack([screen_bgr, feature_img])

        # 写入视频
        video_writer.write(combined_frame)
        frames_recorded += 1

        # 执行动作
        next_feature, reward, alive = env.next(action)
        reward_sum += np.sum(reward)
        feature = next_feature

        # 进度显示
        if step_count % 10 == 0:
            print(f"  录制进度: 步骤 {step_count}, 帧数 {frames_recorded}", end='\r')

    # 游戏结束画面（保持3秒）
    final_screen = env.game.draw()
    cv2.putText(final_screen, f"GAME OVER",
               (env.game.width // 2 - 60, env.game.height // 2 - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(final_screen, f"Final Score: {env.game.score}",
               (env.game.width // 2 - 80, env.game.height // 2 + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    final_screen_bgr = cv2.cvtColor(final_screen, cv2.COLOR_BGRA2BGR)
    final_feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(np.uint8)
    final_combined = np.hstack([final_screen_bgr, final_feature_img])

    for _ in range(fps * 3):  # 3秒
        video_writer.write(final_combined)
        frames_recorded += 1

    video_writer.release()

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"\n✅ 视频录制完成!")
    print(f"  文件: {output_path}")
    print(f"  分数: {env.game.score}")
    print(f"  步数: {step_count}")
    print(f"  帧数: {frames_recorded}")
    print(f"  时长: {frames_recorded/fps:.1f}秒")
    print(f"  大小: {file_size_mb:.2f}MB")

    return {
        'seed': seed,
        'score': env.game.score,
        'reward': reward_sum,
        'steps': step_count,
        'frames': frames_recorded,
        'file': output_path,
        'size_mb': file_size_mb
    }

def find_and_record_top_games(num_scan=100, top_k=3, fps=10, output_dir='videos'):
    """
    找出高分局并录制视频

    参数:
        num_scan: 扫描的游戏局数
        top_k: 录制前k名
        fps: 视频帧率
        output_dir: 输出目录
    """
    print("=" * 70)
    print("🎥 高分局视频录制")
    print("=" * 70)

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 创建目录: {output_dir}")

    # 初始化
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    agent.policy_net.set_state_dict(paddle.load("final.pdparams"))

    print(f"\n🔍 扫描 {num_scan} 局游戏，寻找Top {top_k}...")

    # 扫描游戏
    game_results = []
    for seed in range(num_scan):
        env.reset(seed=seed)

        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        step_count = 0
        reward_sum = 0

        while alive:
            step_count += 1
            action = agent.predict(feature)
            if isinstance(action, np.ndarray):
                action = action.item()
            feature, reward, alive = env.next(action)
            reward_sum += np.sum(reward)

        game_results.append({
            'seed': seed,
            'score': env.game.score,
            'reward': reward_sum,
            'steps': step_count
        })

        if (seed + 1) % 20 == 0:
            print(f"  进度: {seed + 1}/{num_scan}", end='\r')

    # 排序
    game_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n\n🏆 Top {top_k} 高分局:")
    for i, game in enumerate(game_results[:top_k]):
        print(f"  #{i+1}: Seed={game['seed']:3d}, "
              f"分数={game['score']:3d}, "
              f"奖励={game['reward']:6.1f}, "
              f"步数={game['steps']:3d}")

    # 录制视频
    print(f"\n{'='*70}")
    print("开始录制视频...")
    print(f"{'='*70}")

    recorded_videos = []
    for i, game in enumerate(game_results[:top_k]):
        rank = i + 1
        output_path = os.path.join(
            output_dir,
            f"top{rank}_seed{game['seed']}_score{game['score']}.mp4"
        )

        print(f"\n[{rank}/{top_k}] 录制 Seed={game['seed']}, 分数={game['score']}")

        result = record_game_video(agent, env, game['seed'], output_path, fps=fps)
        if result:
            recorded_videos.append(result)

    # 总结
    print(f"\n{'='*70}")
    print("✅ 所有视频录制完成!")
    print(f"{'='*70}")

    total_size = sum([v['size_mb'] for v in recorded_videos])
    total_duration = sum([v['frames']/fps for v in recorded_videos])

    print(f"\n📊 统计:")
    print(f"  录制视频数: {len(recorded_videos)}")
    print(f"  总时长: {total_duration:.1f}秒")
    print(f"  总大小: {total_size:.2f}MB")
    print(f"  保存位置: {os.path.abspath(output_dir)}/")

    print(f"\n📹 视频列表:")
    for i, v in enumerate(recorded_videos, 1):
        print(f"  {i}. {os.path.basename(v['file'])}")
        print(f"     分数={v['score']}, 步数={v['steps']}, "
              f"时长={v['frames']/fps:.1f}s, 大小={v['size_mb']:.2f}MB")

    return recorded_videos

if __name__ == "__main__":
    import sys

    # 参数设置
    NUM_SCAN = 100   # 扫描局数
    TOP_K = 5        # 录制前5名
    FPS = 12         # 视频帧率（可调节，越高越流畅但文件越大）
    OUTPUT_DIR = 'videos'

    print("\n设置:")
    print(f"  扫描局数: {NUM_SCAN}")
    print(f"  录制数量: Top {TOP_K}")
    print(f"  视频帧率: {FPS} FPS")
    print(f"  输出目录: {OUTPUT_DIR}/")

    try:
        videos = find_and_record_top_games(
            num_scan=NUM_SCAN,
            top_k=TOP_K,
            fps=FPS,
            output_dir=OUTPUT_DIR
        )

        print(f"\n🎊 完成! 可以在 {OUTPUT_DIR}/ 目录查看视频")

    except KeyboardInterrupt:
        print("\n\n⚠️  录制被用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
