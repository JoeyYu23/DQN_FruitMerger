#!/usr/bin/env python3
"""
录制MCTS玩游戏的视频
展示决策过程和游戏画面
"""

import numpy as np
import cv2
from GameInterface import GameInterface
from MCTS_optimized import FastMCTSAgent
import os
import imageio
import time

def record_mcts_game(agent, env, seed, output_path, fps=10, show_tree=True):
    """
    录制MCTS玩游戏的视频

    Args:
        agent: MCTS智能体
        env: 游戏环境
        seed: 随机种子
        output_path: 输出视频路径
        fps: 帧率
        show_tree: 是否显示搜索树信息
    """
    print(f"\n🎬 开始录制 Seed={seed} 的游戏...")

    env.reset(seed=seed)

    frames = []
    step_count = 0
    reward_sum = 0

    # 第一步随机
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    while alive:
        step_count += 1

        # MCTS决策
        start_time = time.time()
        grid_action = agent.mcts.search(agent._convert_state(env),
                                       agent.num_simulations)
        think_time = time.time() - start_time

        # 绘制游戏画面
        screen = env.game.draw()

        # 获取搜索树信息
        root = agent.mcts.root
        if root and root.children and show_tree:
            # 绘制决策信息面板
            info_panel = np.zeros((screen.shape[0], 300, 4), dtype=np.uint8)
            info_panel[:, :, 3] = 200  # 半透明背景

            # 标题
            cv2.putText(info_panel, "MCTS Decision", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2)

            # 统计信息
            y = 50
            cv2.putText(info_panel, f"Simulations: {root.visit_count}",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (200, 200, 200, 255), 1)
            y += 20
            cv2.putText(info_panel, f"Think: {think_time:.2f}s",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (200, 200, 200, 255), 1)
            y += 20
            cv2.putText(info_panel, f"Speed: {agent.num_simulations/think_time:.0f} r/s",
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (200, 200, 200, 255), 1)

            # Top候选动作
            y += 35
            cv2.putText(info_panel, "Top Actions:", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100, 255), 1)

            sorted_children = sorted(root.children.items(),
                                    key=lambda x: x[1].visit_count,
                                    reverse=True)

            total_visits = sum(c.visit_count for _, c in root.children.items())

            y += 20
            for idx, (act, child) in enumerate(sorted_children[:5], 1):
                visit_rate = child.visit_count / total_visits * 100 if total_visits > 0 else 0
                q_val = child.get_value()

                # 标记最佳选择
                if idx == 1:
                    color = (0, 255, 0, 255)  # 绿色
                    marker = ">"
                else:
                    color = (200, 200, 200, 255)
                    marker = " "

                text = f"{marker}Col{act}: {child.visit_count:3d} ({visit_rate:4.0f}%)"
                cv2.putText(info_panel, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y += 15

            # Q值条形图
            y += 10
            cv2.putText(info_panel, "Q Values:", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 100, 255), 1)
            y += 20

            # 绘制Q值条形图
            max_q = max(c.get_value() for c in root.children.values()) if root.children else 1
            min_q = min(c.get_value() for c in root.children.values()) if root.children else 0
            q_range = max(max_q - min_q, 1)

            bar_height = 12
            for idx, (act, child) in enumerate(sorted_children[:10]):
                q_val = child.get_value()
                bar_width = int((q_val - min_q) / q_range * 200)

                # 条形图
                color = (0, 255, 0) if idx == 0 else (100, 100, 255)
                cv2.rectangle(info_panel,
                             (80, y),
                             (80 + max(bar_width, 1), y + bar_height),
                             (*color, 255), -1)

                # 标签
                cv2.putText(info_panel, f"C{act}", (10, y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200, 255), 1)
                cv2.putText(info_panel, f"{q_val:.0f}", (85 + bar_width, y + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200, 255), 1)

                y += bar_height + 3

            # 合并信息面板
            screen_with_info = np.zeros((screen.shape[0], screen.shape[1] + 300, 4),
                                       dtype=np.uint8)
            screen_with_info[:, :screen.shape[1], :] = screen
            screen_with_info[:, screen.shape[1]:, :] = info_panel
            screen = screen_with_info

        # 在主画面上绘制选择的列
        unit_w = env.game.width / 10  # Grid width
        best_rect = np.zeros_like(screen[:, :screen.shape[1]//2 if show_tree else screen.shape[1]],
                                  dtype=np.uint8)
        cv2.rectangle(best_rect,
                     (int(grid_action * unit_w), 0),
                     (int((grid_action + 1) * unit_w), env.game.height),
                     (0, 255, 0, 150), 3)

        from render_utils import cover
        cover(screen[:, :env.game.width], best_rect, 1)

        # 添加游戏信息
        info_text = f"Step:{step_count:3d} Score:{env.game.score:4d} Reward:{int(reward_sum):5d}"
        cv2.putText(screen, info_text, (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1)

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
            print(f"  步骤 {step_count}, 帧数 {len(frames)}, 得分 {env.game.score}", end='\r')

    # 最后一帧（游戏结束）
    final_screen = env.game.draw()

    # 添加游戏结束信息
    cv2.putText(final_screen, "GAME OVER",
               (env.game.width // 2 - 70, env.game.height // 2 - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255, 255), 3)
    cv2.putText(final_screen, f"Score: {env.game.score}",
               (env.game.width // 2 - 60, env.game.height // 2 + 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    cv2.putText(final_screen, f"Steps: {step_count}",
               (env.game.width // 2 - 50, env.game.height // 2 + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200, 255), 2)

    final_rgb = cv2.cvtColor(final_screen, cv2.COLOR_BGRA2RGB)

    # 结束画面保持3秒
    for _ in range(fps * 3):
        frames.append(final_rgb)

    print(f"\n  正在保存视频到 {output_path}...")

    # 保存视频
    try:
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
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None


if __name__ == "__main__":
    print("="*70)
    print("🎥 MCTS游戏录制")
    print("="*70)

    # 创建目录
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # 创建智能体
    agent = FastMCTSAgent(num_simulations=200)
    env = GameInterface()

    # 录制3局游戏
    seeds = [888, 999, 1234]

    print(f"\n将录制 {len(seeds)} 局游戏:")
    for i, seed in enumerate(seeds, 1):
        print(f"  #{i}: Seed={seed}")

    videos = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n[{i}/{len(seeds)}] 录制第{i}局...")
        output_path = f"videos/mcts_seed{seed}.mp4"

        result = record_mcts_game(agent, env, seed, output_path, fps=10, show_tree=True)
        if result:
            videos.append(result)

    print(f"\n{'='*70}")
    print("✅ 所有视频录制完成!")
    print(f"{'='*70}")

    if videos:
        total_size = sum([v['size_mb'] for v in videos])
        avg_score = sum([v['score'] for v in videos]) / len(videos)

        print(f"\n📊 统计:")
        print(f"  视频数量: {len(videos)}")
        print(f"  总大小: {total_size:.2f}MB")
        print(f"  平均得分: {avg_score:.0f}")
        print(f"  保存位置: {os.path.abspath('videos')}/\n")

        for v in videos:
            print(f"  {os.path.basename(v['file'])}: "
                  f"得分{v['score']}, {v['size_mb']:.2f}MB")

        print(f"\n🎊 完成! 可以播放videos目录中的视频查看MCTS表现")
        print(f"\n播放命令: open videos/mcts_seed888.mp4")
