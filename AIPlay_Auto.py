"""
DQN AI自动连续玩游戏 - 自动模式
"""
import cv2
import numpy as np
from Game import visualize_feature
from GameInterface import GameInterface
from DQN import Agent, build_model
import paddle
from render_utils import cover

if __name__ == "__main__":
    WINNAME = "🤖 DQN AI 自动玩水果合成"
    WINNAME2 = "🗺️ AI特征视图"

    cv2.namedWindow(WINNAME)
    cv2.namedWindow(WINNAME2)

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    # 降低探索率，让AI更多使用学到的策略
    e_greed = 0.0  # 不再探索，完全使用学到的策略
    e_greed_decrement = 0

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    model_path = "final.pdparams"
    print(f"📦 加载模型: {model_path}")
    agent.policy_net.set_state_dict(paddle.load(model_path))
    print("✅ 模型加载成功!")

    FPS = 8  # 每秒8帧，可以调整速度
    AUTO_RESTART = True  # 自动重新开始
    paused = False

    print("\n" + "=" * 60)
    print("🤖 DQN AI 自动玩水果合成游戏")
    print("=" * 60)
    print("控制:")
    print("  空格键: 暂停/继续")
    print("  + 键: 加速 (提高FPS)")
    print("  - 键: 减速 (降低FPS)")
    print("  R 键: 重新开始")
    print("  Q/ESC: 退出")
    print("=" * 60)

    game_count = 0

    while True:
        game_count += 1
        print(f"\n🎮 第 {game_count} 局游戏开始...")

        env.reset()
        step, rewards_sum = 0, 0
        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        assert alive

        while alive:
            step += 1

            # 显示特征图
            reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
            feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(
                np.uint8
            )
            cv2.imshow(WINNAME2, feature_img)

            # 显示游戏画面
            screen = env.game.draw()

            # 使用predict而不是sample，确保使用最佳策略
            action = agent.predict(feature)

            # 确保action是标量
            if isinstance(action, np.ndarray):
                action = action.item()

            unit_w = 1.0 * env.game.width / action_dim

            # 标记AI选择的位置
            red_rect = np.zeros_like(screen, dtype=np.uint8)
            red_rect = cv2.rectangle(
                red_rect,
                (int(action * unit_w), 0),
                (int((action + 1) * unit_w), env.game.height),
                (0, 0, 255, 80),
                -1,
            )
            cover(screen, red_rect, 1)

            # 显示游戏信息
            info_text = f"Game:{game_count} Step:{step} Score:{env.game.score} Reward:{int(rewards_sum)} FPS:{FPS}"
            cv2.putText(screen, info_text, (5, env.game.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            if paused:
                cv2.putText(screen, "PAUSED", (env.game.width // 2 - 40, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow(WINNAME, screen)

            # 等待按键（自动模式）
            wait_time = 1000 // FPS if not paused else 0
            key = cv2.waitKey(wait_time)

            # 处理按键
            if key == ord('q') or key == 27:  # Q 或 ESC
                print("\n👋 退出游戏")
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord(' '):  # 空格暂停
                paused = not paused
                status = "⏸️  暂停" if paused else "▶️  继续"
                print(f"{status} (当前FPS: {FPS})")
            elif key == ord('+') or key == ord('='):  # 加速
                FPS = min(FPS + 2, 30)
                print(f"⚡ 加速! FPS: {FPS}")
            elif key == ord('-') or key == ord('_'):  # 减速
                FPS = max(FPS - 2, 1)
                print(f"🐌 减速! FPS: {FPS}")
            elif key == ord('r'):  # R 重新开始
                print("🔄 重新开始")
                break

            # 检查窗口是否关闭
            if (cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0 or
                cv2.getWindowProperty(WINNAME2, cv2.WND_PROP_VISIBLE) <= 0):
                print("\n👋 窗口已关闭")
                cv2.destroyAllWindows()
                exit(0)

            # 继续游戏
            if not paused:
                next_feature, reward, alive = env.next(action)

                reward_sum = np.sum(reward)
                rewards_sum += reward_sum

                feature = next_feature

        # 游戏结束
        print(f"💀 游戏结束! 最终分数: {env.game.score}, 总奖励: {int(rewards_sum)}, 步数: {step}")

        # 显示游戏结束画面
        final_screen = env.game.draw()
        cv2.putText(final_screen, f"GAME OVER - Score: {env.game.score}",
                   (20, env.game.height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if AUTO_RESTART:
            cv2.putText(final_screen, "3 seconds to restart...",
                       (30, env.game.height // 2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.putText(final_screen, "Press R to restart, Q to quit",
                       (20, env.game.height // 2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.imshow(WINNAME, final_screen)

        # 等待用户选择
        if AUTO_RESTART:
            print("⏱️  3秒后自动重新开始...")
            key = cv2.waitKey(3000)
        else:
            key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        elif cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
            break

    cv2.destroyAllWindows()
    print(f"\n📊 共玩了 {game_count} 局游戏")
    print("👋 再见!")
