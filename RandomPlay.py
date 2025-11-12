"""
随机Agent自动玩游戏 - 无需训练模型
"""
import cv2
import numpy as np
from Game import GameCore, visualize_feature
from GameInterface import GameInterface
from render_utils import cover

# 随机Agent
class RandomAgent:
    def __init__(self, action_num):
        self.action_num = action_num

    def predict(self, feature):
        return np.random.randint(0, self.action_num)

if __name__ == "__main__":
    WINNAME = "🎮 随机AI玩水果合成"
    WINNAME2 = "🗺️ 特征地图"

    cv2.namedWindow(WINNAME)
    cv2.namedWindow(WINNAME2)

    # 初始化
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM

    env = GameInterface()
    agent = RandomAgent(action_dim)

    FPS = 10  # 降低速度，方便观看
    AUTO_RESTART = True  # 自动重新开始

    print("=" * 60)
    print("🎮 随机AI自动玩水果合成游戏")
    print("=" * 60)
    print("控制:")
    print("  空格键: 暂停/继续")
    print("  R 键: 重新开始")
    print("  Q/ESC: 退出")
    print("=" * 60)

    game_count = 0
    paused = False

    while True:
        game_count += 1
        print(f"\n🎲 第 {game_count} 局游戏开始...")

        env.reset()
        step = 0
        rewards_sum = 0
        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        while alive:
            step += 1

            # 显示特征图
            reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
            feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(np.uint8)
            cv2.imshow(WINNAME2, feature_img)

            # 显示游戏画面
            screen = env.game.draw()
            action = agent.predict(feature)

            # 标记AI选择的位置
            unit_w = 1.0 * env.game.width / action_dim
            red_rect = np.zeros_like(screen, dtype=np.uint8)
            red_rect = cv2.rectangle(
                red_rect,
                (int(action * unit_w), 0),
                (int((action + 1) * unit_w), env.game.height),
                (0, 0, 255, 80),
                -1,
            )
            cover(screen, red_rect, 1)

            # 显示信息
            info_text = f"Game: {game_count} | Step: {step} | Score: {env.game.score} | Reward: {int(rewards_sum)}"
            cv2.putText(screen, info_text, (5, env.game.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            cv2.imshow(WINNAME, screen)

            # 等待按键
            wait_time = 1000 // FPS if not paused else 0
            key = cv2.waitKey(wait_time)

            # 处理按键
            if key == ord('q') or key == 27:  # Q 或 ESC
                print("\n👋 退出游戏")
                cv2.destroyAllWindows()
                exit(0)
            elif key == ord(' '):  # 空格暂停
                paused = not paused
                print("⏸️  暂停" if paused else "▶️  继续")
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
        print(f"💀 游戏结束! 最终分数: {env.game.score}, 总奖励: {int(rewards_sum)}")

        # 显示游戏结束画面
        final_screen = env.game.draw()
        cv2.putText(final_screen, f"GAME OVER - Score: {env.game.score}",
                   (20, env.game.height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
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
