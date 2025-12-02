# import random
# import typing
# import cv2

# import numpy as np
# from Game import GameCore


# class GameInterface:
#     ACTION_NUM = 16
#     SIMULATE_FPS = 60

#     FEATURE_MAP_WIDTH, FEATURE_MAP_HEIGHT = 16, 20

#     def __init__(self) -> None:
#         self.game = GameCore()
#         self.action_num = GameInterface.ACTION_NUM
#         self.action_segment_len = self.game.width / GameInterface.ACTION_NUM

#     def reset(self, seed: int = None) -> None:
#         self.game.reset(seed)

#     def simulate_until_stable(self) -> None:
#         self.game.update_until_stable(GameInterface.SIMULATE_FPS)

#     def decode_action(self, action: int) -> typing.Tuple[int, int]:
#         x = int((action + 0.5) * self.action_segment_len)

#         return (x, 0)

#     def next(self, action: int) -> typing.Tuple[np.ndarray, int, bool]:
#         current_fruit = self.game.current_fruit_type

#         score_1 = self.game.score

#         self.game.click(self.decode_action(action))
#         self.simulate_until_stable()

#         feature = self.game.get_features(
#             GameInterface.FEATURE_MAP_WIDTH, GameInterface.FEATURE_MAP_HEIGHT
#         )

#         score_2 = self.game.score

#         score, reward, alive = self.game.score, score_2 - score_1, self.game.alive

#         # reward = reward if reward > 0 else -current_fruit

#         flatten_feature = feature.flatten().astype(np.float32)
#         # flatten_feature = np.expand_dims(feature.flatten(), axis=0).astype(np.float32)

#         return flatten_feature, reward, alive

#     def auto_play(self):
#         WINNAME, VIDEO_FPS = "fruit-merger", 5
#         cv2.namedWindow(WINNAME)

#         while True:
#             action = random.randint(0, self.action_num - 1)
#             feature, reward, alive = self.next(action)

#             self.game.draw()
#             cv2.imshow(WINNAME, self.game.__screen)

#             key = cv2.waitKey(int(1000 / VIDEO_FPS))

#             print(feature.shape)

#             if not alive:
#                 self.game.rclick((0, 0))

#             # if key != -1:
#             #     print(key)

#             if key == ord("q") or key == 27:
#                 break
#             # close the window
#             if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
#                 break

#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     gi = GameInterface()
#     gi.auto_play()

import random
import typing
import cv2

import numpy as np
from Game import GameCore


class GameInterface:
    ACTION_NUM = 16
    SIMULATE_FPS = 60

    FEATURE_MAP_WIDTH, FEATURE_MAP_HEIGHT = 16, 20

    def __init__(self, reward_mode: int = 1) -> None:
        self.game = GameCore()
        self.action_num = GameInterface.ACTION_NUM
        self.action_segment_len = self.game.width / GameInterface.ACTION_NUM
        self.consecutive_merges = 0  # 追踪连续合成次数
        self.prev_fruit_count = 0  # 记录上一步的水果数量

        # NEW: reward mode selector (1, 2, or 3)
        self.reward_mode = reward_mode

    def reset(self, seed: int = None) -> None:
        self.game.reset(seed)
        self.consecutive_merges = 0
        self.prev_fruit_count = 0

    def simulate_until_stable(self) -> None:
        self.game.update_until_stable(GameInterface.SIMULATE_FPS)

    def decode_action(self, action: int) -> typing.Tuple[int, int]:
        x = int((action + 0.5) * self.action_segment_len)

        return (x, 0)

    def calculate_space_utilization(self) -> float:
        """计算空间利用率 (0-1之间)"""
        if len(self.game.fruits) == 0:
            return 0.0

        total_area = self.game.width * self.game.height
        occupied_area = sum([np.pi * f.r ** 2 for f in self.game.fruits])

        return min(occupied_area / total_area, 1.0)

    def calculate_max_height_ratio(self) -> float:
        """计算最高水果位置占总高度的比例 (0-1之间)"""
        if len(self.game.fruits) == 0:
            return 0.0

        min_y = min([f.y - f.r for f in self.game.fruits])
        # 注意：y轴是从上到下增长的，所以min_y越小表示越高
        height_ratio = 1.0 - (min_y / self.game.height)

        return max(0.0, min(height_ratio, 1.0))

    def calculate_comprehensive_reward(self,
                                      score_delta: int,
                                      current_fruit: int,
                                      alive: bool,
                                      prev_score: int) -> float:
        """
        计算综合奖励

        组成部分：
        1. 合成奖励：指数增长 (2^fruit_type)
        2. 空间惩罚：-填充率 × 10
        3. 高度惩罚：-最高位置比例 × 5
        4. 连续合成奖励：combo × 3
        5. 无效操作惩罚：-current_fruit × 3
        """
        reward = 0.0

        # 1. 合成奖励（指数增长）
        if score_delta > 0:
            # score_delta就是新合成水果的type值（或100）
            if score_delta >= 100:
                merge_reward = 2 ** 11  # 最大水果
            else:
                merge_reward = 2 ** score_delta
            reward += merge_reward

            # 更新连续合成计数
            self.consecutive_merges += 1
        else:
            # 没有合成，重置连续合成计数
            self.consecutive_merges = 0

            # 5. 无效操作惩罚（增强版）
            invalid_penalty = -current_fruit * 3
            reward += invalid_penalty

        # 2. 空间惩罚
        space_utilization = self.calculate_space_utilization()
        space_penalty = -space_utilization * 10
        reward += space_penalty

        # 3. 高度惩罚
        height_ratio = self.calculate_max_height_ratio()
        height_penalty = -height_ratio * 5
        reward += height_penalty

        # 4. 连续合成奖励
        if self.consecutive_merges > 1:
            combo_bonus = (self.consecutive_merges - 1) * 3
            reward += combo_bonus

        return reward
    
    def calculate_reward_mode2(self,
                               score_delta: int,
                               current_fruit: int,
                               alive: bool,
                               prev_score: int) -> float:
        """
        奖励模式 2 组成部分：
        只用分数变化 + 终局惩罚
        """
        reward = 0.0
        reward += float(score_delta)

        return reward

    def calculate_reward_mode3(self,
                               score_delta: int,
                               current_fruit: int,
                               alive: bool,
                               prev_score: int) -> float:
        """
        奖励模式 3（占位示例）：
        组成部分：
        1. 合成奖励：指数增长 (2^fruit_type)
        2. 空间惩罚：-填充率 × 10
        3. 高度惩罚：-最高位置比例 × 5
        4. 连续合成奖励：combo × 3
        """
        reward = 0.0

        # 1. 合成奖励（指数增长）
        if score_delta > 0:
            # score_delta就是新合成水果的type值（或100）
            if score_delta >= 100:
                merge_reward = 2 ** 11  # 最大水果
            else:
                merge_reward = 2 ** score_delta
            reward += merge_reward

            # 更新连续合成计数
            self.consecutive_merges += 1
        else:
            # 没有合成，重置连续合成计数
            self.consecutive_merges = 0

            # 5. 无效操作惩罚（增强版）
            invalid_penalty = -current_fruit * 3
            reward += invalid_penalty

        # 2. 空间惩罚
        space_utilization = self.calculate_space_utilization()
        space_penalty = -space_utilization * 10
        reward += space_penalty

        # 3. 高度惩罚
        height_ratio = self.calculate_max_height_ratio()
        height_penalty = -height_ratio * 5
        reward += height_penalty

        # 4. 连续合成奖励
        if self.consecutive_merges > 1:
            combo_bonus = (self.consecutive_merges - 1) * 3
            reward += combo_bonus
        
        return reward

    def next(self, action: int) -> typing.Tuple[np.ndarray, int, bool]:
        current_fruit = self.game.current_fruit_type

        score_1 = self.game.score

        self.game.click(self.decode_action(action))
        self.simulate_until_stable()

        feature = self.game.get_features(
            GameInterface.FEATURE_MAP_WIDTH, GameInterface.FEATURE_MAP_HEIGHT
        )

        score_2 = self.game.score
        score_delta = score_2 - score_1
        alive = self.game.alive

        # 根据 reward_mode 选择不同的奖励函数
        if self.reward_mode == 1:
            reward = self.calculate_comprehensive_reward(
                score_delta, current_fruit, alive, score_1
            )
        elif self.reward_mode == 2:
            reward = self.calculate_reward_mode2(
                score_delta, current_fruit, alive, score_1
            )
        elif self.reward_mode == 3:
            reward = self.calculate_reward_mode3(
                score_delta, current_fruit, alive, score_1
            )
        else:
            # fallback：非法值时默认用模式 2
            reward = self.calculate_reward_mode2(
                score_delta, current_fruit, alive, score_1
            )

        flatten_feature = feature.flatten().astype(np.float32)
        # flatten_feature = np.expand_dims(feature.flatten(), axis=0).astype(np.float32)

        return flatten_feature, reward, alive

    def auto_play(self):
        WINNAME, VIDEO_FPS = "fruit-merger", 5
        cv2.namedWindow(WINNAME)

        while True:
            action = random.randint(0, self.action_num - 1)
            feature, reward, alive = self.next(action)

            self.game.draw()
            cv2.imshow(WINNAME, self.game.__screen)

            key = cv2.waitKey(int(1000 / VIDEO_FPS))

            print(feature.shape)

            if not alive:
                self.game.rclick((0, 0))

            # if key != -1:
            #     print(key)

            if key == ord("q") or key == 27:
                break
            # close the window
            if cv2.getWindowProperty(WINNAME, cv2.WND_PROP_VISIBLE) <= 0:
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    gi = GameInterface()
    gi.auto_play()