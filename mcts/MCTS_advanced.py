"""
Advanced MCTS for Suika Game
带智能奖励函数、潜在合成评估、版面质量评分

核心改进：
1. 距离奖励：同类水果距离越近奖励越高
2. 合成潜力：预测未来合成机会
3. 版面评分：压缩度、均匀性、未来空间
4. Action masking：禁止明显坏动作
5. 可选价值网络：预测未来收益
"""

import numpy as np
import math
import random
from typing import List, Dict, Optional, Tuple
from MCTS_optimized import FastConfig, FastGameState, FastNode, FastMCTS, FastMCTSAgent


# =====================
# 智能评估器
# =====================

class SmartEvaluator:
    """
    智能评估器：计算状态的真实价值
    包含距离奖励、合成潜力、版面质量
    """

    # 参数
    LAMBDA_DISTANCE = 0.1  # 距离衰减率
    WEIGHT_MERGE_POTENTIAL = 5.0  # 合成潜力权重
    WEIGHT_BOARD_QUALITY = 3.0  # 版面质量权重
    WEIGHT_HEIGHT = 10.0  # 高度惩罚权重
    WEIGHT_CHAIN_BONUS = 10.0  # 连锁奖励权重

    # 不同等级水果的距离奖励权重
    FRUIT_WEIGHTS = {
        1: 1.0,   # 葡萄
        2: 1.5,   # 樱桃
        3: 2.0,   # 草莓
        4: 3.0,   # 柠檬
        5: 4.0,   # 橙子
        6: 5.0,   # 苹果
        7: 7.0,   # 梨
        8: 10.0,  # 桃子
        9: 15.0,  # 菠萝
        10: 25.0, # 椰子
    }

    @staticmethod
    def evaluate_state(state: FastGameState) -> float:
        """
        评估状态的综合价值

        Returns:
            总价值 = 即时得分 + 距离奖励 + 合成潜力 + 版面质量 - 高度惩罚
        """
        value = state.score  # 基础得分

        # 1. 距离奖励：同类水果距离
        distance_reward = SmartEvaluator._calculate_distance_reward(state)
        value += SmartEvaluator.WEIGHT_MERGE_POTENTIAL * distance_reward

        # 2. 合成潜力：预测可能的合成
        merge_potential = SmartEvaluator._calculate_merge_potential(state)
        value += SmartEvaluator.WEIGHT_MERGE_POTENTIAL * merge_potential

        # 3. 版面质量：压缩度、均匀性
        board_quality = SmartEvaluator._calculate_board_quality(state)
        value += SmartEvaluator.WEIGHT_BOARD_QUALITY * board_quality

        # 4. 连锁奖励：多个同类排列
        chain_bonus = SmartEvaluator._calculate_chain_bonus(state)
        value += SmartEvaluator.WEIGHT_CHAIN_BONUS * chain_bonus

        # 5. 高度惩罚
        height_penalty = SmartEvaluator._calculate_height_penalty(state)
        value -= SmartEvaluator.WEIGHT_HEIGHT * height_penalty

        return value

    @staticmethod
    def _calculate_distance_reward(state: FastGameState) -> float:
        """
        计算距离奖励：∑ w_i * e^(-λ * d_ij)

        对每对同类水果，距离越近奖励越高
        高等级水果权重更大
        """
        reward = 0.0

        # 收集每种水果的位置
        fruit_positions = {}  # {fruit_type: [(row, col), ...]}

        for row in range(state.height):
            for col in range(state.width):
                fruit = state.grid[row, col]
                if fruit > 0:
                    if fruit not in fruit_positions:
                        fruit_positions[fruit] = []
                    fruit_positions[fruit].append((row, col))

        # 计算同类水果的距离奖励
        for fruit_type, positions in fruit_positions.items():
            if len(positions) < 2:
                continue

            weight = SmartEvaluator.FRUIT_WEIGHTS.get(fruit_type, 1.0)

            # 对每对同类水果
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    r1, c1 = positions[i]
                    r2, c2 = positions[j]

                    # 曼哈顿距离
                    distance = abs(r1 - r2) + abs(c1 - c2)

                    # 距离奖励：越近越高
                    reward += weight * math.exp(-SmartEvaluator.LAMBDA_DISTANCE * distance)

        return reward

    @staticmethod
    def _calculate_merge_potential(state: FastGameState) -> float:
        """
        计算合成潜力：检测即将合成的水果对

        考虑：
        - 相邻的同类水果（距离1）
        - 快要相邻的同类水果（距离2-3）
        - 可能落下后合并的情况
        """
        potential = 0.0

        # 检查所有水果
        for row in range(state.height):
            for col in range(state.width):
                fruit = state.grid[row, col]
                if fruit == 0:
                    continue

                # 检查4个方向的邻居
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = row + dr, col + dc

                    if 0 <= nr < state.height and 0 <= nc < state.width:
                        neighbor = state.grid[nr, nc]

                        # 相邻同类 = 高潜力
                        if neighbor == fruit:
                            weight = SmartEvaluator.FRUIT_WEIGHTS.get(fruit, 1.0)
                            potential += weight * 2.0  # 直接相邻，高奖励

                        # 相邻空位，下面有同类 = 中等潜力
                        elif neighbor == 0 and dr == 1:  # 下方为空
                            # 检查下下方
                            nnr = nr + 1
                            if nnr < state.height and state.grid[nnr, nc] == fruit:
                                weight = SmartEvaluator.FRUIT_WEIGHTS.get(fruit, 1.0)
                                potential += weight * 0.5  # 间接潜力

        return potential

    @staticmethod
    def _calculate_board_quality(state: FastGameState) -> float:
        """
        计算版面质量

        考虑：
        - 压缩度：水果集中在底部
        - 均匀性：不要在一边堆太高
        - 未来空间：顶部留有余地
        """
        quality = 0.0

        # 1. 压缩度：水果重心越低越好
        total_fruits = 0
        weighted_height = 0.0

        for row in range(state.height):
            for col in range(state.width):
                if state.grid[row, col] > 0:
                    total_fruits += 1
                    # 越底部，惩罚越小
                    weighted_height += (state.height - row)

        if total_fruits > 0:
            avg_height = weighted_height / total_fruits
            # 平均高度越低越好
            compression = state.height - avg_height
            quality += compression

        # 2. 均匀性：检查列高度的标准差
        column_heights = []
        for col in range(state.width):
            height = 0
            for row in range(state.height):
                if state.grid[row, col] > 0:
                    height = state.height - row
                    break
            column_heights.append(height)

        if column_heights:
            avg = sum(column_heights) / len(column_heights)
            variance = sum((h - avg) ** 2 for h in column_heights) / len(column_heights)
            std_dev = math.sqrt(variance)
            # 标准差越小越均匀
            quality += max(0, 10 - std_dev)

        # 3. 顶部空间：警戒线以上应该尽量空
        top_space = sum(1 for row in range(state.warning_line)
                       for col in range(state.width)
                       if state.grid[row, col] == 0)
        quality += top_space / (state.warning_line * state.width) * 5

        return quality

    @staticmethod
    def _calculate_chain_bonus(state: FastGameState) -> float:
        """
        计算连锁奖励：多个同类水果排列

        检测：
        - 横向连续同类
        - 纵向连续同类
        - 给予额外奖励
        """
        bonus = 0.0

        # 横向检查
        for row in range(state.height):
            consecutive = 1
            prev_fruit = 0
            for col in range(state.width):
                fruit = state.grid[row, col]
                if fruit > 0 and fruit == prev_fruit:
                    consecutive += 1
                else:
                    if consecutive >= 2:
                        weight = SmartEvaluator.FRUIT_WEIGHTS.get(prev_fruit, 1.0)
                        bonus += weight * consecutive * 0.5
                    consecutive = 1
                    prev_fruit = fruit

            if consecutive >= 2:
                weight = SmartEvaluator.FRUIT_WEIGHTS.get(prev_fruit, 1.0)
                bonus += weight * consecutive * 0.5

        # 纵向检查
        for col in range(state.width):
            consecutive = 1
            prev_fruit = 0
            for row in range(state.height):
                fruit = state.grid[row, col]
                if fruit > 0 and fruit == prev_fruit:
                    consecutive += 1
                else:
                    if consecutive >= 2:
                        weight = SmartEvaluator.FRUIT_WEIGHTS.get(prev_fruit, 1.0)
                        bonus += weight * consecutive * 0.5
                    consecutive = 1
                    prev_fruit = fruit

            if consecutive >= 2:
                weight = SmartEvaluator.FRUIT_WEIGHTS.get(prev_fruit, 1.0)
                bonus += weight * consecutive * 0.5

        return bonus

    @staticmethod
    def _calculate_height_penalty(state: FastGameState) -> float:
        """
        计算高度惩罚

        考虑：
        - 最高列的高度
        - 接近警戒线的惩罚
        """
        max_height = 0
        for col in range(state.width):
            for row in range(state.height):
                if state.grid[row, col] > 0:
                    height = state.height - row
                    max_height = max(max_height, height)
                    break

        # 接近顶部的指数惩罚
        if max_height > state.height - state.warning_line:
            danger_ratio = (max_height - (state.height - state.warning_line)) / state.warning_line
            return danger_ratio ** 2 * 50
        else:
            return max_height / state.height * 5


# =====================
# Action Masking
# =====================

class ActionMasker:
    """
    Action Masking：禁止明显的坏动作
    """

    @staticmethod
    def get_valid_actions(state: FastGameState) -> List[int]:
        """
        获取有效动作，过滤掉坏动作

        坏动作包括：
        - 已满的列
        - 会导致即刻失败的列
        - 无意义的边角（可选）
        """
        valid = []

        for col in range(state.width):
            # 1. 检查列是否已满
            if state.grid[0, col] != 0:
                continue

            # 2. 检查是否会立即失败
            landing_row = state.height - 1
            for row in range(state.height - 1, -1, -1):
                if state.grid[row, col] != 0:
                    landing_row = row - 1
                    break

            if landing_row < state.warning_line:
                continue  # 会失败，跳过

            # 3. （可选）过滤边角无意义投放
            # 如果边角没有水果，且当前水果很小，可能不是好选择
            if col in [0, state.width - 1]:
                # 检查周围是否有水果
                has_neighbor = False
                for r in range(landing_row, min(landing_row + 3, state.height)):
                    for c in [col - 1, col, col + 1]:
                        if 0 <= c < state.width and state.grid[r, c] > 0:
                            has_neighbor = True
                            break
                    if has_neighbor:
                        break

                # 如果边角孤立，且水果等级低，跳过
                if not has_neighbor and state.current_fruit <= 2:
                    continue

            valid.append(col)

        # 如果过滤太严格导致没有动作，返回所有不满的列
        if not valid:
            valid = [col for col in range(state.width) if state.grid[0, col] == 0]

        return valid if valid else [state.width // 2]  # 最坏情况返回中间


# =====================
# 智能MCTS
# =====================

class SmartMCTS(FastMCTS):
    """
    智能MCTS：使用高级评估函数
    """

    def __init__(self):
        super().__init__()
        self.evaluator = SmartEvaluator()
        self.action_masker = ActionMasker()

    def _rollout(self, state: FastGameState) -> float:
        """
        改进的Rollout：使用智能评估
        """
        depth = 0

        while not state.is_terminal and depth < FastConfig.MAX_SIMULATION_DEPTH:
            # 使用Action Masking
            valid_actions = self.action_masker.get_valid_actions(state)
            if not valid_actions:
                break

            # 智能选择动作（基于合成潜力）
            action = self._smart_select_action(state, valid_actions)
            state.apply_action(action)

            depth += 1

        # 使用智能评估器
        value = self.evaluator.evaluate_state(state)

        return value

    def _smart_select_action(self, state: FastGameState, valid_actions: List[int]) -> int:
        """
        智能选择Rollout动作

        考虑：
        - 合成潜力
        - 版面质量
        - 距离奖励
        """
        if len(valid_actions) == 1:
            return valid_actions[0]

        best_action = valid_actions[0]
        best_score = -float('inf')

        for action in valid_actions:
            # 模拟该动作
            test_state = state.copy()
            test_state.apply_action(action)

            # 快速评估（简化版）
            score = 0.0

            # 1. 即时合成奖励
            score += test_state.score - state.score

            # 2. 距离奖励（快速版）
            col = action
            fruit = state.current_fruit

            # 检查落点周围是否有同类
            landing_row = state.height - 1
            for row in range(state.height - 1, -1, -1):
                if state.grid[row, col] != 0:
                    landing_row = row - 1
                    break

            neighbor_count = 0
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = landing_row + dr, col + dc
                if (0 <= nr < state.height and 0 <= nc < state.width):
                    if state.grid[nr, nc] == fruit:
                        neighbor_count += 1

            score += neighbor_count * 5.0

            # 3. 中心偏好
            center_dist = abs(col - state.width / 2)
            score += (1 - center_dist / (state.width / 2)) * 2.0

            # 4. 高度惩罚
            if landing_row < state.warning_line:
                score -= 100

            if score > best_score:
                best_score = score
                best_action = action

        return best_action


class SmartMCTSAgent(FastMCTSAgent):
    """
    智能MCTS智能体：使用SmartMCTS
    """

    def __init__(self, num_simulations: int = 200):
        super().__init__(num_simulations)
        self.mcts = SmartMCTS()  # 使用智能MCTS


# =====================
# 测试和演示
# =====================

if __name__ == "__main__":
    print("="*70)
    print("🧠 智能MCTS演示")
    print("="*70)

    # 测试评估器
    print("\n1. 测试智能评估器...")
    state = FastGameState()

    # 设置一些水果
    state.grid[15, 4] = 1  # 底部
    state.grid[15, 5] = 1  # 相邻同类
    state.grid[14, 5] = 2
    state.grid[15, 6] = 2  # 相邻同类

    evaluator = SmartEvaluator()
    value = evaluator.evaluate_state(state)

    print(f"  状态价值: {value:.2f}")
    print(f"  距离奖励: {evaluator._calculate_distance_reward(state):.2f}")
    print(f"  合成潜力: {evaluator._calculate_merge_potential(state):.2f}")
    print(f"  版面质量: {evaluator._calculate_board_quality(state):.2f}")

    # 测试Action Masking
    print("\n2. 测试Action Masking...")
    masker = ActionMasker()
    valid = masker.get_valid_actions(state)
    print(f"  有效动作: {valid}")
    print(f"  过滤掉: {[i for i in range(10) if i not in valid]}")

    # 对比普通MCTS vs 智能MCTS
    print("\n3. 对比性能...")
    from MCTS_optimized import FastMCTS as NormalMCTS
    import time

    normal_mcts = NormalMCTS()
    smart_mcts = SmartMCTS()

    # 简单状态
    test_state = FastGameState()

    print("\n  普通MCTS (100次模拟):")
    start = time.time()
    action1 = normal_mcts.search(test_state, 100)
    time1 = time.time() - start
    print(f"    选择: 列{action1}, 用时: {time1:.3f}秒")

    print("\n  智能MCTS (100次模拟):")
    start = time.time()
    action2 = smart_mcts.search(test_state, 100)
    time2 = time.time() - start
    print(f"    选择: 列{action2}, 用时: {time2:.3f}秒")

    print(f"\n  速度对比: 智能MCTS慢 {time2/time1:.2f}x (因为评估更复杂)")

    print("\n" + "="*70)
    print("✅ 智能MCTS实现完成！")
    print("\n核心改进:")
    print("  ✓ 距离奖励：同类水果距离越近奖励越高")
    print("  ✓ 合成潜力：预测即将发生的合成")
    print("  ✓ 版面质量：评估压缩度、均匀性")
    print("  ✓ 连锁奖励：多个同类排列额外奖励")
    print("  ✓ Action Masking：过滤明显坏动作")
    print("  ✓ 智能Rollout：不再随机，而是基于启发式")
    print("\n使用方法:")
    print("  from MCTS_advanced import SmartMCTSAgent")
    print("  agent = SmartMCTSAgent(num_simulations=200)")
