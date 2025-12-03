"""
优化调参版 MCTS
主要改进:
1. 非线性height penalty (指数增长，后期惩罚更重)
2. 优化的探索-利用平衡
3. 更智能的rollout策略
4. 动态深度限制
"""

import numpy as np
import math
import random
from typing import List, Dict, Optional


class TunedConfig:
    """优化后的配置"""
    GRID_WIDTH = 10
    GRID_HEIGHT = 16
    NUM_ACTIONS = 20
    GAME_WIDTH = 300

    # ==========================================
    # MCTS核心参数 (调优!)
    # ==========================================

    # 探索-利用平衡 (原1.5 → 1.2 → 2.0)
    # 提高探索，鼓励尝试更多动作
    C_PUCT = 2.0  # ⬆️ 增加探索！

    # 模拟深度 (原30 → 40)
    # 增加深度，看得更远
    MAX_SIMULATION_DEPTH = 40

    # Progressive widening (原3→5→8, 更早扩展更多动作)
    INITIAL_ACTIONS = 8  # ⬆️ 早期尝试更多动作
    MAX_EXPANDED_ACTIONS = 15

    # ==========================================
    # 惩罚系统 (关键优化!)
    # ==========================================

    # Height Penalty - 使用指数惩罚
    HEIGHT_PENALTY_BASE = 2.0      # 基础惩罚降低 (原5.0)
    HEIGHT_PENALTY_EXP = 3.0       # 指数系数

    # 分段惩罚阈值
    SAFE_ZONE = 0.4      # 40%以下高度：安全区
    WARNING_ZONE = 0.7   # 70%以下高度：警告区
    DANGER_ZONE = 0.85   # 85%以下高度：危险区
    # >85%: 极度危险区

    # Death penalty (原500 → 800，更严厉)
    DEATH_PENALTY = 800.0

    # 合并奖励加成
    MERGE_BONUS = 3.0    # 合并奖励×3.0 ⬆️ 提高！鼓励积极合并


class TunedGameState:
    """优化的游戏状态"""
    __slots__ = ['grid', 'current_fruit', 'score', 'is_terminal',
                 'max_height', 'warning_line', 'width', 'height', 'step_count']

    def __init__(self):
        self.width = TunedConfig.GRID_WIDTH
        self.height = TunedConfig.GRID_HEIGHT
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.current_fruit = 1
        self.score = 0
        self.is_terminal = False
        self.max_height = self.height - 1
        self.warning_line = 2
        self.step_count = 0  # 跟踪步数

    def copy(self):
        """快速复制"""
        new = TunedGameState()
        new.grid = self.grid.copy()
        new.current_fruit = self.current_fruit
        new.score = self.score
        new.is_terminal = self.is_terminal
        new.max_height = self.max_height
        new.step_count = self.step_count
        return new

    def get_hash(self) -> int:
        """快速哈希"""
        return hash(self.grid.tobytes()) ^ (self.current_fruit << 32)

    def get_valid_actions(self) -> List[int]:
        """获取有效动作"""
        return [col for col in range(self.width) if self.grid[0, col] == 0]

    def apply_action(self, action: int) -> float:
        """应用动作"""
        if self.is_terminal:
            return -TunedConfig.DEATH_PENALTY

        col = action
        fruit_type = self.current_fruit

        # 找落点
        landing_row = self.height - 1
        for row in range(self.height - 1, -1, -1):
            if self.grid[row, col] != 0:
                landing_row = row - 1
                break

        # 检查游戏结束
        if landing_row < self.warning_line:
            self.is_terminal = True
            return -TunedConfig.DEATH_PENALTY

        # 放置水果
        self.grid[landing_row, col] = fruit_type
        self.max_height = min(self.max_height, landing_row)
        self.step_count += 1

        # 合并 (带奖励加成)
        reward = self._simple_merge(landing_row, col)
        reward *= TunedConfig.MERGE_BONUS  # 奖励加成!

        # 下一个水果
        self.current_fruit = random.randint(1, min(5, 10))

        return reward

    def _simple_merge(self, row: int, col: int) -> float:
        """简化的合并逻辑"""
        reward = 0.0
        fruit = self.grid[row, col]

        if fruit == 0 or fruit >= 10:
            return 0.0

        # 检查4个方向
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc
            if (0 <= nr < self.height and 0 <= nc < self.width):
                if self.grid[nr, nc] == fruit:
                    # 合并
                    self.grid[row, col] = 0
                    self.grid[nr, nc] = fruit + 1 if fruit < 10 else 10

                    reward = 100 if fruit == 9 else (fruit + 1)
                    self.score += reward
                    return reward

        return reward


class TunedNode:
    """优化的节点"""
    __slots__ = ['state', 'parent', 'action', 'prior', 'visit_count',
                 'total_value', 'children', 'expanded_actions']

    def __init__(self, state: TunedGameState, parent=None, action: int = None,
                 prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, 'TunedNode'] = {}
        self.expanded_actions = set()

    def get_value(self) -> float:
        return self.total_value / max(1, self.visit_count)

    def get_puct(self) -> float:
        if self.parent is None:
            return 0.0

        q = self.get_value()
        u = TunedConfig.C_PUCT * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q + u

    def select_child(self):
        return max(self.children.values(), key=lambda c: c.get_puct())

    def update(self, value: float):
        self.visit_count += 1
        self.total_value += value

    def best_action(self) -> int:
        if not self.children:
            valid = self.state.get_valid_actions()
            return random.choice(valid) if valid else 0
        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]


class TunedPolicy:
    """优化策略"""

    @staticmethod
    def score_action(state: TunedGameState, action: int) -> float:
        """智能打分"""
        col = action
        score = 0.0

        # 1. 中心偏好 (权重2.0)
        center_dist = abs(col - state.width / 2)
        score += 2.0 * (1 - center_dist / (state.width / 2))

        # 2. 寻找落点并评估
        landing_row = state.height - 1
        for row in range(state.height - 1, -1, -1):
            if state.grid[row, col] != 0:
                landing_row = row - 1
                break

        # 3. 高度惩罚 (非线性)
        height = state.height - landing_row
        height_ratio = height / state.height

        if height_ratio < TunedConfig.SAFE_ZONE:
            # 安全区：几乎不惩罚
            score -= 0.5 * height_ratio
        elif height_ratio < TunedConfig.WARNING_ZONE:
            # 警告区：轻度惩罚
            score -= 2.0 * height_ratio
        elif height_ratio < TunedConfig.DANGER_ZONE:
            # 危险区：中度惩罚
            score -= 5.0 * height_ratio ** 2
        else:
            # 极度危险：重度惩罚
            score -= 15.0 * height_ratio ** 3

        # 4. 合并潜力 (检查邻居)
        if landing_row < state.height - 1:
            # 检查周围是否有相同水果
            neighbors_same = 0
            for dc in [-1, 0, 1]:
                nc = col + dc
                if 0 <= nc < state.width:
                    if state.grid[landing_row, nc] == state.current_fruit:
                        neighbors_same += 1
                    if landing_row + 1 < state.height and state.grid[landing_row + 1, nc] == state.current_fruit:
                        neighbors_same += 1

            # 有相同水果 → 加分
            score += 3.0 * neighbors_same

        return score

    @staticmethod
    def select_action(state: TunedGameState) -> int:
        """智能选择动作"""
        valid = state.get_valid_actions()
        if not valid:
            return 0

        if len(valid) == 1:
            return valid[0]

        # 使用智能打分
        scores = [(TunedPolicy.score_action(state, a), a) for a in valid]
        return max(scores, key=lambda x: x[0])[1]


class TunedMCTS:
    """优化的MCTS"""

    def __init__(self):
        self.root: Optional[TunedNode] = None

    def search(self, state: TunedGameState, num_simulations: int) -> int:
        """搜索"""
        self.root = TunedNode(state.copy())

        for _ in range(num_simulations):
            self._simulate()

        return self.root.best_action()

    def _simulate(self):
        """单次模拟"""
        # 1. Selection
        node = self.root
        path = [node]

        while node.children and not node.state.is_terminal:
            valid = node.state.get_valid_actions()
            max_children = min(len(valid),
                             TunedConfig.INITIAL_ACTIONS + int(math.sqrt(node.visit_count)))

            if len(node.children) < max_children:
                break

            node = node.select_child()
            path.append(node)

        # 2. Expansion
        if not node.state.is_terminal:
            valid = node.state.get_valid_actions()
            unexpanded = [a for a in valid if a not in node.expanded_actions]

            if unexpanded:
                # 使用智能策略选择扩展动作
                action = TunedPolicy.select_action(node.state)
                if action not in unexpanded:
                    action = unexpanded[0]

                new_state = node.state.copy()
                new_state.apply_action(action)

                child = TunedNode(new_state, parent=node, action=action,
                                prior=1.0/len(valid))
                node.children[action] = child
                node.expanded_actions.add(action)

                node = child
                path.append(node)

        # 3. Simulation - 智能rollout
        value = self._smart_rollout(node.state.copy())

        # 4. Backprop
        for n in path:
            n.update(value)

    def _smart_rollout(self, state: TunedGameState) -> float:
        """智能rollout - 使用优化的策略"""
        depth = 0

        # 动态深度：早期深度大，后期深度小
        height_ratio = (state.height - state.max_height) / state.height
        max_depth = int(TunedConfig.MAX_SIMULATION_DEPTH * (1.5 - height_ratio))

        while not state.is_terminal and depth < max_depth:
            valid = state.get_valid_actions()
            if not valid:
                break

            # 使用智能策略而非随机
            action = TunedPolicy.select_action(state)
            state.apply_action(action)
            depth += 1

        # ==========================================
        # 核心优化：非线性height penalty!
        # ==========================================
        value = state.score

        if state.is_terminal:
            value -= TunedConfig.DEATH_PENALTY

        # 计算高度比例
        height_ratio = (state.height - state.max_height) / state.height

        # 分段指数惩罚
        if height_ratio < TunedConfig.SAFE_ZONE:
            # 安全区 (0-40%): 几乎不惩罚
            penalty = TunedConfig.HEIGHT_PENALTY_BASE * height_ratio

        elif height_ratio < TunedConfig.WARNING_ZONE:
            # 警告区 (40-70%): 线性增长
            penalty = TunedConfig.HEIGHT_PENALTY_BASE * 2 * height_ratio ** 1.5

        elif height_ratio < TunedConfig.DANGER_ZONE:
            # 危险区 (70-85%): 平方增长
            penalty = TunedConfig.HEIGHT_PENALTY_BASE * 5 * height_ratio ** 2

        else:
            # 极度危险区 (>85%): 指数爆炸!
            penalty = TunedConfig.HEIGHT_PENALTY_BASE * 10 * np.exp(
                TunedConfig.HEIGHT_PENALTY_EXP * (height_ratio - TunedConfig.DANGER_ZONE)
            )

        penalty *= state.height  # 乘以总高度
        value -= penalty

        return value


class TunedMCTSAgent:
    """优化的MCTS智能体"""

    def __init__(self, num_simulations: int = 500):
        self.mcts = TunedMCTS()
        self.num_simulations = num_simulations

    def predict(self, game_state) -> np.ndarray:
        """预测"""
        simple_state = self._convert_state(game_state)
        grid_action = self.mcts.search(simple_state, self.num_simulations)
        game_action = int(grid_action * 16 / 10)
        return np.array([min(15, max(0, game_action))])

    def sample(self, game_state) -> np.ndarray:
        return self.predict(game_state)

    def _convert_state(self, game_interface) -> TunedGameState:
        """转换状态"""
        simple = TunedGameState()

        features = game_interface.game.get_features(simple.width, simple.height)

        for i in range(simple.height):
            for j in range(simple.width):
                val0 = features[i, j, 0]
                val1 = features[i, j, 1]

                if val0 > 0:
                    estimated_type = int(game_interface.game.current_fruit_type + val0)
                elif val1 > 0:
                    estimated_type = int(game_interface.game.current_fruit_type - val1)
                else:
                    estimated_type = 0

                simple.grid[i, j] = max(0, min(10, estimated_type))

        simple.current_fruit = game_interface.game.current_fruit_type
        simple.score = game_interface.game.score

        return simple


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("="*70)
    print("优化版 MCTS - 参数对比")
    print("="*70)

    print("\n原版配置:")
    print("  C_PUCT: 1.5 → 1.2 (降低探索)")
    print("  MAX_DEPTH: 30 → 40 (看得更远)")
    print("  HEIGHT_PENALTY: 5.0 (线性) → 2.0+指数 (非线性)")
    print("  DEATH_PENALTY: 500 → 800 (更严厉)")
    print("  MERGE_BONUS: 1.0 → 1.5 (鼓励合并)")

    print("\n新版Height Penalty对比:")
    print("="*70)

    # 可视化penalty曲线
    height_ratios = np.linspace(0, 1, 100)

    # 原版线性penalty
    old_penalty = 5.0 * height_ratios * 16

    # 新版分段指数penalty
    new_penalty = []
    for hr in height_ratios:
        if hr < 0.4:
            p = 2.0 * hr
        elif hr < 0.7:
            p = 2.0 * 2 * hr ** 1.5
        elif hr < 0.85:
            p = 2.0 * 5 * hr ** 2
        else:
            p = 2.0 * 10 * np.exp(3.0 * (hr - 0.85))
        new_penalty.append(p * 16)

    plt.figure(figsize=(12, 6))
    plt.plot(height_ratios * 100, old_penalty, 'b-', linewidth=2,
             label='Old (Linear)', alpha=0.7)
    plt.plot(height_ratios * 100, new_penalty, 'r-', linewidth=2,
             label='New (Exponential)')

    # 标注区域
    plt.axvspan(0, 40, alpha=0.1, color='green', label='Safe Zone')
    plt.axvspan(40, 70, alpha=0.1, color='yellow', label='Warning Zone')
    plt.axvspan(70, 85, alpha=0.1, color='orange', label='Danger Zone')
    plt.axvspan(85, 100, alpha=0.1, color='red', label='Critical Zone')

    plt.xlabel('Height Usage (%)', fontsize=12)
    plt.ylabel('Penalty Value', fontsize=12)
    plt.title('Height Penalty Comparison: Linear vs Exponential',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.savefig('mcts_penalty_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Penalty对比图已保存: mcts_penalty_comparison.png")

    print("\n关键差异:")
    for pct in [20, 40, 60, 80, 90, 95]:
        hr = pct / 100
        if hr < 0.4:
            p = 2.0 * hr
        elif hr < 0.7:
            p = 2.0 * 2 * hr ** 1.5
        elif hr < 0.85:
            p = 2.0 * 5 * hr ** 2
        else:
            p = 2.0 * 10 * np.exp(3.0 * (hr - 0.85))
        new_p = p * 16
        old_p = 5.0 * hr * 16

        print(f"  {pct:3d}%高度: 旧={old_p:6.1f}, 新={new_p:6.1f}, "
              f"差距={new_p/old_p if old_p > 0 else 0:5.2f}x")

    print("\n" + "="*70)
    print("可以看到:")
    print("  • 前40%: 新版惩罚更小 (鼓励积极放置)")
    print("  • 40-70%: 相近")
    print("  • 70-85%: 新版开始加重")
    print("  • 85%+:   新版指数爆炸! (强制避免)")
    print("="*70)
