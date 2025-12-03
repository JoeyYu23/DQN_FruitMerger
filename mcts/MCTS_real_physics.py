"""
Real Physics MCTS - 使用真实物理模拟的MCTS

关键特性：
1. 使用GameInterface的真实pymunk物理引擎
2. 真实的球体弹跳、碰撞、合并
3. 智能奖励系统：
   - 合并奖励
   - 位置优势奖励（小水果在大水果上）
   - 未来3步奖励累积
"""

import numpy as np
import math
import copy
from typing import List, Dict, Optional
from Game import FRUIT_RADIUS


class RealPhysicsConfig:
    """Real Physics MCTS配置"""

    # MCTS参数
    C_PUCT = 1.5                    # 探索-利用平衡
    NUM_SIMULATIONS = 256           # 每步模拟次数

    # Rollout参数
    ROLLOUT_STEPS = 2              # 每次rollout的步数（增加到5步，看得更远）
    FUTURE_STEPS = 5                # 计算未来奖励的步数

    # 奖励权重
    MERGE_REWARD = 100.0            # 合并基础奖励（大幅提高，鼓励合并）
    POSITION_REWARD = 10.0          # 位置优势奖励（提高，鼓励创造merge机会）
    HEIGHT_PENALTY = 2.0            # 高度惩罚（降低，让合并更重要）
    DEATH_PENALTY = 20000.0           # 游戏结束惩罚

    # 物理模拟参数
    PHYSICS_STEPS_PER_ACTION = 160   # 每个动作物理步数（pymunk steps）
    WAIT_FRAMES = 10                # 等待稳定的帧数


class RealPhysicsNode:
    """MCTS节点"""
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'total_value',
                 'children', 'game_state_snapshot']

    def __init__(self, parent=None, action: int = None, prior: float = 1.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, 'RealPhysicsNode'] = {}
        self.game_state_snapshot = None  # 保存游戏状态快照

    def get_value(self) -> float:
        """获取Q值"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def get_puct(self) -> float:
        """计算PUCT值"""
        if self.parent is None:
            return 0.0

        q = self.get_value()
        u = (RealPhysicsConfig.C_PUCT * self.prior *
             math.sqrt(self.parent.visit_count) / (1 + self.visit_count))
        return q + u

    def select_child(self):
        """选择最佳子节点 - 简化策略：未访问优先，然后选Q值最高"""
        # 如果有未访问的节点，随机选择一个
        unvisited = [child for child in self.children.values() if child.visit_count == 0]
        if unvisited:
            return np.random.choice(unvisited)

        # 所有都访问过了，选择Q值最高的
        return max(self.children.values(), key=lambda c: c.get_value())

    def expand(self, valid_actions: List[int]):
        """扩展节点"""
        for action in valid_actions:
            if action not in self.children:
                prior = 1.0 / len(valid_actions)
                self.children[action] = RealPhysicsNode(
                    parent=self, action=action, prior=prior
                )

    def update(self, value: float):
        """反向传播更新"""
        self.visit_count += 1
        self.total_value += value

    def best_action(self) -> int:
        """选择访问次数最多的动作"""
        if not self.children:
            return 8  # 默认中间位置
        return max(self.children.items(), key=lambda x: x[1].visit_count)[0]


class RealPhysicsMCTS:
    """使用真实物理的MCTS"""

    def __init__(self):
        self.root: Optional[RealPhysicsNode] = None
        self.config = RealPhysicsConfig()

    def search(self, env, num_simulations: int) -> int:
        """
        两步前瞻搜索：
        1. 每个位置扔一个球
        2. 物理更新后再在每个位置扔一个球
        3. 累加两步reward，选最高的

        Args:
            env: GameInterface环境（真实物理）
            num_simulations: 忽略（不需要）

        Returns:
            最佳动作
        """
        # 保存当前环境状态
        original_state = self._save_state(env)

        # 检查每个横坐标位置的最顶部水果
        current_type = env.game.current_fruit_type
        action_segment_len = env.game.width / 16

        # 记录每个action位置的最上方（y值最小）的水果
        top_fruits_by_action = {}  # {action: (y, type)}

        for ball, fruit in zip(env.game.balls, env.game.fruits):
            x = ball.body.position.x
            y = ball.body.position.y
            action_pos = int(x / action_segment_len)
            action_pos = max(0, min(15, action_pos))

            # 更新这个位置的最顶部水果
            if action_pos not in top_fruits_by_action or y < top_fruits_by_action[action_pos][0]:
                top_fruits_by_action[action_pos] = (y, fruit.type)

        # 找出顶部是相同类型水果的位置
        top_match_actions = []
        for action_pos, (y, ftype) in top_fruits_by_action.items():
            if ftype == current_type:
                top_match_actions.append(action_pos)

        # -------------------------------
        # 🔥 新增：过滤危险 action（不要扔到会立即触顶死亡的列）
        # -------------------------------
        safe_actions = []
        danger_actions = []  # 记录哪些位置不能选

        for a in range(16):
            # 如果该列没有水果，则一定安全
            if a not in top_fruits_by_action:
                safe_actions.append(a)
                continue

            top_y, _ = top_fruits_by_action[a]

            # 如果顶部水果已经超过死亡线（init_y），则危险
            if top_y <= env.game.init_y:
                danger_actions.append(a)
            else:
                safe_actions.append(a)

        # 如果所有地方都危险（极端情况），还是允许所有动作避免死循环
        if len(safe_actions) == 0:
            actions_to_try = list(range(16))
        else:
            actions_to_try = safe_actions
        # -------------------------------

        # 标记能merge的位置（用于加成）
        merge_actions = set()
        if top_match_actions:
            merge_actions.update(top_match_actions)
        else:
            # 找所有能merge的位置（包括相邻）
            for ball, fruit in zip(env.game.balls, env.game.fruits):
                if fruit.type == current_type:
                    fruit_x = ball.body.position.x
                    action_pos = int(fruit_x / action_segment_len)
                    action_pos = max(0, min(15, action_pos))
                    for offset in [-1, 0, 1]:
                        action_idx = action_pos + offset
                        if 0 <= action_idx < 16:
                            merge_actions.add(action_idx)

        # ❗ 不允许在危险的位置做merge
        merge_actions = {m for m in merge_actions if m not in danger_actions}

        # 记录每个第一步action的总reward
        action_rewards = {}

        # 遍历要考虑的actions
        for action1 in actions_to_try:
            # 恢复到初始状态
            self._restore_state(env, original_state)

            # 记录初始状态
            score_before = env.game.score
            fruits_before = self._get_fruits_info(env)

            # 执行第一步
            self._apply_action(env, action1)

            # 检查第一步是否导致游戏结束
            if not env.game.alive:
                # 游戏结束，大惩罚
                action_rewards[action1] = -RealPhysicsConfig.DEATH_PENALTY
                continue

            # 计算第一步reward
            reward1 = self._calculate_reward(env, score_before, fruits_before)

            # 记录第一步后的状态
            state_after_step1 = self._save_state(env)
            score_after1 = env.game.score
            fruits_after1 = self._get_fruits_info(env)

            # 计算所有第二步的reward，找最大值
            max_reward2 = float('-inf')

            for action2 in range(16):
                # 恢复到第一步后的状态
                self._restore_state(env, state_after_step1)

                # 执行第二步
                self._apply_action(env, action2)

                # 检查第二步是否导致游戏结束
                if not env.game.alive:
                    # 游戏结束，大惩罚
                    reward2 = -RealPhysicsConfig.DEATH_PENALTY
                else:
                    # 计算第二步reward
                    reward2 = self._calculate_reward(env, score_after1, fruits_after1)

                # 更新最大值
                if reward2 > max_reward2:
                    max_reward2 = reward2

            # 总reward = 第一步reward + 第二步最大reward
            action_rewards[action1] = reward1 + max_reward2

        # 恢复状态
        self._restore_state(env, original_state)

        # 选择reward最高的action
        # 如果有能merge的位置，必须从这些位置中选；否则从所有位置中选
        # -------------------------------
        # 🔥 最终动作选择逻辑（确保不会选择危险区域）
        # -------------------------------

        safe_actions_set = set(actions_to_try)

        # 过滤 merge_actions，只保留安全的
        merge_actions = merge_actions & safe_actions_set

        if len(merge_actions) > 0:
            # 有可用的 merge 行为 → 强制从 merge 中选
            merge_rewards = {a: action_rewards[a] for a in merge_actions}
            best_action = max(merge_rewards.items(), key=lambda x: x[1])[0]

        else:
            # 没有 merge 动作 → 从所有安全动作中选
            if len(safe_actions_set) > 0:
                safe_rewards = {a: action_rewards[a] for a in safe_actions_set}
                best_action = max(safe_rewards.items(), key=lambda x: x[1])[0]
            else:
                # 极端情况：所有动作都是危险（一般不会发生）
                # → 退回使用全部 16 动作中最高 reward
                best_action = max(action_rewards.items(), key=lambda x: x[1])[0]

        return best_action

    def _simulate(self, env):
        """单次模拟"""
        node = self.root
        path = [node]

        # 1. Selection - 选择到叶节点
        while node.children and not self._is_terminal(env):
            node = node.select_child()
            path.append(node)

            # 在环境中执行动作
            if node.action is not None:
                self._apply_action(env, node.action)

        # 2. Expansion - 如果不是终止状态，扩展一个新节点
        if not self._is_terminal(env):
            # 检查是否有未expand的action
            valid_actions = self._get_valid_actions(env)
            unexpanded = [a for a in valid_actions if a not in node.children]

            if unexpanded:
                # 使用智能策略选择要expand的action（优先merge）
                action = self._rollout_policy(env)
                # 如果智能选择的action已经expand，随机选一个未expand的
                if action not in unexpanded:
                    action = np.random.choice(unexpanded)

                # 只expand这一个action
                node.expand([action])
                node = node.children[action]
                path.append(node)
                self._apply_action(env, action)

        # 3. Simulation (Rollout) - 使用真实物理模拟
        value = self._rollout(env)

        # 4. Backpropagation - 反向传播
        for n in path:
            n.update(value)

    def _rollout(self, env) -> float:
        """
        使用真实物理引擎的rollout

        每步：
        1. 执行动作（真实物理）
        2. 计算即时奖励
        3. 累积未来3步奖励
        """
        total_reward = 0.0
        rollout_steps = RealPhysicsConfig.ROLLOUT_STEPS

        for step in range(rollout_steps):
            if self._is_terminal(env):
                total_reward -= RealPhysicsConfig.DEATH_PENALTY
                break

            # 选择动作（简单策略：中间偏好）
            
            action = self._rollout_policy(env)

            # 记录执行前的状态
            score_before = env.game.score
            fruits_before = self._get_fruits_info(env)

            # 执行动作（真实物理）
            self._apply_action(env, action)

            # 计算奖励
            reward = self._calculate_reward(env, score_before, fruits_before)

            # 未来奖励衰减
            discount = 0.7 ** step
            total_reward += reward * discount

        return total_reward

    def _calculate_reward(self, env, score_before: float,
                         fruits_before: List[dict]) -> float:
        """
        计算智能奖励

        奖励来源：
        1. 合并奖励（得分增加）
        2. 位置优势奖励（小水果在大水果上）
        3. 高度惩罚
        """
        reward = 0.0

        # 1. 合并奖励（得分变化）
        score_delta = env.game.score - score_before
        if score_delta > 0:
            # 有合并发生！
            reward += score_delta * RealPhysicsConfig.MERGE_REWARD

        # 2. 位置优势奖励
        position_bonus = self._evaluate_positions(env, fruits_before)
        reward += position_bonus

        # 3. 高度惩罚
        height_penalty = self._calculate_height_penalty(env)
        reward -= height_penalty

        return reward

    def _evaluate_positions(self, env, fruits_before: List[dict]) -> float:
        """
        评估位置优势

        规则：
        - 小水果在上一级水果上面：+0.5奖励
        - 相同水果相邻：+1.0奖励
        """
        bonus = 0.0
        fruits_now = self._get_fruits_info(env)

        # 检查每个水果
        for fruit in fruits_now:
            fruit_type = fruit['type']
            fruit_y = fruit['y']

            # 检查下方是否有上一级水果
            for other in fruits_now:
                if other['type'] == fruit_type + 1:  # 上一级
                    # 检查是否在其上方
                    if fruit_y < other['y'] and abs(fruit['x'] - other['x']) < 30:
                        bonus += RealPhysicsConfig.POSITION_REWARD * 0.5

                # 检查是否有相同类型的水果相邻
                if other['type'] == fruit_type and fruit != other:
                    distance = math.sqrt(
                        (fruit['x'] - other['x'])**2 +
                        (fruit['y'] - other['y'])**2
                    )
                    if distance < 60:  # 相邻
                        bonus += RealPhysicsConfig.POSITION_REWARD * 1.0

        return bonus

    def _calculate_height_penalty(self, env) -> float:
        """计算高度惩罚"""
        balls = env.game.balls
        if not balls:
            return 0.0

        # 找最高点
        min_y = min(b.body.position.y for b in balls)

        # 游戏高度
        game_height = env.game.height
        warning_line_y = env.game.init_y  # 红线位置（0.15 * height）
        # print(f"Warning Line Y: {warning_line_y}, Min Fruit Y: {min_y}")
        # 计算占用比例

        height_ratio = (min_y-warning_line_y ) / warning_line_y
        # print(f"Height Ratio: {height_ratio}")

        # 指数惩罚
        penalty = abs(RealPhysicsConfig.HEIGHT_PENALTY /(height_ratio+0.01))*(-1)
        # print(f"Height Penalty: {penalty}")
        return penalty

    def _get_fruits_info(self, env) -> List[dict]:
        """获取所有水果信息"""
        fruits_info = []
        for ball, fruit in zip(env.game.balls, env.game.fruits):
            fruits_info.append({
                'type': fruit.type,
                'x': ball.body.position.x,
                'y': ball.body.position.y,
            })
        return fruits_info

    def _rollout_policy(self, env) -> int:
        """
        智能Rollout策略：只优先merge，无偏好
        """
        # 获取当前要扔的水果类型
        current_type = env.game.current_fruit_type

        # 获取场上所有水果信息
        fruits_info = self._get_fruits_info(env)

        # 初始化权重（所有位置相等）
        weights = np.ones(16)

        # 如果场上有相同类型的水果，大幅提高那些位置的权重
        action_segment_len = env.game.width / 16
        for fruit in fruits_info:
            if fruit['type'] == current_type:
                # 找到这个水果对应的action位置
                fruit_x = fruit['x']
                best_action = int(fruit_x / action_segment_len)
                best_action = max(0, min(15, best_action))  # 限制范围

                # 大幅提高该位置及相邻位置的权重（鼓励merge）
                for offset in [-1, 0, 1]:
                    action_idx = best_action + offset
                    if 0 <= action_idx < 16:
                        weights[action_idx] *= 100.0  # 提高100倍权重！

        # 归一化
        weights = weights / weights.sum()

        action = np.random.choice(16, p=weights)
        
        return action

    def _apply_action(self, env, action: int):
        """在环境中执行动作（真实物理）"""
        # 执行动作
        env.next(action)
        if not env.game.alive:
            return
        # 等待物理稳定
        for _ in range(RealPhysicsConfig.WAIT_FRAMES):
            env.game.space.step(1/60.0)

    def _get_valid_actions(self, env) -> List[int]:
        """获取有效动作"""
        # 所有16列都可用（真实物理会处理碰撞）
        return list(range(16))

    def _is_terminal(self, env) -> bool:
        """检查是否终止"""
        return not env.game.alive

    def _save_state(self, env) -> dict:
        """保存游戏状态"""
        game = env.game
        state = {
            'score': game.score,
            'alive': game.alive,
            'current_fruit_type': game.current_fruit_type,
            'largest_fruit_type': game.largest_fruit_type,
            'balls': []
        }

        # 保存所有水果信息
        for ball, fruit in zip(game.balls, game.fruits):
            ball_info = {
                'type': fruit.type,
                'x': ball.body.position.x,
                'y': ball.body.position.y,
                'vx': ball.body.velocity.x,
                'vy': ball.body.velocity.y,
                'angle': ball.body.angle,
                'angular_velocity': ball.body.angular_velocity,
            }
            state['balls'].append(ball_info)

        return state

    def _restore_state(self, env, state: dict):
        """恢复游戏状态"""
        game = env.game

        # 清除所有水果
        for ball in list(game.balls):
            game.space.remove(ball, ball.body)
        game.balls.clear()
        game.fruits.clear()

        # 恢复基本状态
        game.score = state['score']
        game.alive = state['alive']
        game.current_fruit_type = state['current_fruit_type']
        game.largest_fruit_type = state['largest_fruit_type']

        # 恢复水果
        from Game import Fruit
        for ball_info in state['balls']:
            # 创建ball
            ball = game.create_ball(
                game.space,
                ball_info['x'],
                ball_info['y'],
                radius=FRUIT_RADIUS[ball_info['type']],
                type=ball_info['type']
            )
            ball.body.velocity = (ball_info['vx'], ball_info['vy'])
            ball.body.angle = ball_info['angle']
            ball.body.angular_velocity = ball_info['angular_velocity']

            # 创建对应的fruit
            fruit = Fruit(ball_info['type'], ball_info['x'], ball_info['y'])

            game.balls.append(ball)
            game.fruits.append(fruit)


class RealPhysicsMCTSAgent:
    """Real Physics MCTS Agent"""

    def __init__(self, num_simulations: int = 100):
        self.mcts = RealPhysicsMCTS()
        self.num_simulations = num_simulations

    def predict(self, env) -> np.ndarray:
        """预测动作"""
        action = self.mcts.search(env, self.num_simulations)
        return np.array([action])

    def sample(self, env) -> np.ndarray:
        """采样动作"""
        return self.predict(env)


if __name__ == "__main__":
    print("="*70)
    print("Real Physics MCTS")
    print("="*70)
    print("\n配置:")
    print(f"  模拟次数: {RealPhysicsConfig.NUM_SIMULATIONS}")
    print(f"  Rollout步数: {RealPhysicsConfig.ROLLOUT_STEPS}")
    print(f"  未来步数: {RealPhysicsConfig.FUTURE_STEPS}")
    print(f"\n奖励:")
    print(f"  合并奖励: {RealPhysicsConfig.MERGE_REWARD}")
    print(f"  位置奖励: {RealPhysicsConfig.POSITION_REWARD}")
    print(f"  高度惩罚: {RealPhysicsConfig.HEIGHT_PENALTY}")
    print(f"  死亡惩罚: {RealPhysicsConfig.DEATH_PENALTY}")
    print("="*70)
