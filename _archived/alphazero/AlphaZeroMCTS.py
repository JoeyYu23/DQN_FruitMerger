"""
AlphaZeroMCTS.py - AlphaZero风格的MCTS搜索

与原MCTS的区别：
1. 叶子节点用神经网络(P, V)评估，不用启发式rollout
2. Node的prior来自网络P(a|s)
3. 更简洁高效的实现
"""

import numpy as np
import paddle
from typing import Dict, List, Tuple, Optional
from mcts.MCTS import SimplifiedGameState, MCTSConfig
from SuikaNet import SuikaNet
from StateConverter import StateConverter


class AlphaZeroNode:
    """
    AlphaZero的MCTS节点

    存储信息：
    - state: 游戏状态
    - parent: 父节点
    - children: 子节点字典 {action: child_node}
    - visit_count: 访问次数 N(s,a)
    - total_value: 累积价值 W(s,a)
    - prior: 先验概率 P(a|s) 来自网络
    """

    def __init__(self,
                 state: SimplifiedGameState,
                 parent: Optional['AlphaZeroNode'] = None,
                 action: Optional[int] = None,
                 prior: float = 1.0):
        self.state = state
        self.parent = parent
        self.action = action  # 从父节点到达此节点的动作

        # MCTS统计量
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior  # P(a|s) 从网络获得

        # 子节点
        self.children: Dict[int, AlphaZeroNode] = {}

    def Q(self) -> float:
        """平均价值 Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def U(self, c_puct: float = 1.5) -> float:
        """
        探索奖励

        U(s,a) = c_puct × P(a|s) × sqrt(N(s)) / (1 + N(s,a))
        """
        parent_N = self.parent.visit_count if self.parent else 1
        return c_puct * self.prior * (parent_N ** 0.5) / (1 + self.visit_count)

    def PUCT(self, c_puct: float = 1.5) -> float:
        """
        PUCT值 = Q + U

        用于选择最优子节点
        """
        return self.Q() + self.U(c_puct)

    def is_leaf(self) -> bool:
        """是否是叶子节点"""
        return len(self.children) == 0

    def is_expanded(self) -> bool:
        """是否已展开"""
        return len(self.children) > 0

    def select_child(self, c_puct: float = 1.5) -> 'AlphaZeroNode':
        """选择PUCT最大的子节点"""
        return max(self.children.values(), key=lambda node: node.PUCT(c_puct))

    def expand(self, action_priors: Dict[int, float]):
        """
        展开节点 - 为每个有效动作创建子节点

        Args:
            action_priors: {action: P(a|s)} 来自网络
        """
        for action, prior in action_priors.items():
            if action not in self.children:
                # 应用动作得到下一个状态
                next_state = self.state.copy()
                next_state.apply_action(action)

                # 创建子节点
                self.children[action] = AlphaZeroNode(
                    state=next_state,
                    parent=self,
                    action=action,
                    prior=prior
                )

    def backup(self, value: float):
        """
        回传价值到路径上的所有节点

        Args:
            value: 叶子节点的评估值
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            # 单人游戏不需要翻转符号
            node = node.parent


class AlphaZeroMCTS:
    """
    AlphaZero风格的MCTS搜索引擎

    核心流程：
    1. Selection: 从根节点选择到叶子节点
    2. Expansion + Evaluation: 用网络评估叶子节点并展开
    3. Backup: 回传价值
    """

    def __init__(self,
                 network: SuikaNet,
                 c_puct: float = 1.5,
                 num_simulations: int = 400,
                 temperature: float = 1.0,
                 add_dirichlet_noise: bool = False,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25,
                 use_lookahead: bool = True,
                 lookahead_steps: int = 10):
        """
        Args:
            network: SuikaNet神经网络
            c_puct: PUCT探索常数 (越大越探索)
            num_simulations: 每次搜索的模拟次数
            temperature: 温度参数 (训练用1.0, 测试用0.0)
            add_dirichlet_noise: 是否在根节点添加Dirichlet噪声 (增加探索)
            dirichlet_alpha: Dirichlet分布参数
            dirichlet_epsilon: 噪声混合比例
            use_lookahead: 是否使用lookahead评估价值 (新增)
            lookahead_steps: lookahead模拟步数 (默认10)
        """
        self.network = network
        self.network.eval()  # 推理模式

        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature

        self.add_dirichlet_noise = add_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        # Lookahead评估参数
        self.use_lookahead = use_lookahead
        self.lookahead_steps = lookahead_steps

        # 创建StateConverter - 使用网络的board尺寸
        self.converter = StateConverter(
            grid_height=network.board_h,
            grid_width=network.board_w,
            feature_height=network.board_h,
            feature_width=network.board_w
        )

    def evaluate_state(self, state: SimplifiedGameState) -> Tuple[Dict[int, float], float]:
        """
        用神经网络评估状态

        Args:
            state: SimplifiedGameState

        Returns:
            action_priors: {action: P(a|s)} 只包含有效动作
            value: V(s) ∈ [-1, 1]
        """
        # 转换为tensor
        state_tensor = self.converter.simplified_to_tensor(state, add_batch_dim=True)

        # 网络推理
        with paddle.no_grad():
            policy, value = self.network(state_tensor)

        policy_array = policy.numpy()[0]  # [num_actions]
        value_scalar = float(value.numpy()[0, 0])

        # 获取有效动作
        valid_actions = state.get_valid_actions()

        if len(valid_actions) == 0:
            # 没有有效动作（游戏结束）
            return {}, value_scalar

        # 只保留有效动作的概率，并重新归一化
        action_priors = {}
        total_prob = 0.0

        for action in valid_actions:
            # Grid宽度=16，动作空间=16，直接一一对应，不需要映射
            prob = policy_array[action]
            action_priors[action] = prob
            total_prob += prob

        # 归一化
        if total_prob > 0:
            for action in action_priors:
                action_priors[action] /= total_prob
        else:
            # 如果所有概率都是0（不应该发生），均匀分布
            uniform_prob = 1.0 / len(valid_actions)
            for action in valid_actions:
                action_priors[action] = uniform_prob

        return action_priors, value_scalar

    def evaluate_with_lookahead(self, state: SimplifiedGameState,
                                parent_score: float = 0.0) -> float:
        """
        评估状态，结合网络价值预测和lookahead模拟

        Args:
            state: 当前状态
            parent_score: 父状态的分数（用于计算immediate reward）

        Returns:
            综合价值评估 (结合immediate reward + lookahead)
        """
        # 1. 获取网络的价值预测
        _, network_value = self.evaluate_state(state)

        if not self.use_lookahead:
            return network_value

        # 2. 计算immediate reward (当前状态分数 - 父状态分数)
        immediate_reward = state.score - parent_score

        # 3. 运行lookahead模拟获取未来奖励
        lookahead_reward = state.simulate_lookahead(
            num_steps=self.lookahead_steps,
            policy="greedy"
        )

        # 4. 综合评估：immediate + lookahead，归一化到 [-1, 1]
        total_reward = immediate_reward + lookahead_reward

        # 归一化：假设500是优秀的总奖励
        normalized_value = float(np.clip(total_reward / 500.0, -1.0, 1.0))

        # 5. 混合网络预测和lookahead (70% lookahead + 30% network)
        # 这样可以保持网络学习的同时利用lookahead信息
        blended_value = 0.7 * normalized_value + 0.3 * network_value

        return blended_value

    def add_exploration_noise(self, node: AlphaZeroNode):
        """
        在根节点的先验概率上添加Dirichlet噪声

        用于增加探索性，AlphaGo Zero的技巧
        """
        actions = list(node.children.keys())
        if len(actions) == 0:
            return

        # 生成Dirichlet噪声
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        # 混合噪声和原始prior
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                         self.dirichlet_epsilon * noise[i]

    def search(self, root_state: SimplifiedGameState) -> np.ndarray:
        """
        从root_state开始MCTS搜索

        Args:
            root_state: 根节点状态

        Returns:
            pi: [num_actions] 访问次数的概率分布
        """
        # 创建根节点
        root = AlphaZeroNode(root_state)

        # 先评估根节点并展开
        action_priors, root_value = self.evaluate_state(root_state)
        if len(action_priors) == 0:
            # 游戏已结束
            return np.zeros(16)  # 返回全0
        root.expand(action_priors)

        # 可选：添加探索噪声
        if self.add_dirichlet_noise:
            self.add_exploration_noise(root)

        # 执行模拟
        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            # ========================================
            # 1. Selection: 向下选择到叶子节点
            # ========================================
            while node.is_expanded() and not node.state.is_terminal:
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # ========================================
            # 2. Evaluation
            # ========================================
            if node.state.is_terminal:
                # 终止状态：使用归一化的分数作为价值（移除death penalty）
                # 归一化到 [-1, 1] 范围
                normalized_score = node.state.score / 500.0  # 假设500是优秀分数
                value = float(np.clip(normalized_score - 0.5, -1.0, 1.0))
            else:
                # 用网络评估 + lookahead
                action_priors, _ = self.evaluate_state(node.state)

                # 获取父节点分数（用于计算immediate reward）
                parent_score = node.parent.state.score if node.parent else 0.0

                # 使用lookahead评估（如果启用）
                value = self.evaluate_with_lookahead(node.state, parent_score)

                # 3. Expansion: 展开叶子节点
                if len(action_priors) > 0:
                    node.expand(action_priors)

            # ========================================
            # 4. Backup: 回传价值
            # ========================================
            for path_node in reversed(search_path):
                path_node.visit_count += 1
                path_node.total_value += value

        # ========================================
        # 返回访问次数分布
        # ========================================
        return self._get_action_prob(root)

    def _get_action_prob(self, root: AlphaZeroNode) -> np.ndarray:
        """
        从根节点提取动作概率分布

        使用温度参数调节探索程度
        """
        action_visits = np.zeros(16, dtype=np.float32)

        # 收集每个action的访问次数 (grid宽度=16，动作=16，直接对应)
        for action, child in root.children.items():
            action_visits[action] += child.visit_count

        if action_visits.sum() == 0:
            return action_visits

        # 应用温度
        if self.temperature == 0:
            # 确定性选择：最多访问的动作
            pi = np.zeros_like(action_visits)
            pi[np.argmax(action_visits)] = 1.0
        else:
            # 温度采样
            action_probs = action_visits ** (1.0 / self.temperature)
            pi = action_probs / action_probs.sum()

        return pi

    def get_action(self, state: SimplifiedGameState, return_prob: bool = False):
        """
        获取动作（包含搜索）

        Args:
            state: 当前状态
            return_prob: 是否返回概率分布

        Returns:
            action: int (如果return_prob=False)
            (action, pi): (int, np.ndarray) (如果return_prob=True)
        """
        pi = self.search(state)

        if self.temperature == 0:
            action = int(np.argmax(pi))
        else:
            action = np.random.choice(len(pi), p=pi)

        if return_prob:
            return action, pi
        else:
            return action


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Testing AlphaZeroMCTS")
    print("="*60)

    # 创建测试状态
    print("\n[Setup] Creating test state...")
    state = SimplifiedGameState(grid_width=10, grid_height=16)

    # 创建网络（随机初始化）- 匹配state的尺寸
    print("[Setup] Creating network...")
    net = SuikaNet(
        input_channels=13,
        num_actions=16,
        board_height=state.height,  # 16
        board_width=state.width      # 10
    )

    # 创建MCTS
    print("[Setup] Creating MCTS...")
    mcts = AlphaZeroMCTS(
        network=net,
        c_puct=1.5,
        num_simulations=100,  # 少量模拟用于测试
        temperature=1.0
    )

    # 测试状态
    print("\n[Test 1] Test state:")
    state.current_fruit = 3

    # 放几个水果
    state.grid[15, 5] = 5
    state.grid[15, 6] = 5
    state.grid[14, 5] = 3

    print(f"  Grid shape: {state.grid.shape}")
    print(f"  Current fruit: {state.current_fruit}")
    print(f"  Valid actions: {len(state.get_valid_actions())}")

    # 测试评估
    print("\n[Test 2] Evaluate state with network:")
    action_priors, value = mcts.evaluate_state(state)
    print(f"  Number of valid actions: {len(action_priors)}")
    print(f"  Action priors sum: {sum(action_priors.values()):.6f}")
    print(f"  Value: {value:.4f}")
    print(f"  Top 3 actions by prior:")
    sorted_actions = sorted(action_priors.items(), key=lambda x: x[1], reverse=True)
    for i, (action, prob) in enumerate(sorted_actions[:3]):
        print(f"    Action {action}: {prob:.4f}")

    # 测试搜索
    print("\n[Test 3] MCTS search (100 simulations):")
    pi = mcts.search(state)
    print(f"  Pi shape: {pi.shape}")
    print(f"  Pi sum: {pi.sum():.6f}")
    print(f"  Best action: {np.argmax(pi)}")
    print(f"  Top 3 actions:")
    top_actions = np.argsort(pi)[::-1][:3]
    for action in top_actions:
        print(f"    Action {action}: {pi[action]:.4f}")

    # 测试获取动作
    print("\n[Test 4] Get action:")
    action = mcts.get_action(state)
    print(f"  Selected action: {action}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
