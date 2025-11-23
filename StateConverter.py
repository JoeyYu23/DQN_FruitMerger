"""
StateConverter.py - 状态转换工具

负责在不同状态表示之间转换：
- GameInterface ↔ SimplifiedGameState (用于MCTS)
- SimplifiedGameState → Tensor (用于神经网络)
"""

import numpy as np
import paddle
from typing import Tuple
from GameInterface import GameInterface
from mcts.MCTS import SimplifiedGameState, MCTSConfig


class StateConverter:
    """状态转换器"""

    def __init__(self,
                 grid_height=MCTSConfig.GRID_HEIGHT,
                 grid_width=MCTSConfig.GRID_WIDTH,
                 feature_height=20,  # GameInterface.FEATURE_MAP_HEIGHT
                 feature_width=16):  # GameInterface.FEATURE_MAP_WIDTH
        self.grid_h = grid_height
        self.grid_w = grid_width
        self.feature_h = feature_height
        self.feature_w = feature_width

    def game_to_simplified(self, game: GameInterface) -> SimplifiedGameState:
        """
        将GameInterface转换为SimplifiedGameState

        策略：
        1. 从game.game获取features (压缩后的grid表示)
        2. 重建grid (反向推导水果类型)
        3. 复制score, current_fruit等信息

        Args:
            game: GameInterface实例

        Returns:
            SimplifiedGameState实例
        """
        state = SimplifiedGameState(self.grid_w, self.grid_h)

        # 获取压缩的特征 (已经是grid形式)
        features = game.game.get_features(self.grid_w, self.grid_h)
        # features shape: [H, W, 2]
        #   channel 0: 比当前水果小的差值
        #   channel 1: 比当前水果大的差值

        current_fruit = game.game.current_fruit_type

        # 重建grid
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                smaller_diff = features[i, j, 0]  # 比current小
                larger_diff = features[i, j, 1]   # 比current大

                if smaller_diff > 0:
                    # 这个位置的水果比current小
                    fruit_type = int(current_fruit - smaller_diff)
                    state.grid[i, j] = max(1, min(10, fruit_type))
                elif larger_diff > 0:
                    # 这个位置的水果比current大
                    fruit_type = int(current_fruit + larger_diff)
                    state.grid[i, j] = max(1, min(10, fruit_type))
                else:
                    # 空位
                    state.grid[i, j] = 0

        # 复制其他信息
        state.current_fruit = current_fruit
        state.score = game.game.score
        state.is_terminal = not game.game.alive

        # 计算最高高度
        state.max_height = 0
        for col in range(self.grid_w):
            for row in range(self.grid_h):
                if state.grid[row, col] != 0:
                    state.max_height = max(state.max_height, row)
                    break

        return state

    def simplified_to_tensor(self,
                            state: SimplifiedGameState,
                            add_batch_dim=True) -> paddle.Tensor:
        """
        将SimplifiedGameState转换为网络输入tensor

        Args:
            state: SimplifiedGameState实例
            add_batch_dim: 是否添加batch维度

        Returns:
            tensor: [1, 13, H, W] 或 [13, H, W]
        """
        H, W = state.grid.shape
        channels = []

        # Channels 0-10: 每个水果等级的binary mask
        for fruit_type in range(11):
            mask = (state.grid == fruit_type).astype(np.float32)
            channels.append(mask)

        # Channel 11: 当前水果类型（归一化）
        current_fruit_normalized = state.current_fruit / 10.0
        current_fruit_channel = np.full((H, W), current_fruit_normalized,
                                       dtype=np.float32)
        channels.append(current_fruit_channel)

        # Channel 12: 高度信息（归一化）
        height_normalized = state.max_height / H if state.max_height > 0 else 0.0
        height_channel = np.full((H, W), height_normalized, dtype=np.float32)
        channels.append(height_channel)

        # Stack channels
        state_array = np.stack(channels, axis=0)  # [13, H, W]

        # 转tensor
        tensor = paddle.to_tensor(state_array, dtype='float32')

        if add_batch_dim:
            tensor = paddle.unsqueeze(tensor, axis=0)  # [1, 13, H, W]

        return tensor

    def game_to_tensor(self, game: GameInterface, add_batch_dim=True) -> paddle.Tensor:
        """
        直接从GameInterface转到tensor（组合操作）

        Args:
            game: GameInterface实例
            add_batch_dim: 是否添加batch维度

        Returns:
            tensor: [1, 13, H, W] 或 [13, H, W]
        """
        simplified = self.game_to_simplified(game)
        return self.simplified_to_tensor(simplified, add_batch_dim)

    def decode_action(self, grid_action: int, num_game_actions=16) -> int:
        """
        将grid列索引转换为GameInterface动作

        Args:
            grid_action: 0 到 grid_width-1
            num_game_actions: GameInterface.ACTION_NUM (16)

        Returns:
            game_action: 0 到 num_game_actions-1
        """
        # 简单线性映射
        game_action = int(grid_action * num_game_actions / self.grid_w)
        return min(game_action, num_game_actions - 1)

    def encode_action(self, game_action: int, num_game_actions=16) -> int:
        """
        将GameInterface动作转换为grid列索引

        Args:
            game_action: 0 到 num_game_actions-1
            num_game_actions: GameInterface.ACTION_NUM (16)

        Returns:
            grid_action: 0 到 grid_width-1
        """
        grid_action = int(game_action * self.grid_w / num_game_actions)
        return min(grid_action, self.grid_w - 1)


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Testing StateConverter")
    print("="*60)

    converter = StateConverter()

    # Test 1: GameInterface → SimplifiedGameState
    print("\n[Test 1] GameInterface → SimplifiedGameState:")
    game = GameInterface()
    game.reset(seed=42)

    # 玩几步
    for _ in range(5):
        action = np.random.randint(0, 16)
        _, _, alive = game.next(action)
        if not alive:
            break

    simplified = converter.game_to_simplified(game)
    print(f"  Grid shape: {simplified.grid.shape}")
    print(f"  Current fruit: {simplified.current_fruit}")
    print(f"  Score: {simplified.score}")
    print(f"  Max height: {simplified.max_height}")
    print(f"  Grid sample (top-left 5x5):")
    print(simplified.grid[:5, :5])

    # Test 2: SimplifiedGameState → Tensor
    print("\n[Test 2] SimplifiedGameState → Tensor:")
    tensor = converter.simplified_to_tensor(simplified)
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor dtype: {tensor.dtype}")
    print(f"  Channel 11 (current fruit) value: {tensor[0, 11, 0, 0]:.4f}")
    print(f"  Channel 12 (height) value: {tensor[0, 12, 0, 0]:.4f}")

    # Test 3: GameInterface → Tensor (直接)
    print("\n[Test 3] GameInterface → Tensor (direct):")
    tensor_direct = converter.game_to_tensor(game)
    print(f"  Tensor shape: {tensor_direct.shape}")
    print(f"  Values match: {paddle.allclose(tensor, tensor_direct)}")

    # Test 4: Action conversion
    print("\n[Test 4] Action conversion:")
    for game_act in [0, 5, 10, 15]:
        grid_act = converter.encode_action(game_act)
        game_act_back = converter.decode_action(grid_act)
        print(f"  Game action {game_act:2d} → Grid action {grid_act:2d} → Game action {game_act_back:2d}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
