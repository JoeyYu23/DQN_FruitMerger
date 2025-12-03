"""
SuikaNet.py - AlphaZero风格的神经网络

网络架构：
    Input: [B, C, H, W] 多通道状态表示
    Body: 3层CNN提取特征
    Head1 (Policy): 输出每个动作的概率 P(a|s)
    Head2 (Value): 输出状态价值 V(s) ∈ [-1, 1]

与DQN的区别：
    DQN: FC网络 → Q(s,a) for each action
    AlphaZero: CNN → (Policy分布, Value标量)
"""

import paddle
from paddle import nn
import paddle.nn.functional as F
import numpy as np


class SuikaNet(nn.Layer):
    """
    AlphaZero风格的网络

    参数:
        input_channels: 输入通道数 (default 13)
            - 0-10: 每个水果等级的binary mask (11通道)
            - 11: 当前即将掉落的水果类型
            - 12: 高度信息（归一化）

        num_actions: 动作空间大小 (default 16, from GameInterface.ACTION_NUM)

        hidden_channels: 卷积层的通道数 (default 64)
    """

    def __init__(self,
                 input_channels=13,
                 num_actions=16,
                 hidden_channels=64,
                 board_height=20,  # GameInterface.FEATURE_MAP_HEIGHT
                 board_width=16):  # GameInterface.FEATURE_MAP_WIDTH
        super(SuikaNet, self).__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.hidden_channels = hidden_channels
        self.board_h = board_height
        self.board_w = board_width

        # ============================================
        # Shared Convolutional Body (特征提取器)
        # ============================================
        # 使用3个3x3卷积层，保持空间分辨率

        self.conv1 = nn.Conv2D(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1  # Same padding
        )
        self.bn1 = nn.BatchNorm2D(hidden_channels)

        self.conv2 = nn.Conv2D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2D(hidden_channels)

        self.conv3 = nn.Conv2D(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2D(hidden_channels)

        # ============================================
        # Policy Head (输出动作概率分布)
        # ============================================
        # 策略头：CNN降维 → Flatten → FC

        self.policy_conv = nn.Conv2D(
            in_channels=hidden_channels,
            out_channels=2,  # 降维到2通道
            kernel_size=1,   # 1x1卷积
            stride=1
        )
        self.policy_bn = nn.BatchNorm2D(2)

        # 全连接层：(2 × H × W) → num_actions
        policy_fc_input_size = 2 * board_height * board_width
        self.policy_fc = nn.Linear(policy_fc_input_size, num_actions)

        # ============================================
        # Value Head (输出状态价值)
        # ============================================
        # 价值头：CNN降维 → Flatten → FC → FC → tanh

        self.value_conv = nn.Conv2D(
            in_channels=hidden_channels,
            out_channels=1,  # 降维到1通道
            kernel_size=1,
            stride=1
        )
        self.value_bn = nn.BatchNorm2D(1)

        # 全连接层：(1 × H × W) → 64 → 1
        value_fc1_input_size = 1 * board_height * board_width
        self.value_fc1 = nn.Linear(value_fc1_input_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        print(f"[SuikaNet] Initialized:")
        print(f"  Input: [{input_channels}, {board_height}, {board_width}]")
        print(f"  Actions: {num_actions}")
        print(f"  Hidden channels: {hidden_channels}")
        print(f"  Policy FC input: {policy_fc_input_size}")
        print(f"  Value FC input: {value_fc1_input_size}")

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, C, H, W] 状态tensor
                B = batch size
                C = input_channels (13)
                H = board_height (20)
                W = board_width (16)

        Returns:
            policy: [B, num_actions] - softmax概率分布，和为1
            value: [B, 1] - tanh输出，范围[-1, 1]
        """
        # ============================================
        # Shared Body: 特征提取
        # ============================================
        h = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 20, 16]
        h = F.relu(self.bn2(self.conv2(h)))  # [B, 64, 20, 16]
        h = F.relu(self.bn3(self.conv3(h)))  # [B, 64, 20, 16]

        # ============================================
        # Policy Head: 动作概率
        # ============================================
        p = F.relu(self.policy_bn(self.policy_conv(h)))  # [B, 2, 20, 16]
        p = paddle.flatten(p, start_axis=1)               # [B, 2*20*16=640]
        logits = self.policy_fc(p)                        # [B, 16]
        policy = F.softmax(logits, axis=-1)               # [B, 16], 和为1

        # ============================================
        # Value Head: 状态价值
        # ============================================
        v = F.relu(self.value_bn(self.value_conv(h)))  # [B, 1, 20, 16]
        v = paddle.flatten(v, start_axis=1)             # [B, 1*20*16=320]
        v = F.relu(self.value_fc1(v))                   # [B, 64]
        value = paddle.tanh(self.value_fc2(v))          # [B, 1], 范围[-1,1]

        return policy, value

    def predict(self, state_tensor):
        """
        单个状态的推理（不计算梯度）

        Args:
            state_tensor: [C, H, W] 单个状态 (无batch维度)

        Returns:
            policy_array: [num_actions] numpy数组
            value_scalar: float
        """
        self.eval()  # 设置为评估模式

        # 添加batch维度
        if len(state_tensor.shape) == 3:
            state_tensor = paddle.unsqueeze(state_tensor, axis=0)  # [1, C, H, W]

        with paddle.no_grad():
            policy, value = self.forward(state_tensor)

        # 转numpy
        policy_array = policy.numpy()[0]  # [num_actions]
        value_scalar = float(value.numpy()[0, 0])

        return policy_array, value_scalar


# ============================================
# 辅助函数：状态表示转换
# ============================================

def state_to_tensor(grid, current_fruit, max_height=None,
                    grid_height=20, grid_width=16):
    """
    将游戏状态转换为网络输入tensor

    Args:
        grid: [H, W] numpy数组，值为0-10 (水果类型)
        current_fruit: int, 当前即将掉落的水果类型 (1-10)
        max_height: float, 当前最高堆叠高度 (可选)

    Returns:
        tensor: [1, 13, H, W] paddle.Tensor
    """
    H, W = grid.shape
    channels = []

    # ========================================
    # Channels 0-10: 每个水果等级的binary mask
    # ========================================
    # 为什么这样做？
    # - 相比直接用grid数值，one-hot编码更明确
    # - 每个通道关注特定水果类型的位置
    # - 避免网络学习"水果5比水果3大"这种数值关系（应该学空间关系）

    for fruit_type in range(11):  # 0 (empty) 到 10 (最大水果)
        mask = (grid == fruit_type).astype(np.float32)
        channels.append(mask)

    # ========================================
    # Channel 11: 当前水果类型（全图广播）
    # ========================================
    # 为什么需要这个？
    # - MCTS搜索时，相同的grid配置 + 不同的current_fruit = 不同策略
    # - 例如：当前是小水果 → 可以冒险放高处
    #        当前是大水果 → 要更小心

    current_fruit_normalized = current_fruit / 10.0  # 归一化到[0, 1]
    current_fruit_channel = np.full((H, W), current_fruit_normalized,
                                    dtype=np.float32)
    channels.append(current_fruit_channel)

    # ========================================
    # Channel 12: 高度信息（可选）
    # ========================================
    # 为什么有用？
    # - 帮助网络快速判断"危险程度"
    # - 高度接近顶部 → 更保守的策略

    if max_height is not None:
        height_normalized = max_height / H
    else:
        # 自动计算：找每列最高的非空格子
        height_normalized = 0.0
        for col in range(W):
            for row in range(H):
                if grid[row, col] != 0:
                    height_normalized = max(height_normalized, row / H)
                    break

    height_channel = np.full((H, W), height_normalized, dtype=np.float32)
    channels.append(height_channel)

    # ========================================
    # Stack所有通道
    # ========================================
    state_array = np.stack(channels, axis=0)  # [13, H, W]

    # 转paddle tensor并添加batch维度
    tensor = paddle.to_tensor(state_array, dtype='float32')
    tensor = paddle.unsqueeze(tensor, axis=0)  # [1, 13, H, W]

    return tensor


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Testing SuikaNet")
    print("="*60)

    # 创建网络
    net = SuikaNet(
        input_channels=13,
        num_actions=16,
        hidden_channels=64,
        board_height=20,
        board_width=16
    )

    # 创建假的输入
    print("\n[Test 1] Random input:")
    batch_size = 4
    x = paddle.randn([batch_size, 13, 20, 16])

    # 前向传播
    policy, value = net(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Policy shape: {policy.shape}")  # [4, 16]
    print(f"  Value shape: {value.shape}")    # [4, 1]
    print(f"  Policy sum: {policy.sum(axis=1)}")  # 应该都是1.0
    print(f"  Value range: [{value.min():.3f}, {value.max():.3f}]")  # 应该在[-1,1]

    # 测试单个状态推理
    print("\n[Test 2] Single state prediction:")
    test_grid = np.zeros((20, 16), dtype=np.int8)
    test_grid[19, 8] = 5  # 底部中间放一个水果
    test_grid[19, 7] = 5  # 旁边再放一个

    state_tensor = state_to_tensor(test_grid, current_fruit=3)
    print(f"  State tensor shape: {state_tensor.shape}")

    policy_array, value_scalar = net.predict(state_tensor)
    print(f"  Policy: {policy_array}")
    print(f"  Policy sum: {policy_array.sum():.6f}")
    print(f"  Best action: {np.argmax(policy_array)}")
    print(f"  Value: {value_scalar:.4f}")

    # 参数量统计
    print("\n[Test 3] Model statistics:")
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if not p.stop_gradient)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
