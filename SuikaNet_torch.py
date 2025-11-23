"""
SuikaNet_torch.py - AlphaZero风格的神经网络 (PyTorch版本)

网络架构：
    Input: [B, C, H, W] 多通道状态表示
    Body: 3层CNN提取特征
    Head1 (Policy): 输出每个动作的概率 P(a|s)
    Head2 (Value): 输出状态价值 V(s) ∈ [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SuikaNet(nn.Module):
    """
    AlphaZero风格的网络 (PyTorch版本)

    参数:
        input_channels: 输入通道数 (default 13)
        num_actions: 动作空间大小 (default 16)
        hidden_channels: 卷积层的通道数 (default 64)
        board_height: 棋盘高度 (default 20)
        board_width: 棋盘宽度 (default 16)
    """

    def __init__(self,
                 input_channels=13,
                 num_actions=16,
                 hidden_channels=64,
                 board_height=20,
                 board_width=16):
        super(SuikaNet, self).__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions
        self.hidden_channels = hidden_channels
        self.board_h = board_height
        self.board_w = board_width

        # ============================================
        # Shared Convolutional Body (特征提取器)
        # ============================================
        self.conv1 = nn.Conv2d(input_channels, hidden_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels)

        # ============================================
        # Policy Head (输出动作概率分布)
        # ============================================
        self.policy_conv = nn.Conv2d(hidden_channels, 2,
                                     kernel_size=1, stride=1)
        self.policy_bn = nn.BatchNorm2d(2)

        policy_fc_input_size = 2 * board_height * board_width
        self.policy_fc = nn.Linear(policy_fc_input_size, num_actions)

        # ============================================
        # Value Head (输出状态价值)
        # ============================================
        self.value_conv = nn.Conv2d(hidden_channels, 1,
                                    kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)

        value_fc1_input_size = 1 * board_height * board_width
        self.value_fc1 = nn.Linear(value_fc1_input_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

        print(f"[SuikaNet-PyTorch] Initialized:")
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

        Returns:
            policy: [B, num_actions] - softmax概率分布
            value: [B, 1] - tanh输出，范围[-1, 1]
        """
        # Shared Body
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.view(p.size(0), -1)  # Flatten
        logits = self.policy_fc(p)
        policy = F.softmax(logits, dim=-1)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy, value

    def predict(self, state_tensor):
        """
        单个状态的推理（不计算梯度）

        Args:
            state_tensor: [C, H, W] 或 [1, C, H, W]

        Returns:
            policy_array: [num_actions] numpy数组
            value_scalar: float
        """
        self.eval()

        # 添加batch维度
        if len(state_tensor.shape) == 3:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            policy, value = self.forward(state_tensor)

        # 转numpy
        policy_array = policy.cpu().numpy()[0]
        value_scalar = float(value.cpu().numpy()[0, 0])

        return policy_array, value_scalar


# ============================================
# 辅助函数：状态表示转换
# ============================================

def state_to_tensor(grid, current_fruit, max_height=None,
                    grid_height=20, grid_width=16, device='cpu'):
    """
    将游戏状态转换为网络输入tensor (PyTorch版本)

    Args:
        grid: [H, W] numpy数组
        current_fruit: int 当前水果类型
        max_height: float 最高堆叠高度
        device: 'cpu' 或 'cuda'

    Returns:
        tensor: [1, 13, H, W] torch.Tensor
    """
    H, W = grid.shape
    channels = []

    # Channels 0-10: 水果类型masks
    for fruit_type in range(11):
        mask = (grid == fruit_type).astype(np.float32)
        channels.append(mask)

    # Channel 11: 当前水果类型
    current_fruit_normalized = current_fruit / 10.0
    current_fruit_channel = np.full((H, W), current_fruit_normalized,
                                    dtype=np.float32)
    channels.append(current_fruit_channel)

    # Channel 12: 高度信息
    if max_height is not None:
        height_normalized = max_height / H
    else:
        height_normalized = 0.0
        for col in range(W):
            for row in range(H):
                if grid[row, col] != 0:
                    height_normalized = max(height_normalized, row / H)
                    break

    height_channel = np.full((H, W), height_normalized, dtype=np.float32)
    channels.append(height_channel)

    # Stack并转tensor
    state_array = np.stack(channels, axis=0)  # [13, H, W]
    tensor = torch.from_numpy(state_array).float().unsqueeze(0)  # [1, 13, H, W]

    return tensor.to(device)


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Testing SuikaNet (PyTorch)")
    print("="*60)

    # 创建网络
    net = SuikaNet(
        input_channels=13,
        num_actions=16,
        hidden_channels=64,
        board_height=20,
        board_width=16
    )

    # 测试1: Random input
    print("\n[Test 1] Random input:")
    batch_size = 4
    x = torch.randn(batch_size, 13, 20, 16)

    policy, value = net(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Policy sum: {policy.sum(dim=1)}")
    print(f"  Value range: [{value.min():.3f}, {value.max():.3f}]")

    # 测试2: Single state prediction
    print("\n[Test 2] Single state prediction:")
    test_grid = np.zeros((20, 16), dtype=np.int8)
    test_grid[19, 8] = 5
    test_grid[19, 7] = 5

    state_tensor = state_to_tensor(test_grid, current_fruit=3)
    print(f"  State tensor shape: {state_tensor.shape}")

    policy_array, value_scalar = net.predict(state_tensor)
    print(f"  Policy sum: {policy_array.sum():.6f}")
    print(f"  Best action: {np.argmax(policy_array)}")
    print(f"  Value: {value_scalar:.4f}")

    # 测试3: Model statistics
    print("\n[Test 3] Model statistics:")
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
