"""
快速测试完整AlphaZero流程
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 使用CPU

from alphazero_config import AlphaZeroConfig as cfg

print("="*60)
print("Testing AlphaZero Pipeline")
print("="*60)
print(f"Grid Size: {cfg.GRID_HEIGHT} x {cfg.GRID_WIDTH}")
print("="*60)

# 1. 测试网络
print("\n[1/4] Testing SuikaNet...")
from SuikaNet import SuikaNet
import paddle
net = SuikaNet(
    input_channels=cfg.INPUT_CHANNELS,
    num_actions=cfg.NUM_ACTIONS,
    board_height=cfg.GRID_HEIGHT,
    board_width=cfg.GRID_WIDTH,
    hidden_channels=cfg.HIDDEN_CHANNELS
)
test_input = paddle.randn([2, cfg.INPUT_CHANNELS, cfg.GRID_HEIGHT, cfg.GRID_WIDTH])
policy, value = net(test_input)
assert policy.shape == [2, cfg.NUM_ACTIONS]
assert value.shape == [2, 1]
print(f"✓ SuikaNet works! Output shapes: {policy.shape}, {value.shape}")

# 2. 测试状态转换
print("\n[2/4] Testing StateConverter...")
from StateConverter import StateConverter
from GameInterface import GameInterface
converter = StateConverter(
    grid_height=cfg.GRID_HEIGHT,
    grid_width=cfg.GRID_WIDTH,
    feature_height=cfg.GRID_HEIGHT,
    feature_width=cfg.GRID_WIDTH
)
game = GameInterface()
game.reset(seed=42)
tensor = converter.game_to_tensor(game)
expected_shape = [1, cfg.INPUT_CHANNELS, cfg.GRID_HEIGHT, cfg.GRID_WIDTH]
assert list(tensor.shape) == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"
print(f"✓ StateConverter works! Tensor shape: {tensor.shape}")

# 3. 测试MCTS
print("\n[3/4] Testing AlphaZeroMCTS...")
from AlphaZeroMCTS import AlphaZeroMCTS
mcts = AlphaZeroMCTS(
    network=net,
    num_simulations=50,
    c_puct=cfg.C_PUCT,
    temperature=cfg.TEMPERATURE
)
simplified = converter.game_to_simplified(game)
pi = mcts.search(simplified)
assert pi.shape == (cfg.NUM_ACTIONS,)
assert abs(pi.sum() - 1.0) < 0.01
print(f"✓ AlphaZeroMCTS works! Pi shape: {pi.shape}, sum: {pi.sum():.6f}")

# 4. 测试Self-Play
print("\n[4/4] Testing SelfPlay...")
from SelfPlay import SelfPlayCollector
collector = SelfPlayCollector(
    network=net,
    num_simulations=20,
    temperature=cfg.TEMPERATURE
)
data = collector.play_one_episode(seed=123, verbose=False)
assert len(data) > 0
state, pi_sample, z = data[0]
expected_state_shape = (cfg.INPUT_CHANNELS, cfg.GRID_HEIGHT, cfg.GRID_WIDTH)
assert state.shape == expected_state_shape, f"Expected {expected_state_shape}, got {state.shape}"
assert pi_sample.shape == (cfg.NUM_ACTIONS,)
assert isinstance(z, float)
print(f"✓ SelfPlay works! Collected {len(data)} samples, final score z={z:.4f}")

print("\n" + "="*60)
print("All Pipeline Tests Passed! ✓")
print("="*60)
print("\nYou can now:")
print("1. Quick test: python TrainAlphaZero.py --iterations 2 --games 5 --simulations 50")
print("2. Full train: ./train_cloud.sh")
print("3. Evaluate: python CompareAgents.py --num-games 10")
print("="*60)
