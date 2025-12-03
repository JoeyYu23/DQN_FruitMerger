#!/usr/bin/env python3
"""
测试MCTS配置和rollout策略
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent, RealPhysicsConfig
import numpy as np

print("="*70)
print("🔍 检查MCTS配置")
print("="*70)

# 1. 检查配置
print("\n📊 当前配置:")
print(f"  ROLLOUT_STEPS: {RealPhysicsConfig.ROLLOUT_STEPS}")
print(f"  MERGE_REWARD: {RealPhysicsConfig.MERGE_REWARD}")
print(f"  HEIGHT_PENALTY: {RealPhysicsConfig.HEIGHT_PENALTY}")
print(f"  POSITION_REWARD: {RealPhysicsConfig.POSITION_REWARD}")
print(f"  比例 MERGE/HEIGHT: {RealPhysicsConfig.MERGE_REWARD/RealPhysicsConfig.HEIGHT_PENALTY:.1f}:1")

# 2. 测试rollout策略
print("\n🎯 测试Rollout策略:")
env = GameInterface()
env.reset(seed=888)

# 第一步随机
env.next(8)

# 模拟一个可以merge的场景
print(f"\n当前场上水果:")
for i, (ball, fruit) in enumerate(zip(env.game.balls, env.game.fruits)):
    print(f"  水果{i}: type={fruit.type}, x={ball.body.position.x:.1f}")

print(f"\n手里的水果: type={env.game.current_fruit_type}")

# 创建agent并测试rollout策略
agent = RealPhysicsMCTSAgent(num_simulations=10)

# 测试rollout策略100次，看权重分布
print(f"\n测试rollout策略（100次采样）:")
actions = []
for _ in range(100):
    action = agent.mcts._rollout_policy(env)
    actions.append(action)

# 统计
from collections import Counter
action_counts = Counter(actions)
print(f"动作分布:")
for action in sorted(action_counts.keys()):
    count = action_counts[action]
    bar = "█" * (count // 2)
    print(f"  Action {action:2d}: {count:3d} {bar}")

# 3. 测试一步MCTS
print(f"\n🎮 运行一步MCTS (10 sims):")
action = agent.predict(env)[0]
print(f"  选择的动作: {action}")

print("\n" + "="*70)
