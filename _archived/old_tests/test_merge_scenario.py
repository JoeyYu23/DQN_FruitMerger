#!/usr/bin/env python3
"""
测试merge场景
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent
import numpy as np

print("="*70)
print("🧪 测试Merge场景")
print("="*70)

env = GameInterface()
env.reset(seed=888)

# 玩几步，确保场上有水果
print("\n执行几步，让场上有水果...")
for i in range(5):
    action = np.random.randint(0, 16)
    env.next(action)
    print(f"  Step {i+1}: 场上 {len(env.game.fruits)} 个水果, Score={env.game.score}")

# 现在查看场上水果和当前水果
print(f"\n📊 当前状态:")
print(f"  场上水果数: {len(env.game.fruits)}")
print(f"  手里水果类型: {env.game.current_fruit_type}")

print(f"\n场上水果详情:")
for i, (ball, fruit) in enumerate(zip(env.game.balls, env.game.fruits)):
    x = ball.body.position.x
    y = ball.body.position.y
    action_pos = int(x / (env.game.width / 16))
    print(f"  水果{i}: type={fruit.type}, x={x:.1f}, y={y:.1f}, action_pos≈{action_pos}")

# 测试rollout策略
print(f"\n🎯 测试Rollout策略（当前水果type={env.game.current_fruit_type}）:")

# 创建agent
agent = RealPhysicsMCTSAgent(num_simulations=10)

# 采样100次看分布
actions = []
for _ in range(100):
    action = agent.mcts._rollout_policy(env)
    actions.append(action)

# 统计
from collections import Counter
action_counts = Counter(actions)

print(f"\n动作分布（100次采样）:")
for action in sorted(action_counts.keys()):
    count = action_counts[action]
    bar = "█" * max(1, count // 2)
    # 检查这个位置是否有相同类型的水果
    has_match = False
    for ball, fruit in zip(env.game.balls, env.game.fruits):
        if fruit.type == env.game.current_fruit_type:
            action_pos = int(ball.body.position.x / (env.game.width / 16))
            if abs(action - action_pos) <= 1:
                has_match = True
                break

    marker = " ⭐MERGE" if has_match else ""
    print(f"  Action {action:2d}: {count:3d} {bar}{marker}")

# 执行MCTS
print(f"\n🎮 运行MCTS (50 sims):")
action = agent.predict(env)[0]
print(f"  MCTS选择的动作: {action}")

# 检查是否能merge
can_merge = False
for ball, fruit in zip(env.game.balls, env.game.fruits):
    if fruit.type == env.game.current_fruit_type:
        action_pos = int(ball.body.position.x / (env.game.width / 16))
        if abs(action - action_pos) <= 1:
            can_merge = True
            print(f"  ✅ 可以merge! (目标水果在action {action_pos}附近)")
            break

if not can_merge:
    print(f"  ⚠️  无法merge (场上没有type={env.game.current_fruit_type}的水果)")

print("="*70)
