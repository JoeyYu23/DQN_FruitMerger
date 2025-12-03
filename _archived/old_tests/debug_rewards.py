#!/usr/bin/env python3
"""
调试每个位置的reward
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent
import numpy as np

print("="*70)
print("🔍 调试Reward计算")
print("="*70)

env = GameInterface()
env.reset(seed=888)

# 玩几步
for i in range(5):
    action = np.random.randint(0, 16)
    env.next(action)

print(f"\n📊 当前状态:")
print(f"  场上水果数: {len(env.game.fruits)}")
print(f"  手里水果类型: {env.game.current_fruit_type}")

print(f"\n场上水果详情:")
for i, (ball, fruit) in enumerate(zip(env.game.balls, env.game.fruits)):
    x = ball.body.position.x
    action_pos = int(x / (env.game.width / 16))
    print(f"  水果{i}: type={fruit.type}, action_pos≈{action_pos}")

# 手动计算每个action的reward
agent = RealPhysicsMCTSAgent(num_simulations=10)

original_state = agent.mcts._save_state(env)

print(f"\n📈 每个位置的Reward计算:")
print(f"{'Action':>6} | {'Reward1':>8} | {'MaxReward2':>10} | {'Total':>8}")
print("-"*50)

for action1 in range(16):
    # 恢复初始状态
    agent.mcts._restore_state(env, original_state)

    # 记录初始状态
    score_before = env.game.score
    fruits_before = agent.mcts._get_fruits_info(env)

    # 执行第一步
    agent.mcts._apply_action(env, action1)

    # 计算第一步reward
    reward1 = agent.mcts._calculate_reward(env, score_before, fruits_before)

    # 记录第一步后状态
    state_after_step1 = agent.mcts._save_state(env)
    score_after1 = env.game.score
    fruits_after1 = agent.mcts._get_fruits_info(env)

    # 计算第二步所有可能的reward
    max_reward2 = float('-inf')
    best_action2 = -1

    for action2 in range(16):
        agent.mcts._restore_state(env, state_after_step1)
        agent.mcts._apply_action(env, action2)
        reward2 = agent.mcts._calculate_reward(env, score_after1, fruits_after1)

        if reward2 > max_reward2:
            max_reward2 = reward2
            best_action2 = action2

    total_reward = reward1 + max_reward2

    # 检查是否能merge
    can_merge = False
    for ball, fruit in zip(env.game.balls, env.game.fruits):
        if fruit.type == env.game.current_fruit_type:
            action_pos = int(ball.body.position.x / (env.game.width / 16))
            if abs(action1 - action_pos) <= 1:
                can_merge = True
                break

    marker = "⭐" if can_merge else "  "
    print(f"{action1:6d} | {reward1:8.2f} | {max_reward2:10.2f} | {total_reward:8.2f} {marker}")

# 运行MCTS
print(f"\n🎮 MCTS选择:")
agent.mcts._restore_state(env, original_state)
best_action = agent.mcts.search(env, 10)
print(f"  最佳动作: {best_action}")

print("="*70)
