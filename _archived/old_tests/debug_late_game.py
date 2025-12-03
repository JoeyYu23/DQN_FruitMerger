#!/usr/bin/env python3
"""
调试后期游戏为什么不扔水果
"""

from GameInterface import GameInterface
from mcts.MCTS_real_physics import RealPhysicsMCTSAgent
import numpy as np

print("="*70)
print("🔍 调试后期游戏状态")
print("="*70)

env = GameInterface()
agent = RealPhysicsMCTSAgent(num_simulations=10)

env.reset(seed=888)

# 快进到50步
print("\n快进到Step 50...")
env.next(8)  # 第一步随机
for i in range(49):
    action = agent.predict(env)[0]
    feature, reward, alive = env.next(action)
    if not alive:
        print(f"  游戏在Step {i+2}结束")
        break

print(f"\n📊 Step 50状态:")
print(f"  Score: {env.game.score}")
print(f"  Fruits: {len(env.game.fruits)}")
print(f"  Alive: {env.game.alive}")
print(f"  Current fruit type: {env.game.current_fruit_type}")

# 执行几步，详细观察
for step in range(51, 56):
    print(f"\n{'='*70}")
    print(f"Step {step}:")

    # 记录执行前状态
    score_before = env.game.score
    fruits_before = len(env.game.fruits)
    current_type = env.game.current_fruit_type

    print(f"  执行前: Score={score_before}, Fruits={fruits_before}, Type={current_type}")

    # 获取action
    action = agent.predict(env)[0]
    print(f"  MCTS选择: Action {action}")

    # 检查这个位置是否能merge
    can_merge = False
    for ball, fruit in zip(env.game.balls, env.game.fruits):
        if fruit.type == current_type:
            action_pos = int(ball.body.position.x / (env.game.width / 16))
            if abs(action - action_pos) <= 1:
                can_merge = True
                print(f"    → 可以merge! (场上有type={current_type}在action {action_pos})")
                break

    if not can_merge:
        print(f"    → 不能merge (场上没有type={current_type})")

    # 执行
    feature, reward, alive = env.next(action)

    # 记录执行后状态
    score_after = env.game.score
    fruits_after = len(env.game.fruits)

    print(f"  执行后: Score={score_after}, Fruits={fruits_after}")
    print(f"  变化: ΔScore={score_after-score_before}, ΔFruits={fruits_after-fruits_before}")

    if not alive:
        print(f"  ⚠️ 游戏结束!")
        break

    # 检查是否有问题
    if score_after == score_before and fruits_after == fruits_before:
        print(f"  ⚠️ 警告: 得分和水果数都没变!")
        print(f"  当前场上水果:")
        for i, (ball, fruit) in enumerate(zip(env.game.balls, env.game.fruits)):
            x = ball.body.position.x
            y = ball.body.position.y
            print(f"    水果{i}: type={fruit.type}, x={x:.1f}, y={y:.1f}")

print("\n" + "="*70)
