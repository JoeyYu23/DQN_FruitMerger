#!/usr/bin/env python3
"""完整测试CNN-DQN模型（100局）"""
from CNN_DQN import CNN_DQN_Agent, test_cnn_dqn
from GameInterface import GameInterface

print('='*60)
print('CNN-DQN Final Evaluation (100 episodes)')
print('='*60)

env = GameInterface()
agent = CNN_DQN_Agent(action_dim=16)

# 测试所有checkpoint
checkpoints = [
    ('ep500', 'weights_cnn_dqn/checkpoint_ep500.pth'),
    ('ep1000', 'weights_cnn_dqn/checkpoint_ep1000.pth'),
    ('ep1500', 'weights_cnn_dqn/checkpoint_ep1500.pth'),
    ('ep2000', 'weights_cnn_dqn/checkpoint_ep2000.pth'),
    ('final', 'weights_cnn_dqn/final_model.pth'),
]

results = {}

for name, path in checkpoints:
    print(f'\n{"="*60}')
    print(f'Testing: {name}')
    print('='*60)

    agent.load(path)
    scores = test_cnn_dqn(env, agent, num_episodes=100)
    results[name] = scores

# 总结对比
print('\n' + '='*60)
print('FINAL COMPARISON')
print('='*60)
print(f'{"Checkpoint":<12} {"Mean":<10} {"Std":<10} {"Max":<8} {"Min":<8}')
print('-'*60)

import numpy as np
for name, scores in results.items():
    mean = np.mean(scores)
    std = np.std(scores)
    max_s = max(scores)
    min_s = min(scores)
    print(f'{name:<12} {mean:<10.1f} {std:<10.1f} {max_s:<8} {min_s:<8}')

# 找出最佳checkpoint
best_name = max(results, key=lambda x: np.mean(results[x]))
best_score = np.mean(results[best_name])
print(f'\n🏆 Best Checkpoint: {best_name} with {best_score:.1f} average score')
