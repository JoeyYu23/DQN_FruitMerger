#!/usr/bin/env python3
"""Quick test script for CNN DQN training"""
import sys
from CNN_DQN import CNN_DQN_Agent, ReplayMemory, train_cnn_dqn
from GameInterface import GameInterface

# Force unbuffered output
sys.stdout.flush()

print('Starting CNN DQN training test...', flush=True)
print(flush=True)

# Create environment and agent
env = GameInterface()
agent = CNN_DQN_Agent(action_dim=16)
memory = ReplayMemory()

# Run short training (100 episodes)
scores, losses = train_cnn_dqn(env, agent, memory,
                               num_episodes=100,
                               save_interval=100,
                               model_dir='weights_cnn_dqn')

print('\nTraining test completed!', flush=True)
print(f'Average score (last 50): {sum(scores[-50:])/len(scores[-50:]):.1f}', flush=True)
