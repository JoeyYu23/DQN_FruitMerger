#!/bin/bash
# 实时监控CNN-DQN训练
echo "CNN-DQN训练监控"
echo "==============="
echo ""
tail -f cnn_dqn_training.log | grep --line-buffered "Episode"
