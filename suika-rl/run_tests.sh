#!/bin/bash
# Run all algorithm tests

cd "$(dirname "$0")"

echo "Testing DQN..."
python training/test_dqn_performance.py

echo "Testing AlphaZero..."
# Add test command here

echo "All tests completed!"
