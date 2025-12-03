#!/bin/bash
# Project Reorganization Script

set -e  # Exit on error

PROJECT_ROOT="/Users/ycy/Downloads/DQN_FruitMerger"
NEW_ROOT="$PROJECT_ROOT/suika-rl"

echo "========================================"
echo "  Suika-RL Project Reorganization"
echo "========================================"

# Step 1: Verify directories exist
echo "[1/5] Verifying directory structure..."
cd "$PROJECT_ROOT"

# Step 2: Create Python package structure
echo "[2/5] Creating __init__.py files..."

cat > "$NEW_ROOT/__init__.py" << 'EOF'
"""Suika-RL: Reinforcement Learning for Suika Game"""
__version__ = "1.0.0"
EOF

cat > "$NEW_ROOT/env/__init__.py" << 'EOF'
from .Game import GameCore
from .GameInterface import GameInterface
from .PRNG import PRNG

__all__ = ['GameCore', 'GameInterface', 'PRNG']
EOF

cat > "$NEW_ROOT/models/__init__.py" << 'EOF'
from .SuikaNet import SuikaNet
from .StateConverter import StateConverter

__all__ = ['SuikaNet', 'StateConverter']
EOF

cat > "$NEW_ROOT/algorithms/__init__.py" << 'EOF'
"""All RL algorithms"""
EOF

cat > "$NEW_ROOT/algorithms/dqn/__init__.py" << 'EOF'
from .DQN import Agent, RandomAgent, build_model

__all__ = ['Agent', 'RandomAgent', 'build_model']
EOF

cat > "$NEW_ROOT/algorithms/alphazero/__init__.py" << 'EOF'
from .AlphaZeroMCTS import AlphaZeroMCTS, AlphaZeroNode
from .SelfPlay import play_one_episode

__all__ = ['AlphaZeroMCTS', 'AlphaZeroNode', 'play_one_episode']
EOF

cat > "$NEW_ROOT/algorithms/mcts_basic/__init__.py" << 'EOF'
from .MCTS import SimplifiedGameState, MCTSAgent

__all__ = ['SimplifiedGameState', 'MCTSAgent']
EOF

# Step 3: Create main README
echo "[3/5] Creating README files..."

cat > "$NEW_ROOT/README.md" << 'EOF'
# Suika-RL: Reinforcement Learning for Suika Game

A comprehensive comparison of RL algorithms for the Suika (Watermelon) Game.

## 📂 Project Structure

```
suika-rl/
├── algorithms/          # All RL algorithm implementations
│   ├── dqn/            # Deep Q-Network
│   ├── mcts_basic/     # Basic MCTS
│   ├── mcts_optimized/ # Optimized MCTS (32x faster)
│   ├── mcts_smart/     # Smart MCTS with heuristics
│   └── alphazero/      # AlphaZero with lookahead reward
│
├── models/             # Neural network architectures
│   ├── SuikaNet.py    # Policy-Value network
│   └── StateConverter.py
│
├── weights/            # Trained model weights
│   ├── dqn/           # DQN checkpoints (183.9 avg score)
│   ├── alphazero/     # AlphaZero iterations
│   └── mcts/          # No weights needed (rule-based)
│
├── training/           # Training scripts
│   ├── train_alphazero.py
│   └── test_dqn_performance.py
│
├── results/            # Experiment results & visualizations
│
├── env/                # Game environment
│   ├── Game.py        # Core game logic
│   └── GameInterface.py
│
└── docs/               # Documentation
    ├── COMPLETE_TRAINING_HISTORY.md
    └── LOOKAHEAD_REWARD_UPDATE.md
```

## 🎯 Algorithm Performance

| Algorithm | Avg Score | Speed | Training Cost |
|-----------|-----------|-------|---------------|
| **Optimized MCTS** | 255 | 0.17s/step | None |
| **DQN** | 183.9 | <0.01s/step | 5000 episodes |
| **Smart MCTS** | 177.3 | 0.43s/step | None |
| **AlphaZero (new)** | TBD | Medium | In progress |
| Random | 133.5 | Fast | None |

## 🚀 Quick Start

### Run DQN Agent
```bash
cd suika-rl/training
python test_dqn_performance.py
```

### Run MCTS
```bash
cd suika-rl/algorithms/mcts_optimized
python run_mcts.py
```

### Train AlphaZero
```bash
cd suika-rl/training
python train_alphazero.py
```

## 📊 Results

See `results/` directory for:
- Performance comparison charts
- Training curves
- Score distributions
- Speed benchmarks

## 📖 Documentation

- `docs/COMPLETE_TRAINING_HISTORY.md` - Full training history
- `docs/LOOKAHEAD_REWARD_UPDATE.md` - New reward system
- `docs/CODE_REVIEW_MCTS.md` - MCTS implementation review

## 🔧 Requirements

```
paddlepaddle==3.2.1
numpy==1.26.4
opencv-python==4.11.0.86
matplotlib==3.7.2
```

## 📝 License

MIT License
EOF

# Step 4: Create results directory structure
echo "[4/5] Setting up results directory..."

mkdir -p "$NEW_ROOT/results/figures"
mkdir -p "$NEW_ROOT/results/data"

cat > "$NEW_ROOT/results/README.md" << 'EOF'
# Experiment Results

## Directory Structure

```
results/
├── figures/           # All visualizations
│   ├── dqn_vs_mcts.png
│   ├── alphazero_training_curve.png
│   └── score_distributions.png
│
└── data/             # Raw experiment data
    ├── dqn_scores.csv
    ├── mcts_scores.csv
    └── comparison.json
```

## How to Generate Results

Run `python generate_results.py` from the training directory.
EOF

# Step 5: Create convenient run scripts
echo "[5/5] Creating convenience scripts..."

cat > "$NEW_ROOT/run_tests.sh" << 'EOF'
#!/bin/bash
# Run all algorithm tests

cd "$(dirname "$0")"

echo "Testing DQN..."
python training/test_dqn_performance.py

echo "Testing AlphaZero..."
# Add test command here

echo "All tests completed!"
EOF

chmod +x "$NEW_ROOT/run_tests.sh"

echo ""
echo "========================================"
echo "  ✅ Reorganization Complete!"
echo "========================================"
echo ""
echo "Project structure created at: $NEW_ROOT"
echo ""
echo "Next steps:"
echo "1. Update import paths in copied files"
echo "2. Test the reorganized code"
echo "3. Generate comparison visualizations"
echo ""
