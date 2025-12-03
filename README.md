# Suika Game RL: Algorithm File Guide

**Generated:** 2025-12-03

---

## 🎮 **Core Game Environment Files** (Required for All Algorithms)

### Essential Files (5 files)
| File | Size | Description |
|------|------|-------------|
| `Game.py` | 17K | Core game logic (pymunk physics engine) |
| `GameInterface.py` | 5.1K | Environment interface (state/reward/action) |
| `GameEvent.py` | 1.1K | Game event handling |
| `PRNG.py` | 1.0K | Pseudo-random number generator (reproducible seeds) |
| `render_utils.py` | 3.5K | Rendering utilities |

### Resource Folders
- `resources/images/` - Fruit image assets
- `resources/illustrations/` - Illustration resources

---

## 1️⃣ **DQN (MLP-DQN with PaddlePaddle)**

### Core Files (4 files)
| File | Size | Description |
|------|------|-------------|
| `DQN.py` | 9.8K | **Main file** - MLP-DQN implementation |
| `SuikaNet.py` | 11K | Neural network definition (MLP architecture) |
| `StateConverter.py` | 7.6K | State converter (optional) |
| `evaluate.py` | 1.7K | Standard evaluation script |

### Training Scripts
- `train_5000.py` (12K) - 5000 episodes training
- `train_with_logging.py` (7.6K) - Training with logging

### Weight Files
```
weights/
├── best_model.pdparams          # Best model
├── checkpoint_ep500.pdparams
├── checkpoint_ep1000.pdparams
├── ...
└── checkpoint_ep5000.pdparams
```

Also in root directory:
- `final.pdparams` (197K) - Final training weights
- `final_5000.pdparams` (197K) - 5000 episodes training weights

### Dependencies
```python
import paddle
from GameInterface import GameInterface
from PRNG import PRNG
```

### Testing/Demo
- `test_dqn_performance.py` (3.1K) - Performance testing
- `AIPlay.py` (2.4K) - AI game demo
- `AIPlay_Auto.py` (6.1K) - Automatic AI demo

---

## 2️⃣ **CNN-DQN (PyTorch)**

### Core Files (3 files)
| File | Size | Description |
|------|------|-------------|
| `CNN_DQN.py` | 19K | **Main file** - CNN-DQN implementation (complete) |
| `SuikaNet_torch.py` | 7.9K | PyTorch neural network (optional, included in CNN_DQN.py) |
| `test_cnn_final.py` | 1.4K | Final test script |

### Weight Files
```
weights_cnn_dqn/
├── best_model.pth              # 🏆 Best model (ep1600, score 205.7)
├── final_model.pth             # Final model (ep2000)
├── checkpoint_ep500.pth
├── checkpoint_ep1000.pth
├── checkpoint_ep1500.pth
└── checkpoint_ep2000.pth
```

### Dependencies
```python
import torch
import torch.nn as nn
from GameInterface import GameInterface
```

### Features
- **CNN Architecture**: Preserves spatial structure
- **Input Format**: (2, 20, 16) - 2 channels, 20x16 grid
- **Higher Performance**: 205.7 ± 51.1 (vs DQN's 183.9)
- **Less Training**: 1600 episodes (vs DQN's 5000)

---

## 3️⃣ **MCTS (Real Physics)**

### Core Files
| File | Size | Description |
|------|------|-------------|
| `mcts/MCTS_real_physics.py` | 21K | **Main file** - Real physics MCTS implementation |
| `test_real_physics_mcts.py` | 5.8K | Testing and comparison script |
| `evaluate_mcts_real_physics.py` | 3.9K | MCTS evaluation script |

### Dependencies
```python
import numpy as np
from Game import FRUIT_RADIUS
from GameInterface import GameInterface
```

### Features
- **No Training Required**: Pure search algorithm
- **Uses Real Physics Engine**: Complete pymunk simulation
- **Two-Step Lookahead**: Evaluates current action + best future action
- **Smart Rewards**: Merge rewards + position advantages - height penalty
- **Safety Filtering**: Avoids dangerous positions near death line
- **Merge Priority**: Prioritizes actions that create merges

---

## 📁 **Utility Files/Tools**

### Visualization/Analysis
- `test_model_visual.py` (9.6K) - Model decision visualization
- `analyze_high_score.py` (9.7K) - High score game analysis
- `CompareAgents.py` (11K) - Multi-algorithm comparison
- `benchmark_all.py` (13K) - Complete benchmark

### Script Tools (scripts/)
- `scripts/run_mcts.py` - Run MCTS
- `scripts/record_top_games.py` - Record high-score games
- `scripts/compare_mcts_versions.py` - Compare MCTS versions

### Other Game Modes
- `InteractivePlay.py` (1.0K) - Human play
- `RandomPlay.py` (4.7K) - Random play
- `SelfPlay.py` (7.7K) - Self-play

---

## 📦 **suika-rl Subproject** (Optional, Standalone Version)

```
suika-rl/
├── algorithms/
│   ├── dqn/              # PaddlePaddle DQN (complete implementation)
│   ├── cnn_dqn/          # PyTorch CNN-DQN (complete implementation)
│   ├── mcts_basic/       # Basic MCTS
│   ├── mcts_optimized/   # Optimized MCTS
│   └── mcts_smart/       # Smart MCTS
│
├── weights/              # Algorithm weights
│   ├── dqn/
│   ├── cnn_dqn/
│   └── mcts/
│
├── results/              # Results summary
│   ├── data/comparison.json
│   └── figures/
│
└── training/             # Training scripts
```

**Note**: `suika-rl/` is the organized version of the project with complete algorithm implementations. The main directory contains original development files. **Both are functionally identical; use either one.**

---

## 🎯 **Quick Start Guide**

### Running DQN
```bash
# Use trained model
python AIPlay.py  # Uses final.pdparams

# Evaluate performance
python evaluate.py  # 200 tests
```

### Running CNN-DQN
```bash
# Use best model
python test_cnn_final.py

# Or modify test_cnn_final.py to use best_model.pth
```

### Running MCTS
```bash
# Real Physics MCTS
python test_real_physics_mcts.py --seed 888 --sims 50 --steps 100

# Compare different MCTS versions
python test_real_physics_mcts.py --compare
```

---

## 📊 **Performance Comparison**

| Algorithm | Avg Score | Training Cost | Inference Speed | Core Files |
|-----------|-----------|---------------|-----------------|------------|
| **CNN-DQN** | 205.7 | 1600 ep | 0.01s/step | 3 |
| **DQN** | 183.9 | 5000 ep | 0.01s/step | 4 |
| **MCTS (Real)** | 231.92 | None | 1.0s/step | 1 |

---

## 🗂️ **Minimal File Sets**

### To Run DQN (minimum required):
```
Game.py
GameInterface.py
GameEvent.py
PRNG.py
DQN.py
weights/best_model.pdparams
resources/images/
```

### To Run CNN-DQN (minimum required):
```
Game.py
GameInterface.py
GameEvent.py
CNN_DQN.py
weights_cnn_dqn/best_model.pth
resources/images/
```

### To Run MCTS (minimum required):
```
Game.py
GameInterface.py
GameEvent.py
PRNG.py
mcts/MCTS_real_physics.py
resources/images/
```

---

## 📝 **Dependencies**

### DQN (PaddlePaddle)
```
paddlepaddle
pymunk
numpy
opencv-python
```

### CNN-DQN (PyTorch)
```
torch
pymunk
numpy
opencv-python
```

### MCTS (No Machine Learning)
```
pymunk
numpy
opencv-python
```

---

## 📚 **Documentation**

For detailed documentation, see:
- `docs/technical/ALGORITHMS_FILES.md` - Complete algorithm file guide
- `docs/technical/RESULTS_SUMMARY.md` - Performance results and analysis
- `docs/training/` - Training guides and tutorials
- `docs/deployment/` - Deployment and setup guides

---

## 🚀 **Key Highlights**

- **Simplest**: MCTS (no training, 1 file)
- **Fastest**: CNN-DQN/DQN (0.01s/step)
- **Most Accurate**: MCTS (231.92 score)
- **Best Learning**: CNN-DQN (205.7 score, 70% less training than DQN)
- **Most Flexible**: MCTS (many tunable parameters)

---

## 📄 **License**

See [LICENSE](LICENSE) file for details.

---

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Project Repository:** https://github.com/JoeyYu23/DQN_FruitMerger

**Last Updated:** 2025-12-03
