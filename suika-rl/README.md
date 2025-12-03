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
