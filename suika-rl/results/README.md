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
