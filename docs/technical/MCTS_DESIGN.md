# MCTS Agent Design for Suika Game

## Algorithm Overview

### Core MCTS Stages

```
1. SELECTION: Navigate tree using PUCT formula
2. EXPANSION: Add new child nodes (with progressive widening)
3. SIMULATION: Fast rollout using heuristic policy
4. BACKPROPAGATION: Update statistics up the tree
```

### PUCT Formula (AlphaZero-style UCT)

```
UCT(s, a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))

Where:
- Q(s,a): Mean value of taking action a in state s
- P(a|s): Prior policy probability (from heuristics or NN)
- N(s): Visit count of state s
- N(s,a): Visit count of state-action pair
- c_puct: Exploration constant (~1.5)
```

## Explosion Control Strategies

### 1. State Hashing & Transposition Table
- **Zobrist Hashing**: Fast collision-resistant hashing
- **Transposition Table**: Merge identical states across tree
- **Memory**: O(unique_states) instead of O(tree_nodes)

### 2. Progressive Widening
```python
max_children(N) = min(MAX_ACTIONS, 3 + int(sqrt(N)))
```
- Start with 3-5 actions
- Expand to more as visits increase
- Reduces initial branching factor

### 3. Depth Limiting
- Simulation depth: 30-50 moves
- Early termination on game over
- Prevents infinite loops

### 4. Smart Action Selection
- Discretize X positions to ~20 bins
- Prune obviously bad moves
- Focus on promising regions

## Simplified Physics Model

### Grid Representation
```
Grid: 10 (width) × 16 (height)
Cell values: 0 (empty) to 10 (watermelon)
```

### Deterministic Physics Approximation
1. **Drop**: Fruit falls straight down at X position
2. **Settle**: Stack on first collision
3. **Merge**: Adjacent same-type fruits combine
4. **Cascade**: Repeat until stable

This approximation trades accuracy for speed (~1000x faster).

## Heuristic Simulation Policy

### Action Selection During Rollout
```python
score(x) = merge_potential(x)
           - overflow_risk(x)
           + center_bias(x)
           - height_penalty(x)
```

### Heuristics:
1. **Merge potential**: Count adjacent same-type fruits
2. **Overflow risk**: Penalize if height near limit
3. **Center bias**: Prefer x in [0.3, 0.7] range
4. **Height penalty**: Avoid making towers

## Value Estimation

### Basic (No NN)
```python
value = total_score - 10 * max_height
```

### Advanced (With NN)
```python
Input: Flattened grid (160 floats)
Hidden: [128, 64] with ReLU
Output: value (1 float) + policy (20 floats)
```

## Performance Targets

- **Rollouts/move**: >2000 in 1 second
- **Memory**: <500MB for full tree
- **Tree size**: <100K nodes typical
- **Depth**: Average 30-40 moves ahead

## Implementation Architecture

```
MCTSAgent
├── GameState (simplified, hashable)
├── MCTSNode (with PUCT)
├── TranspositionTable (Zobrist hashing)
├── SimulationPolicy (heuristic-based)
└── SearchTree (selection/expansion/simulation/backprop)
```
