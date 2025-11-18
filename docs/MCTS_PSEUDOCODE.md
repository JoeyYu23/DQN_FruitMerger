# MCTS Pseudocode for Suika Game

## High-Level Algorithm

```
function MCTS_Search(initial_state, num_simulations):
    root = MCTSNode(initial_state)
    transposition_table = {}

    for i in 1 to num_simulations:
        node = Select(root)
        if not node.is_terminal():
            node = Expand(node)
        value = Simulate(node.state)
        Backpropagate(node, value)

    return BestAction(root)
```

## 1. Selection (PUCT-based)

```
function Select(node):
    path = [node]

    while node has children and not is_terminal:
        if not node.is_fully_expanded():
            break

        # Select child with highest PUCT score
        node = argmax_{child} PUCT(child)
        path.append(node)

    return node

function PUCT(node):
    """
    PUCT Formula (AlphaZero):
    UCT(s,a) = Q(s,a) + c_puct * P(a|s) * sqrt(N(s)) / (1 + N(s,a))
    """
    Q = node.total_value / max(1, node.visit_count)

    U = c_puct * node.prior * sqrt(node.parent.visit_count)
    U = U / (1 + node.visit_count)

    return Q + U
```

## 2. Expansion (with Progressive Widening)

```
function Expand(node):
    valid_actions = get_valid_actions(node.state)

    # Progressive widening: limit number of children
    max_children = min(MAX_ACTIONS,
                      INITIAL_ACTIONS + sqrt(node.visit_count))

    if len(node.children) >= max_children:
        return node

    # Select best unexpanded action
    unexpanded = [a for a in valid_actions
                  if a not in node.expanded_actions]

    if not unexpanded:
        return node

    # Get priors from heuristic policy
    priors = HeuristicPolicy(node.state)

    # Choose action with highest prior
    action = argmax_{a in unexpanded} priors[a]

    # Create new state
    new_state = node.state.copy()
    new_state.apply_action(action)

    # Check transposition table
    state_hash = hash(new_state)
    if state_hash in transposition_table:
        child = transposition_table[state_hash]
        node.children[action] = child
    else:
        child = MCTSNode(new_state, parent=node,
                        action=action, prior=priors[action])
        node.children[action] = child
        transposition_table[state_hash] = child

    node.expanded_actions.add(action)
    return child
```

## 3. Simulation (Heuristic Rollout)

```
function Simulate(state):
    """
    Fast rollout using heuristic policy (not random!)
    """
    state = state.copy()
    depth = 0
    total_reward = 0

    while not state.is_terminal and depth < MAX_DEPTH:
        valid_actions = get_valid_actions(state)
        if not valid_actions:
            break

        # Use heuristic to select action (NOT random)
        action = HeuristicSelectAction(state)

        reward = state.apply_action(action)
        total_reward += reward
        depth += 1

    # Calculate final value
    value = state.score - HEIGHT_PENALTY * state.max_height

    if state.is_terminal:
        value -= DEATH_PENALTY

    return value

function HeuristicSelectAction(state):
    """
    Score actions based on domain knowledge
    """
    scores = {}
    for action in valid_actions:
        score = 0

        # 1. Merge potential (adjacency to same fruit)
        score += 5.0 * count_adjacent_matches(state, action)

        # 2. Center bias (prefer middle columns)
        center_dist = abs(action - state.width/2)
        score += 2.0 * (1 - center_dist / (state.width/2))

        # 3. Height penalty (avoid tall stacks)
        column_height = get_column_height(state, action)
        score -= 3.0 * (column_height / state.height)

        # 4. Overflow risk
        if would_overflow(state, action):
            score -= 10.0

        scores[action] = score

    # Softmax sampling
    probs = softmax(scores)
    return sample(probs)
```

## 4. Backpropagation

```
function Backpropagate(node, value):
    """
    Update statistics from leaf to root
    """
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        node = node.parent
```

## 5. Best Action Selection

```
function BestAction(root):
    """
    Select action with most visits (most reliable)
    """
    if not root.children:
        return random_valid_action()

    return argmax_{action} root.children[action].visit_count
```

## Zobrist Hashing (State Deduplication)

```
function InitializeZobristTable():
    """
    Pre-generate random numbers for hashing
    """
    table = {}
    for row in 0 to GRID_HEIGHT:
        for col in 0 to GRID_WIDTH:
            for fruit_type in 0 to 10:
                table[(row, col, fruit_type)] = random_64bit()
    return table

function ZobristHash(state):
    """
    Fast incremental hashing
    """
    hash = 0
    for row in 0 to state.height:
        for col in 0 to state.width:
            fruit = state.grid[row, col]
            if fruit != 0:
                hash ^= zobrist_table[(row, col, fruit)]

    # Include current fruit
    hash ^= zobrist_table[('current', state.current_fruit)]

    return hash
```

## Simplified Physics (Deterministic Approximation)

```
function ApplyAction(state, action):
    """
    Drop fruit at column and process merges
    """
    col = action
    fruit_type = state.current_fruit

    # 1. Find landing position
    row = state.height - 1
    for r in reverse(range(state.height)):
        if state.grid[r, col] != 0:
            row = r - 1
            break

    # 2. Check game over
    if row < WARNING_LINE:
        state.is_terminal = True
        return -DEATH_PENALTY

    # 3. Place fruit
    state.grid[row, col] = fruit_type

    # 4. Process merges (cascade)
    reward = ProcessMerges(state, row, col)

    # 5. Generate next fruit
    state.current_fruit = random_fruit_type()

    return reward

function ProcessMerges(state, start_row, start_col):
    """
    Deterministic merge processing
    """
    total_reward = 0
    changed = True

    while changed:
        changed = False

        # Check all cells for mergeable neighbors
        for row, col in all_cells:
            fruit = state.grid[row, col]
            if fruit == 0 or fruit >= 10:
                continue

            # Check 4-neighbors
            for (dr, dc) in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc) and state.grid[nr, nc] == fruit:
                    # Merge!
                    state.grid[row, col] = 0
                    state.grid[nr, nc] = 0

                    # Place new fruit at lower position
                    new_type = fruit + 1
                    target_row = max(row, nr)
                    target_col = col if row > nr else nc

                    # Apply gravity
                    final_row = apply_gravity(state, target_row, target_col)
                    state.grid[final_row, target_col] = new_type

                    # Score
                    reward = 100 if new_type == 10 else new_type
                    total_reward += reward
                    state.score += reward

                    changed = True
                    break

            if changed:
                break

    return total_reward
```

## Complexity Analysis

### Time Complexity
- **Selection**: O(depth) with depth ≈ 30-50
- **Expansion**: O(1) amortized with progressive widening
- **Simulation**: O(MAX_DEPTH × GRID_SIZE) ≈ O(50 × 160) = O(8000)
- **Backpropagation**: O(depth) ≈ O(30-50)
- **Per simulation**: O(MAX_DEPTH × GRID_SIZE) dominated by simulation
- **Total**: O(NUM_SIMS × MAX_DEPTH × GRID_SIZE)
  - With NUM_SIMS=2000: ~16M operations
  - Target: <1 second → >16M ops/sec (achievable in Python with numpy)

### Space Complexity
- **Tree nodes**: O(NUM_SIMS × branching_factor / progressive_widening)
  - With progressive widening: ≈ 10K-100K nodes
  - Each node: ~100 bytes
  - Total: 1-10 MB (very manageable!)
- **Transposition table**: O(unique_states) ≈ 10K-100K entries
  - ~10 MB
- **Total memory**: <100 MB (safe!)

## Key Optimizations

1. **Progressive Widening**: Reduces branching from 20 → 3-10
2. **Transposition Table**: Deduplicates states (10-50% reduction)
3. **Heuristic Rollouts**: 10x faster + better signal than random
4. **Depth Limiting**: Prevents infinite loops
5. **Grid Compression**: 10×16 grid vs continuous physics (1000x faster)
6. **Early Termination**: Stop on game over immediately
7. **Numpy Operations**: Vectorized grid operations

## Expected Performance

- **Rollouts/second**: 2000-5000
- **Tree size**: 10K-100K nodes
- **Memory usage**: 50-200 MB
- **Search depth**: 30-50 moves ahead
- **Quality**: Significantly better than DQN for planning
