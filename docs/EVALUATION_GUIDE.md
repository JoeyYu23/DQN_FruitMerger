# Evaluation Guide: Comparing Different Models

This guide explains how to test and compare different AI agents for the Fruit Merger game.

## Quick Start

### Test All Models at Once

```bash
python benchmark_all.py
```

This will:
- Evaluate DQN, MCTS variants, and Random baseline
- Run 100 episodes per agent (customizable)
- Generate comprehensive comparison statistics
- Save results to JSON and LaTeX table

### Custom Number of Episodes

```bash
python benchmark_all.py 200  # Run 200 episodes
```

## Testing Individual Models

### 1. DQN Model

**Standard Evaluation:**
```bash
python evaluate_multi_games.py
```

**Quick Evaluation:**
```bash
python evaluate.py
```

**What it tests:**
- DQN agent vs Random baseline
- 100-200 episodes
- Outputs mean, max, min scores
- Shows score distribution histogram
- Calculates win rate

### 2. MCTS Variants

**Compare MCTS versions:**
```bash
python scripts/compare_mcts_versions.py [num_games] [simulations]

# Examples:
python scripts/compare_mcts_versions.py 10 100   # 10 games, 100 simulations/step
python scripts/compare_mcts_versions.py 20 200   # 20 games, 200 simulations/step
```

**Run individual MCTS:**
```bash
python scripts/run_mcts.py
python scripts/run_fast_mcts.py
```

## Evaluation Metrics

Each evaluation reports:

| Metric | Description |
|--------|-------------|
| **Score** | Final game score (primary metric) |
| **Reward** | Cumulative reward during episode |
| **Steps** | Number of actions taken |
| **Time/Step** | Average computation time per action |
| **Win Rate** | Percentage of games won vs baseline |

### Statistical Measures

- **Mean (μ)**: Average performance
- **Std Dev (σ)**: Performance variance
- **Max**: Best score achieved
- **Min**: Worst score
- **Median**: 50th percentile (robust to outliers)

## Understanding the Results

### Score Interpretation

| Score Range | Performance Level |
|-------------|-------------------|
| 0-100 | Poor (random level) |
| 100-150 | Below average |
| 150-200 | Average |
| 200-250 | Good |
| 250-300 | Very good |
| 300+ | Excellent |

### Win Rate

- **>70%**: Significantly better
- **50-70%**: Moderately better
- **45-55%**: Roughly equal
- **<45%**: Worse performance

### Effect Size (Cohen's d)

- **|d| < 0.2**: Negligible difference
- **0.2 ≤ |d| < 0.5**: Small difference
- **0.5 ≤ |d| < 0.8**: Medium difference
- **|d| ≥ 0.8**: Large difference

## Comparing Your Own Models

### For New DQN Models

1. Train your model and save weights:
   ```python
   paddle.save(agent.policy_net.state_dict(), "my_model.pdparams")
   ```

2. Modify `benchmark_all.py` to load your model:
   ```python
   my_agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
   my_agent.policy_net.set_state_dict(paddle.load("my_model.pdparams"))
   benchmark.results['My Model'] = benchmark.evaluate_agent(
       my_agent, 'My Model', use_env=False
   )
   ```

3. Run the benchmark:
   ```bash
   python benchmark_all.py
   ```

### For New MCTS Variants

1. Implement your agent with a `predict(env)` method:
   ```python
   class MyMCTS:
       def predict(self, env):
           # Your MCTS logic here
           return action  # Should return integer or [integer]
   ```

2. Add to `benchmark_all.py`:
   ```python
   my_mcts = MyMCTS(num_simulations=100)
   benchmark.results['My MCTS'] = benchmark.evaluate_agent(
       my_mcts, 'My MCTS', use_env=True
   )
   ```

3. Run benchmark

### For Custom Algorithms

Implement an agent class with either:
- `predict(feature)` - for feature-based methods (like DQN)
- `predict(env)` - for methods needing full environment (like MCTS)

Then add to benchmark as shown above.

## Testing Protocol

### Standard Protocol (Recommended)

1. **Sample Size**: 100-200 episodes
2. **Seeds**: Sequential (0, 1, 2, ..., N-1)
3. **First Action**: Random (to ensure game starts)
4. **Comparison**: Same seeds for all agents

### Why Use Same Seeds?

Using identical seeds ensures:
- Each agent faces the exact same fruit sequences
- Fair head-to-head comparison
- Results are reproducible
- Differences reflect algorithm quality, not luck

## Output Files

After running benchmarks, you'll get:

1. **`benchmark_results_YYYYMMDD_HHMMSS.json`**
   - Complete raw data
   - All scores, rewards, steps for each episode
   - Statistics for each agent
   - Metadata (timestamp, configuration)

2. **`benchmark_table.tex`**
   - LaTeX table ready for papers
   - Formatted with proper math notation
   - Can be directly included in reports

3. **Console output**
   - Real-time progress
   - Summary statistics
   - Win rate matrix
   - Comparative analysis

## Example Workflow

### Scenario: Testing 3 Different DQN Models

```bash
# 1. Train models (creates model1.pdparams, model2.pdparams, model3.pdparams)
python quick_train.py  # or your custom training script

# 2. Create custom benchmark script
cp benchmark_all.py benchmark_my_models.py

# 3. Edit benchmark_my_models.py to load your 3 models
# (see "Comparing Your Own Models" section above)

# 4. Run benchmark
python benchmark_my_models.py 200  # 200 episodes for statistical power

# 5. Analyze results
# - Check JSON file for detailed data
# - Use LaTeX table in your report
# - Identify best model based on mean score and win rate
```

### Scenario: Tuning MCTS Simulations

```bash
# Test different simulation counts
python scripts/compare_mcts_versions.py 20 50
python scripts/compare_mcts_versions.py 20 100
python scripts/compare_mcts_versions.py 20 200
python scripts/compare_mcts_versions.py 20 500

# Compare performance vs. computational cost
# Plot: simulation count vs. (score, time/step)
```

## Best Practices

✅ **DO:**
- Use at least 100 episodes for reliable statistics
- Use same seed set when comparing agents
- Report both mean and standard deviation
- Include computational cost (time/step)
- Save raw data, not just summaries
- Version control your evaluation scripts

❌ **DON'T:**
- Cherry-pick favorable seeds
- Compare with different seed sets
- Report only best results
- Ignore variance/stability
- Skip baseline comparisons

## Troubleshooting

**Problem**: "MCTS modules not found"
```bash
# Solution: Make sure MCTS files are in mcts/ folder
ls mcts/MCTS*.py
```

**Problem**: "final.pdparams not found"
```bash
# Solution: Train a model first or specify different path
python quick_train.py
```

**Problem**: Evaluation is too slow
```bash
# Solution: Reduce episodes or use faster agent
python benchmark_all.py 50  # Only 50 episodes
# Or reduce MCTS simulations in the script
```

**Problem**: Results have high variance
```bash
# Solution: Increase sample size
python benchmark_all.py 500  # More episodes = more stable
```

## For Your Report/Paper

Include these elements:

1. **Method Description**: Brief description of each agent
2. **Evaluation Setup**:
   - Number of episodes
   - Seed range
   - Hardware specs (if timing is critical)
3. **Results Table**: Use the generated LaTeX table
4. **Statistical Analysis**:
   - Mean ± Std Dev
   - Win rates vs. baseline
   - Effect sizes
5. **Discussion**:
   - Performance vs. computational cost trade-offs
   - Practical implications
   - Limitations

## References

- **Full Methodology**: See `docs/evaluation_methodology.tex`
- **Game Mechanics**: See `docs/game_mechanics.tex`
- **Code Examples**: See `evaluate_multi_games.py`, `scripts/compare_mcts_versions.py`

## Questions?

Check the documentation:
- Game mechanics: `docs/game_mechanics.tex`
- Evaluation methodology: `docs/evaluation_methodology.tex`
- MCTS guides: `docs/MCTS_README.md`, `docs/智能MCTS说明.md`
