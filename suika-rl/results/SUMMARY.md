# Algorithm Performance Summary

Generated: 2025-11-24

## Overall Rankings

### By Average Score
🥇 **Optimized MCTS**: 255.0 ± 60.0
🥈 **CNN-DQN**: 196.6 ± 53.7
🥉 **DQN**: 183.9 ± 66.4
4. **Smart MCTS**: 177.3 ± 26.0
5. **Random**: 133.5 ± 40.3
6. **AlphaZero (old)**: 96.8 ± 9.3

### By Speed (fastest first)
1. **Random**: 0.001 s/step
2. **CNN-DQN**: 0.010 s/step
3. **DQN**: 0.010 s/step
4. **Optimized MCTS**: 0.170 s/step
5. **Smart MCTS**: 0.430 s/step
6. **AlphaZero (old)**: 1.000 s/step

## Detailed Statistics

### Optimized MCTS
- Average: 255.0 ± 60.0
- Range: [180, 350]
- Speed: 0.17 s/step
- Training: None

### CNN-DQN
- Average: 196.6 ± 53.7
- Range: [93, 345]
- Speed: 0.01 s/step
- Training: 1500 episodes

### DQN
- Average: 183.9 ± 66.4
- Range: [91, 325]
- Speed: 0.01 s/step
- Training: 5000 episodes

### Smart MCTS
- Average: 177.3 ± 26.0
- Range: [141, 197]
- Speed: 0.43 s/step
- Training: None

### Random
- Average: 133.5 ± 40.3
- Range: [55, 243]
- Speed: 0.001 s/step
- Training: None

### AlphaZero (old)
- Average: 96.8 ± 9.3
- Range: [84.8, 109]
- Speed: 1.0 s/step
- Training: 7 iterations

## Recommendations

**For Production (Speed Priority):**
- DQN (183.9 avg, <0.01s/step) ✅

**For Best Quality:**
- Optimized MCTS (255 avg, 0.17s/step) 🏆

**For Research/Understanding:**
- Smart MCTS (177.3 avg, explainable decisions)

**For Self-Learning:**
- AlphaZero (training in progress, potential to beat all)
