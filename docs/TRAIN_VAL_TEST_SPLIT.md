# Train/Validation/Test Split - Proper Seed Management

## 🚨 Current Problem

**训练集、验证集、测试集的种子没有正确分离！**

### What's Wrong?

在当前代码中：

```python
# DQN.py 和 evaluate.py 都使用相同的PRNG种子
evaluate_random = PRNG()
evaluate_random.seed("RedContritio")  # ❌ 同一个种子！

# 训练中评估（25次）
for i in range(25):
    seed = evaluate_random.random()  # 生成seed[0], seed[1], ..., seed[24]
    evaluate(env, agent, seed)

# 最终测试（200次）
for i in range(200):
    seed = evaluate_random.random()  # 生成seed[0], seed[1], ..., seed[199]
    evaluate(env, agent, seed)
```

**问题：前25个测试种子在训练期间已经被看过了！**

这导致：
- ❌ **数据泄露（Data Leakage）**：模型间接看到了测试集
- ❌ **过拟合风险**：模型可能记住了这些特定场景
- ❌ **评估不准确**：测试分数可能被高估
- ❌ **不符合ML最佳实践**

---

## ✅ Correct Approach

### Principle: 三个数据集必须完全独立

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Dataset                         │
│  - 用途: 训练模型，更新参数                                    │
│  - 种子: 随机（每次episode不同）                               │
│  - 数量: 越多越好（500-2000+ episodes）                        │
│  - 特点: 高度多样化，避免过拟合                                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Validation Dataset                        │
│  - 用途: 监控训练进度，早停，超参数调优                          │
│  - 种子: 固定（PRNG("VALIDATION_2024")）                      │
│  - 数量: 中等（50-100 episodes）                              │
│  - 特点: 训练期间可以看，但不用于更新参数                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                      Test Dataset                           │
│  - 用途: 最终评估，报告模型性能                                 │
│  - 种子: 固定（PRNG("TEST_2024")，与验证集不同！）               │
│  - 数量: 较多（200-500 episodes）                             │
│  - 特点: 训练期间绝不使用，完全独立                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Rules

1. **训练集（Training）**: 随机种子，每次都不同
2. **验证集（Validation）**: 固定种子集A，训练期间可用
3. **测试集（Test）**: 固定种子集B，与A完全不同，训练后才用

**绝对不能：训练中看到的场景出现在测试集中！**

---

## 🔧 Implementation

### Using SeedManager (Recommended)

我们提供了 `seed_management.py` 来正确管理种子：

```python
from seed_management import SeedManager

# 1. 初始化种子管理器
seed_mgr = SeedManager(
    val_seed="VALIDATION_2024",    # 验证集主种子
    test_seed="TEST_2024",          # 测试集主种子（必须不同！）
    num_val=50,                     # 50个验证场景
    num_test=200                    # 200个测试场景
)

# 2. 训练期间
for episode in range(num_train_episodes):
    # 使用随机种子训练
    train_seed = seed_mgr.get_train_seed()  # 返回None
    env.reset(seed=train_seed)
    # ... 训练代码 ...

    # 每N个episode，在验证集上评估
    if episode % 100 == 0:
        val_seeds = seed_mgr.get_val_seeds()
        val_scores = []
        for seed in val_seeds:
            env.reset(seed=seed)
            score, _ = evaluate(env, agent)
            val_scores.append(score)
        print(f"Validation mean: {np.mean(val_scores)}")

# 3. 训练完成后，在测试集上评估
test_seeds = seed_mgr.get_test_seeds()
test_scores = []
for seed in test_seeds:
    env.reset(seed=seed)
    score, _ = evaluate(env, agent)
    test_scores.append(score)

print(f"Final test mean: {np.mean(test_scores)}")
```

### Manual Implementation

如果不想用SeedManager，可以手动设置：

```python
from PRNG import PRNG

# 生成验证集种子（固定）
val_prng = PRNG()
val_prng.seed("VALIDATION_2024")
val_seeds = [val_prng.random() for _ in range(50)]

# 生成测试集种子（固定，但不同）
test_prng = PRNG()
test_prng.seed("TEST_2024")
test_seeds = [test_prng.random() for _ in range(200)]

# 确保没有重叠
assert len(set(val_seeds) & set(test_seeds)) == 0, "Val and test overlap!"
```

---

## 📊 Recommended Configuration

### For Quick Experiments

```python
seed_mgr = SeedManager(
    val_seed="VAL_QUICK",
    test_seed="TEST_QUICK",
    num_val=20,      # 快速验证
    num_test=50      # 快速测试
)
```

### For Standard Research

```python
seed_mgr = SeedManager(
    val_seed="VALIDATION_2024",
    test_seed="TEST_2024",
    num_val=50,      # 标准验证
    num_test=200     # 标准测试
)
```

### For Publication-Quality Results

```python
seed_mgr = SeedManager(
    val_seed="VALIDATION_FINAL",
    test_seed="TEST_FINAL",
    num_val=100,     # 充分验证
    num_test=500     # 充分测试
)
```

---

## 🔄 Migration Guide

### Updating Existing Code

**Before (❌ Wrong):**

```python
# DQN.py
evaluate_random = PRNG()
evaluate_random.seed("RedContritio")

def compare_with_random(env, agent, action_count):
    for _ in range(25):
        seed = evaluate_random.random()  # ❌ 与测试集重叠！
        evaluate(env, agent, seed)
```

**After (✅ Correct):**

```python
# DQN.py
from seed_management import get_default_seed_manager

seed_mgr = get_default_seed_manager()

def compare_with_random(env, agent, action_count):
    val_seeds = seed_mgr.get_val_seeds()[:25]  # 只用前25个
    for seed in val_seeds:
        evaluate(env, agent, seed)
```

---

**Before (❌ Wrong):**

```python
# evaluate.py
evaluate_random = PRNG()
evaluate_random.seed("RedContritio")

for _ in range(200):
    seed = evaluate_random.random()  # ❌ 与验证集重叠！
    evaluate(env, agent, seed)
```

**After (✅ Correct):**

```python
# evaluate.py
from seed_management import get_default_seed_manager

seed_mgr = get_default_seed_manager()
test_seeds = seed_mgr.get_test_seeds()

for seed in test_seeds:
    evaluate(env, agent, seed)
```

---

## 📈 Example: Complete Training Script

```python
"""
正确的训练流程示例
"""
import numpy as np
from DQN import Agent, build_model, ReplayMemory, run_episode
from GameInterface import GameInterface
from seed_management import SeedManager

# 初始化
env = GameInterface()
agent = Agent(build_model, feature_dim, action_dim)
memory = ReplayMemory()

# 种子管理
seed_mgr = SeedManager(
    val_seed="VALIDATION_2024",
    test_seed="TEST_2024",
    num_val=50,
    num_test=200
)

# ===== 训练阶段 =====
print("Training...")
best_val_score = 0

for episode in range(2000):
    # 训练：使用随机种子
    train_seed = seed_mgr.get_train_seed()  # None
    env.reset(seed=train_seed)
    run_episode(env, agent, memory, episode)

    # 验证：每100 episodes
    if episode % 100 == 0:
        val_seeds = seed_mgr.get_val_seeds()
        val_scores = []

        for seed in val_seeds:
            env.reset(seed=seed)
            score, _ = evaluate(env, agent)
            val_scores.append(score)

        val_mean = np.mean(val_scores)
        print(f"Episode {episode} - Val score: {val_mean:.1f}")

        # 保存最佳模型（基于验证集）
        if val_mean > best_val_score:
            best_val_score = val_mean
            paddle.save(agent.policy_net.state_dict(), "best_model.pdparams")
            print(f"  ✅ New best model saved!")

# ===== 测试阶段（训练完成后） =====
print("\nFinal Testing...")

# 加载最佳模型
agent.policy_net.set_state_dict(paddle.load("best_model.pdparams"))

# 在测试集上评估（完全独立的种子）
test_seeds = seed_mgr.get_test_seeds()
test_scores = []

for seed in test_seeds:
    env.reset(seed=seed)
    score, _ = evaluate(env, agent)
    test_scores.append(score)

print(f"Final Test Results:")
print(f"  Mean: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}")
print(f"  Max: {np.max(test_scores)}")
print(f"  Median: {np.median(test_scores):.1f}")

# 保存种子配置（用于论文复现）
seed_mgr.save_seeds("final_seeds.txt")
```

---

## 🎯 Best Practices

### ✅ DO

1. **使用SeedManager或手动确保种子分离**
   ```python
   seed_mgr = SeedManager(val_seed="VAL", test_seed="TEST")
   ```

2. **训练时使用随机种子**
   ```python
   env.reset(seed=None)  # 或 env.reset()
   ```

3. **保存种子配置**
   ```python
   seed_mgr.save_seeds("seeds.txt")  # 便于复现
   ```

4. **验证没有重叠**
   ```python
   seed_mgr.verify_no_overlap()
   ```

5. **文档化你的种子策略**
   ```python
   # 在论文中写明：
   # "We used 50 validation episodes (seed: VALIDATION_2024)
   #  and 200 test episodes (seed: TEST_2024), ensuring no overlap."
   ```

### ❌ DON'T

1. **不要在训练和测试中使用相同的PRNG种子**
   ```python
   # ❌ 错误
   prng = PRNG()
   prng.seed("SAME_SEED")
   train_seeds = [prng.random() for _ in range(100)]
   test_seeds = [prng.random() for _ in range(100)]  # 会重叠！
   ```

2. **不要在训练中使用测试集**
   ```python
   # ❌ 错误
   test_seeds = seed_mgr.get_test_seeds()
   # 在训练循环中使用test_seeds做任何事情
   ```

3. **不要忘记固定随机种子（验证/测试时）**
   ```python
   # ❌ 错误：测试时用随机种子
   env.reset()  # 每次测试结果都不同！
   ```

4. **不要用sequential seeds**
   ```python
   # ❌ 错误：容易重叠
   val_seeds = list(range(0, 50))
   test_seeds = list(range(50, 250))
   # 虽然看起来分开了，但建议用PRNG生成更随机的种子
   ```

---

## 📝 For Your Report/Paper

在论文中，应该这样描述：

### Method Section

```
We split our evaluation into validation and test sets to prevent
data leakage. During training, we used randomly seeded episodes
to maximize diversity. For validation (monitoring training progress),
we used 50 episodes with seeds generated from PRNG("VALIDATION_2024").
For final testing, we used 200 completely independent episodes with
seeds from PRNG("TEST_2024"). We verified no overlap between
validation and test sets. All seeds are saved for reproducibility.
```

### Results Section

```
Table 1: Performance on Test Set (200 episodes, seed: TEST_2024)

Model          Mean Score   Std Dev   Max    Median
---------------------------------------------------
DQN            245.3±42.1   42.1      368    238.5
Smart MCTS     223.7±38.5   38.5      319    220.0
Random         145.8±28.7   28.7      203    142.0

Note: Test set was completely independent from training and validation,
ensuring unbiased evaluation.
```

---

## 🔍 Verification

运行以下代码验证你的设置：

```bash
python seed_management.py
```

输出应该显示：
```
✅ Seed Manager initialized:
   Validation: 50 episodes (seed: 'VALIDATION_2024')
   Test: 200 episodes (seed: 'TEST_2024')
✅ Verified: No overlap between validation and test sets
```

---

## 📚 Summary

| Dataset | Purpose | Seed | Size | Usage |
|---------|---------|------|------|-------|
| **Train** | 训练模型 | 随机（None） | 500-2000+ | 更新参数 |
| **Validation** | 监控进度 | PRNG("VAL") | 50-100 | 调优、早停 |
| **Test** | 最终评估 | PRNG("TEST") | 200-500 | 报告性能 |

**关键原则：三个集合完全独立，训练中绝不使用测试集！**

---

## 🔗 Related Files

- `seed_management.py` - Seed management implementation
- `docs/evaluation_methodology.tex` - Full evaluation methodology
- `docs/EVALUATION_GUIDE.md` - Evaluation guide
- `benchmark_all.py` - Update this to use SeedManager!

---

## ❓ FAQ

**Q: 为什么要用PRNG而不是直接用list(range(200))?**

A: PRNG生成的种子更随机，避免顺序偏差。比如range(0, 50)可能都是简单场景，而range(150, 200)都是复杂场景。

**Q: 验证集和测试集可以用相同的种子吗？**

A: 绝对不行！这样会导致你在验证集上调优的模型，在测试集上得到虚高的分数。

**Q: 训练时可以偶尔在测试集上看看效果吗？**

A: 不建议！如果你根据测试集表现调整模型，那测试集就变成了验证集。

**Q: 我已经训练好了模型，现在才发现种子有问题，怎么办？**

A: 用新的测试集（不同的种子）重新评估。报告时说明这是独立的测试集。

---

**Remember: Proper data splitting is crucial for honest, reproducible research!**
