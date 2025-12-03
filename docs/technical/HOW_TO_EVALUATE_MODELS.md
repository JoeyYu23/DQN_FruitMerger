# 如何评估不同模型 - 完整指南

## 🎯 评估模型的完整流程

### 1️⃣ **核心评估原理**

评估的关键是：**让所有模型在完全相同的条件下玩游戏，然后比较表现**

```python
# 核心评估循环
for seed in seeds:  # 每个模型用相同的种子序列
    env.reset(seed=seed)  # 重置环境，固定随机性

    while game_not_over:
        action = agent.predict(state)  # 模型做决策
        state, reward, done = env.step(action)  # 执行动作

    record_score(env.score)  # 记录最终得分
```

**为什么用相同的种子？**
- 种子决定了水果掉落序列
- 相同种子 = 完全相同的游戏场景
- 确保公平对比（不是运气好坏，而是策略优劣）

### 2️⃣ **现有的三种评估方式**

#### **方式A：单模型快速评估** (`evaluate.py`)

```bash
python evaluate.py
```

**特点：**
- 评估200局
- DQN vs Random对比
- 输出简单统计（均值、最大值、最小值）

**适用场景：** 快速检查模型是否工作

---

#### **方式B：详细单模型评估** (`evaluate_multi_games.py`)

```bash
python evaluate_multi_games.py
```

**特点：**
- 评估100局
- 详细的统计分析（均值、标准差、中位数）
- 分数分布直方图
- 胜率统计
- 保存详细结果到文件

**输出示例：**
```
📊 DQN Agent 统计结果
======================================================================
🎯 分数统计:
  平均分数: 245.30
  最高分数: 368
  最低分数: 156
  标准差:   42.15
  中位数:   238.50

📈 分数分布:
  0  -100:  0局 (  0.0%)
  100-150: 12局 ( 12.0%) ████████
  150-200: 28局 ( 28.0%) ██████████████
  200-250: 35局 ( 35.0%) █████████████████
  250-300: 18局 ( 18.0%) █████████
  300-400:  7局 (  7.0%) ███

⚔️  DQN vs 随机Agent 对比
  DQN胜率: 78.0% (78/100局)
```

**适用场景：** 深入分析单个模型的性能

---

#### **方式C：多模型统一对比** (`benchmark_all.py`) ⭐ 推荐

```bash
python benchmark_all.py 100  # 100局评估
```

**特点：**
- 一次性测试所有模型
- 相同的种子集合
- 全面的对比分析
- 自动生成报告（JSON + LaTeX）

**适用场景：** 对比多个模型，为论文准备数据

---

### 3️⃣ **统一评估框架详解**

让我详细说明 `benchmark_all.py` 的工作流程：

#### **Step 1: 初始化**

```python
# 设置评估参数
NUM_EPISODES = 100
seeds = [0, 1, 2, ..., 99]  # 固定种子序列

# 创建环境
env = GameInterface()
```

#### **Step 2: 评估每个模型**

```python
def evaluate_agent(agent, agent_name):
    scores = []
    times = []

    for seed in seeds:
        env.reset(seed=seed)  # 固定环境

        # 第一步随机（确保游戏开始）
        action = random_action()
        state, _, alive = env.next(action)

        episode_time = 0
        while alive:
            # 计时开始
            start = time.time()

            # 模型预测
            action = agent.predict(state)

            # 计时结束
            episode_time += time.time() - start

            # 执行动作
            state, reward, alive = env.next(action)

        # 记录结果
        scores.append(env.score)
        times.append(episode_time / num_steps)

    return {
        'scores': scores,
        'avg_time': mean(times),
        'mean_score': mean(scores),
        'std_score': std(scores),
        ...
    }
```

#### **Step 3: 对比分析**

```python
# 计算提升百分比
improvement = (dqn_mean - random_mean) / random_mean * 100

# 计算胜率
win_rate = sum(dqn_scores > mcts_scores) / len(scores)

# 效应大小（Cohen's d）
cohens_d = (mean1 - mean2) / pooled_std
```

---

### 4️⃣ **如何添加你自己的模型**

#### **场景1：你训练了一个新的DQN模型**

```python
# 1. 训练并保存模型
python quick_train.py  # 或你的训练脚本
# 这会生成 my_new_model.pdparams

# 2. 修改 benchmark_all.py
# 在 main() 函数中添加：

# Load your new model
if os.path.exists("my_new_model.pdparams"):
    my_agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    my_agent.policy_net.set_state_dict(paddle.load("my_new_model.pdparams"))

    # Evaluate it
    benchmark.results['My New DQN'] = benchmark.evaluate_agent(
        my_agent,
        'My New DQN',
        use_env=False  # DQN使用feature输入
    )

# 3. 运行对比
python benchmark_all.py 200
```

---

#### **场景2：你实现了一个新的MCTS算法**

```python
# 1. 创建你的agent类
# my_mcts.py
class MyAdvancedMCTS:
    def __init__(self, num_simulations=100):
        self.num_simulations = num_simulations

    def predict(self, env):
        # 你的MCTS逻辑
        # env包含完整的游戏状态
        best_action = self.run_mcts(env)
        return best_action  # 返回整数或[整数]

# 2. 添加到benchmark
from my_mcts import MyAdvancedMCTS

my_mcts = MyAdvancedMCTS(num_simulations=200)
benchmark.results['My Advanced MCTS'] = benchmark.evaluate_agent(
    my_mcts,
    'My Advanced MCTS',
    use_env=True  # MCTS需要完整环境
)
```

---

#### **场景3：你想测试不同的超参数**

```python
# 比如测试不同的epsilon值
for epsilon in [0.0, 0.1, 0.2, 0.3]:
    agent = Agent(build_model, feature_dim, action_dim, e_greed=epsilon)
    agent.policy_net.set_state_dict(paddle.load("final.pdparams"))

    results = benchmark.evaluate_agent(
        agent,
        f'DQN (ε={epsilon})',
        use_env=False
    )
    benchmark.results[f'DQN (ε={epsilon})'] = results
```

---

### 5️⃣ **理解评估结果**

#### **关键指标解读**

```
Agent               Mean Score    Std Dev    Max    Time/Step
------------------------------------------------------------------
DQN                 245.3 ± 42.1  42.1       368    0.012s
Smart MCTS          223.7 ± 38.5  38.5       319    0.245s
Fast MCTS           201.2 ± 35.2  35.2       287    0.089s
Random              145.8 ± 28.7  28.7       203    0.001s
```

**如何判断模型好坏？**

#### 1. **平均分数（Mean Score）**
- 主要指标
- DQN: 245.3 → 比随机高68%
- **越高越好**

#### 2. **标准差（Std Dev）**
- 衡量稳定性
- **小标准差 = 稳定**
- 大标准差 = 不稳定（运气成分大）
- 例：DQN的42.1表示分数在203-287之间波动（±1σ）

#### 3. **最大值（Max）**
- 潜力上限
- DQN能达到368分，说明策略有潜力
- 如果最大值远高于平均值，说明算法偶尔能发现好策略

#### 4. **计算时间（Time/Step）**
- 实用性考量
- DQN: 0.012秒（快，适合实时）
- Smart MCTS: 0.245秒（慢20倍）
- **需要权衡性能 vs 速度**

---

#### **胜率矩阵**

```
Win Rate Matrix:
                DQN      Smart MCTS  Fast MCTS   Random
DQN             ---      78.5%       85.2%       94.3%
Smart MCTS      21.5%    ---         68.7%       87.9%
Fast MCTS       14.8%    31.3%       ---         75.2%
Random          5.7%     12.1%       24.8%       ---
```

**解读：**
- DQN在78.5%的游戏中击败Smart MCTS
- DQN在94.3%的游戏中击败Random
- **DQN是最强的模型**

---

#### **改进百分比**

```
📈 Improvement over Random:
  DQN                : +68.3% | Win rate: 94.3% | Cohen's d:  2.51
  Smart MCTS         : +53.4% | Win rate: 87.9% | Cohen's d:  2.12
  Fast MCTS          : +38.0% | Win rate: 75.2% | Cohen's d:  1.67
```

**Cohen's d 解释：**（效应大小）
- `|d| < 0.2`: 可忽略的差异
- `0.2 ≤ |d| < 0.5`: 小效应
- `0.5 ≤ |d| < 0.8`: 中等效应
- **`|d| ≥ 0.8`: 大效应（显著更好）**

DQN的d=2.51 → **非常显著的改进！**

---

### 6️⃣ **实战示例：对比3个DQN模型**

假设你训练了3个不同配置的DQN：

**创建评估脚本：**

```python
# benchmark_my_dqns.py
from benchmark_all import BenchmarkRunner
from GameInterface import GameInterface
from DQN import Agent, build_model
import paddle

# 初始化
benchmark = BenchmarkRunner(num_episodes=200)

feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
feature_map_width = GameInterface.FEATURE_MAP_WIDTH
action_dim = GameInterface.ACTION_NUM
feature_dim = feature_map_height * feature_map_width * 2

# 模型1：原始DQN
print("Loading DQN v1 (baseline)...")
agent1 = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
agent1.policy_net.set_state_dict(paddle.load("model_v1.pdparams"))
benchmark.results['DQN v1 (baseline)'] = benchmark.evaluate_agent(
    agent1, 'DQN v1', use_env=False
)

# 模型2：改进的网络结构
print("Loading DQN v2 (deeper network)...")
agent2 = Agent(build_better_model, feature_dim, action_dim, e_greed=0.0)
agent2.policy_net.set_state_dict(paddle.load("model_v2_better_arch.pdparams"))
benchmark.results['DQN v2 (deeper net)'] = benchmark.evaluate_agent(
    agent2, 'DQN v2', use_env=False
)

# 模型3：更长的训练
print("Loading DQN v3 (longer training)...")
agent3 = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
agent3.policy_net.set_state_dict(paddle.load("model_v3_more_epochs.pdparams"))
benchmark.results['DQN v3 (longer train)'] = benchmark.evaluate_agent(
    agent3, 'DQN v3', use_env=False
)

# 对比结果
benchmark.compare_results()
benchmark.save_results("my_dqn_comparison.json")
benchmark.export_latex_table("my_dqn_table.tex")

print("\n✅ Evaluation complete!")
print("📊 Results saved to: my_dqn_comparison.json")
print("📄 LaTeX table saved to: my_dqn_table.tex")
```

**运行：**
```bash
python benchmark_my_dqns.py
```

**你会得到：**
- ✅ 详细对比表格
- ✅ 哪个模型最好
- ✅ 改进是否显著
- ✅ 适合论文的LaTeX表格

---

### 7️⃣ **评估的最佳实践**

#### ✅ **DO（应该做的）：**

##### 1. **使用足够的测试局数**
```python
# 调试阶段
NUM_EPISODES = 20

# 正式评估
NUM_EPISODES = 100

# 论文级别
NUM_EPISODES = 200-500
```

##### 2. **固定种子序列**
```python
# ✅ 好的做法
seeds = list(range(200))  # 0-199

# ❌ 不好的做法
seeds = [random.randint(0, 1000) for _ in range(200)]  # 每次不同
```

##### 3. **报告完整统计**
```python
# ✅ 完整报告
print(f"Mean: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
print(f"Median: {np.median(scores):.2f}")
print(f"Max: {np.max(scores)}, Min: {np.min(scores)}")

# ❌ 不完整报告
print(f"Mean: {np.mean(scores)}")  # 只报告均值
```

##### 4. **保存原始数据**
```python
# ✅ 保存完整数据
benchmark.save_results()  # 保存JSON

# ❌ 只保存摘要
with open("results.txt", "w") as f:
    f.write(f"Mean: {mean_score}")  # 原始数据丢失
```

##### 5. **对比基线**
```python
# ✅ 总是包含基线
benchmark.results['Random'] = evaluate(random_agent)
benchmark.results['My Model'] = evaluate(my_agent)

# ❌ 没有基线
benchmark.results['My Model'] = evaluate(my_agent)  # 无法判断好坏
```

---

#### ❌ **DON'T（不应该做的）：**

##### 1. **不要挑选种子**
```python
# ❌ 错误：只报告表现好的种子
good_seeds = [s for s in seeds if my_agent.score(s) > 200]
evaluate_on(good_seeds)

# ✅ 正确：用预先确定的种子集
seeds = list(range(100))
evaluate_on(seeds)
```

##### 2. **不要样本太小**
```python
# ❌ 样本太小，结果不可靠
NUM_EPISODES = 10

# ✅ 足够的样本
NUM_EPISODES = 100  # 最少
```

##### 3. **不要忽略方差**
```python
# ❌ 只看均值
model_a_mean = 245.0
model_b_mean = 243.0
# 结论：A更好？不一定！

# ✅ 看均值和方差
model_a: 245.0 ± 5.0   # 稳定
model_b: 243.0 ± 50.0  # 不稳定
# 结论：A虽然均值只高一点，但更稳定
```

##### 4. **不要用不同的种子对比**
```python
# ❌ 错误：不同种子
agent_a_scores = evaluate(agent_a, seeds=range(0, 100))
agent_b_scores = evaluate(agent_b, seeds=range(100, 200))

# ✅ 正确：相同种子
seeds = list(range(100))
agent_a_scores = evaluate(agent_a, seeds=seeds)
agent_b_scores = evaluate(agent_b, seeds=seeds)
```

---

### 8️⃣ **快速命令参考**

```bash
# ==================== 快速测试（适合调试） ====================
python evaluate.py
# 200局，快速检查DQN是否工作

# ==================== 详细单模型评估（适合分析） ====================
python evaluate_multi_games.py
# 100局，详细统计，分数分布

# ==================== 对比所有模型（适合论文） ====================
python benchmark_all.py 200
# 200局，所有模型，LaTeX输出

# ==================== 只对比MCTS ====================
python scripts/compare_mcts_versions.py 20 100
# 20局，每步100次模拟

# ==================== 自定义评估 ====================
python my_custom_benchmark.py
# 你自己的评估脚本
```

---

### 9️⃣ **常见问题解答**

#### Q1: 为什么需要相同的种子？

**A:** 想象两个学生考试：
- 学生A做试卷1（简单）
- 学生B做试卷2（困难）
- 他们的分数不能直接比较！

相同种子 = 相同试卷 = 公平比较

---

#### Q2: 多少局评估才够？

**A:** 取决于目的：
- **调试/快速检查**: 10-20局
- **日常评估**: 100局
- **论文/发表**: 200-500局

**经验法则**：标准误差 = σ/√n
- 100局：标准误差约为σ/10
- 400局：标准误差约为σ/20（更精确）

---

#### Q3: 胜率多少才算显著更好？

**A:** 经验标准：
- **>70%**: 明显更好
- **60-70%**: 较好
- **50-60%**: 略好
- **45-55%**: 差不多
- **<45%**: 更差

但最好用统计检验（如Mann-Whitney U test）确认。

---

#### Q4: DQN和MCTS用的接口不一样怎么办？

**A:** `benchmark_all.py` 已经处理了：
```python
# DQN：用feature
benchmark.evaluate_agent(dqn_agent, 'DQN', use_env=False)

# MCTS：用环境
benchmark.evaluate_agent(mcts_agent, 'MCTS', use_env=True)
```

内部会自动选择正确的调用方式。

---

#### Q5: 如何比较性能和速度？

**A:** 计算效率分数：
```python
efficiency = mean_score / mean_time_per_step

# 例如：
DQN:        245.3 / 0.012 = 20,442
Smart MCTS: 223.7 / 0.245 =    913

# DQN的效率是Smart MCTS的22倍！
```

---

### 🔟 **论文写作建议**

#### **Results部分应该包含：**

1. **方法描述**
   ```
   We evaluated our DQN agent against three baselines:
   random policy, FastMCTS (100 simulations), and
   SmartMCTS (100 simulations). Each agent was tested
   on 200 episodes with fixed seeds 0-199.
   ```

2. **结果表格**
   ```
   Table 1: Performance comparison (200 episodes)

   Agent          Mean Score   Std Dev   Max   Win Rate vs Random
   ----------------------------------------------------------------
   DQN            245.3±42.1   42.1      368   94.3%
   Smart MCTS     223.7±38.5   38.5      319   87.9%
   Fast MCTS      201.2±35.2   35.2      287   75.2%
   Random         145.8±28.7   28.7      203   ---
   ```

3. **统计检验**
   ```
   DQN significantly outperformed all baselines
   (Mann-Whitney U test, p < 0.001). The effect size
   compared to random baseline was large (Cohen's d = 2.51).
   ```

4. **计算成本**
   ```
   DQN achieved the highest score while being 20× faster
   than Smart MCTS (0.012s vs 0.245s per action), making
   it suitable for real-time applications.
   ```

---

### 总结

评估模型的**核心原则**：

1. ✅ **公平性** - 相同种子，相同条件
2. ✅ **统计性** - 足够样本，完整统计
3. ✅ **可重复** - 固定随机性，保存数据
4. ✅ **全面性** - 多个指标，不只看分数

**你现在拥有的工具：**

| 工具 | 用途 | 命令 |
|------|------|------|
| `evaluate.py` | 快速检查 | `python evaluate.py` |
| `evaluate_multi_games.py` | 详细分析 | `python evaluate_multi_games.py` |
| `benchmark_all.py` | 全面对比 | `python benchmark_all.py 200` |
| `compare_mcts_versions.py` | MCTS对比 | `python scripts/compare_mcts_versions.py 20 100` |

**可以评估：**
- ✅ DQN vs MCTS
- ✅ 不同超参数
- ✅ 新算法
- ✅ 生成论文表格

---

**还有问题？**
- 游戏机制: `docs/game_mechanics.tex`
- 评估方法论: `docs/evaluation_methodology.tex`
- 英文评估指南: `docs/EVALUATION_GUIDE.md`
