# AlphaZero for Suika Game (水果合成游戏)

## 📖 项目简介

本项目将 **AlphaZero 框架**迁移到 **Suika Game (水果合成游戏)**，实现了：

- ✅ **神经网络** - CNN架构，输出 Policy P(a|s) 和 Value V(s)
- ✅ **MCTS搜索** - PUCT算法，网络驱动的蒙特卡洛树搜索
- ✅ **Self-Play** - 自我博弈收集训练数据
- ✅ **训练循环** - 迭代优化网络策略
- ✅ **评估对比** - 与DQN、随机、启发式MCTS对比

---

## 🏗️ 项目结构

```
DQN_FruitMerger/
├── Core Modules (新增)
│   ├── SuikaNet.py              # 神经网络 (Policy + Value)
│   ├── AlphaZeroMCTS.py         # AlphaZero MCTS搜索
│   ├── StateConverter.py        # 状态转换工具
│   ├── SelfPlay.py              # 自我博弈模块
│   ├── TrainAlphaZero.py        # 训练主循环
│   └── CompareAgents.py         # 评估对比脚本
│
├── Original Modules (保留)
│   ├── Game.py                  # 游戏物理引擎
│   ├── GameInterface.py         # RL接口
│   ├── DQN.py                   # DQN agent
│   └── mcts/MCTS.py             # 启发式MCTS
│
├── Deployment
│   ├── requirements_alphazero.txt  # 依赖包
│   ├── train_cloud.sh              # 云端训练脚本
│   └── README_ALPHAZERO.md         # 本文档
│
└── Outputs
    ├── weights/alphazero/       # AlphaZero模型
    ├── logs/                    # 训练日志
    └── evaluation_results.json  # 评估结果
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境 (推荐)
conda create -n alphazero python=3.8
conda activate alphazero

# 安装依赖
pip install -r requirements_alphazero.txt

# GPU版本 (如果有GPU)
# pip install paddlepaddle-gpu
```

### 2. 测试网络

```bash
# 测试SuikaNet网络
python SuikaNet.py

# 期望输出：
# [SuikaNet] Initialized: ...
# All tests passed!
```

### 3. 快速训练 (本地测试)

```bash
# 小规模训练测试 (2轮迭代)
python TrainAlphaZero.py \
    --iterations 2 \
    --games 10 \
    --simulations 50 \
    --batch-size 16 \
    --epochs 3

# 训练约10-20分钟完成
```

---

## ☁️ 云端训练 (推荐配置)

### 服务器要求

- **CPU**: 4核以上
- **内存**: 8GB以上
- **GPU**: 可选，但推荐 (加速网络训练)
- **磁盘**: 5GB以上

### 训练命令

```bash
# 方式1: 使用脚本 (推荐)
./train_cloud.sh 20 50 200 32 5 10

# 参数说明:
# 20  - 迭代次数
# 50  - 每轮游戏数
# 200 - MCTS模拟次数
# 32  - 批量大小
# 5   - 每轮epoch数
# 10  - 评估游戏数

# 方式2: 直接运行
python TrainAlphaZero.py \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    --batch-size 32 \
    --epochs 5 \
    --eval-games 10
```

### 后台运行

```bash
# 使用 nohup 后台运行
nohup ./train_cloud.sh 20 50 200 32 5 10 > train.log 2>&1 &

# 查看进度
tail -f train.log

# 或者使用 tmux/screen
tmux new -s train
./train_cloud.sh 20 50 200 32 5 10
# Ctrl+B, D 分离会话
```

### 监控训练

```bash
# 查看GPU使用 (如果有GPU)
nvidia-smi

# 查看训练历史
cat weights/alphazero/history.json | python -m json.tool

# 实时监控日志
tail -f logs/train_*.log
```

---

## 📊 评估与对比

### 评估单个模型

```bash
python CompareAgents.py \
    --num-games 50 \
    --alphazero-model weights/alphazero/iter_20.pdparams \
    --alphazero-sims 200
```

### 完整对比 (AlphaZero vs DQN vs Random vs MCTS)

```bash
python CompareAgents.py \
    --num-games 50 \
    --alphazero-model weights/alphazero/iter_20.pdparams \
    --dqn-model weights/final.pdparams \
    --alphazero-sims 200 \
    --mcts-sims 200 \
    --output evaluation_results.json
```

### 预期结果

| Agent | Mean Score | Max Score | 说明 |
|-------|------------|-----------|------|
| Random | 150 | 300 | 基线 |
| DQN | 500 | 1200 | 经验回放学习 |
| MCTS Baseline | 800 | 2000 | 启发式搜索 |
| **AlphaZero** | **1200+** | **3500+** | 网络+MCTS |

---

## 🔧 高级配置

### 调整训练参数

**加快训练速度:**
```python
# 减少模拟次数
--simulations 100  # 默认200

# 减少每轮游戏数
--games 30  # 默认50

# 减少网络复杂度
# 修改 SuikaNet.py:
hidden_channels=32  # 默认64
```

**提高最终性能:**
```python
# 增加模拟次数
--simulations 400

# 增加每轮游戏数
--games 100

# 更多训练epoch
--epochs 10

# 更多迭代
--iterations 50
```

### 恢复训练

```bash
# 从第10轮继续
python TrainAlphaZero.py \
    --resume 10 \
    --iterations 20 \
    --checkpoint-dir weights/alphazero
```

---

## 📈 训练流程详解

### 单轮迭代

```
迭代 i:
├─ [1/3] Self-Play (30-60分钟)
│   ├─ 玩 50 局游戏
│   ├─ 每步用MCTS(200次模拟)选择动作
│   └─ 收集 (s, π, z) 训练数据
│
├─ [2/3] Train (5-10分钟)
│   ├─ 5 个 epoch
│   ├─ Loss = MSE(V, z) + CrossEntropy(P, π)
│   └─ 更新网络参数
│
└─ [3/3] Evaluate (10-15分钟)
    ├─ 测试 10 局游戏
    ├─ 计算平均得分
    └─ 保存检查点
```

### 完整训练时间估算

| 配置 | 单轮时间 | 20轮总时间 |
|------|---------|-----------|
| 快速 (sim=100, games=30) | ~30分钟 | ~10小时 |
| 标准 (sim=200, games=50) | ~60分钟 | ~20小时 |
| 高质量 (sim=400, games=100) | ~120分钟 | ~40小时 |

---

## 🧪 核心技术解析

### 1. 神经网络设计

```python
Input: [13, 20, 16]
  ├─ 0-10: 水果等级 (one-hot)
  ├─ 11: 当前水果类型
  └─ 12: 高度信息

Network:
  Conv2D(13→64) → BN → ReLU
  Conv2D(64→64) → BN → ReLU
  Conv2D(64→64) → BN → ReLU
  ├─ Policy Head → [16] 动作概率
  └─ Value Head → [1] 状态价值
```

### 2. MCTS + PUCT

```python
# Selection
UCB(s,a) = Q(s,a) + c × P(a|s) × √N(s) / (1 + N(s,a))

# Expansion
用网络 (P, V) = f(s) 评估叶子节点

# Backup
反向传播 V 到路径上所有节点
```

### 3. 训练Loss

```python
Loss = MSE(V(s), z) + CrossEntropy(P(s), π) + L2_reg

where:
  V(s) - 网络价值输出
  z - 最终得分（归一化）
  P(s) - 网络策略输出
  π - MCTS搜索得到的增强策略
```

---

## 🐛 常见问题

### Q1: 训练很慢怎么办?

**A:**
- 减少`--simulations`到100
- 减少`--games`到30
- 使用GPU版本PaddlePaddle
- 关闭可视化渲染

### Q2: 内存不足?

**A:**
- 减少`--batch-size`到16
- 减少`hidden_channels`到32
- 限制历史数据buffer大小

### Q3: 得分没有提升?

**A:**
- 检查网络是否收敛 (loss下降)
- 增加训练迭代次数
- 增加MCTS模拟次数
- 检查状态转换是否正确

### Q4: 如何可视化训练过程?

**A:**
```python
# 读取历史
import json
with open('weights/alphazero/history.json') as f:
    history = json.load(f)

# 绘图
import matplotlib.pyplot as plt
plt.plot(history['iterations'], history['eval_scores'])
plt.xlabel('Iteration')
plt.ylabel('Evaluation Score')
plt.show()
```

---

## 📚 论文参考

1. **AlphaGo Zero** - Silver et al., Nature 2017
   - PUCT算法
   - Self-play训练

2. **AlphaZero** - Silver et al., Science 2018
   - 通用框架
   - 单一网络架构

3. **MuZero** - Schrittwieser et al., Nature 2020
   - 模型学习
   - Planning in latent space

---

## 🤝 贡献与扩展

### 可能的改进方向

1. **网络结构**
   - ResNet残差连接
   - Attention机制
   - 更深的网络

2. **MCTS优化**
   - Virtual loss (并行搜索)
   - RAVE (快速行动值估计)
   - Progressive widening改进

3. **训练技巧**
   - 优先级经验回放
   - 课程学习
   - 对抗训练

4. **工程优化**
   - 分布式训练
   - 混合精度训练
   - 模型压缩

---

## 📧 联系方式

如有问题或建议，欢迎：
- 提Issue
- Pull Request
- 邮件联系

---

## ⚖️ 开源协议

本项目基于原DQN_FruitMerger项目扩展，遵循相同的开源协议。

---

## 🎉 致谢

- PaddlePaddle深度学习框架
- AlphaGo/AlphaZero团队
- Pymunk物理引擎
- 原DQN_FruitMerger项目

---

**祝训练顺利！🚀**
