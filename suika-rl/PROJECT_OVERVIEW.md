# Suika-RL: Complete Project Overview

## 📁 Directory Structure

```
suika-rl/
│
├── algorithms/              # 所有强化学习算法实现
│   ├── dqn/                # Deep Q-Network
│   │   ├── DQN.py         # DQN实现 (183.9分)
│   │   └── __init__.py
│   │
│   ├── mcts_basic/         # 基础MCTS
│   │   ├── MCTS.py        # 原始MCTS + Lookahead
│   │   └── __init__.py
│   │
│   ├── mcts_optimized/     # 优化MCTS (32倍提速)
│   │   └── (待迁移)       # 255分，最高性能
│   │
│   ├── mcts_smart/         # 智能MCTS (启发式)
│   │   └── (待迁移)       # 177.3分，可解释
│   │
│   └── alphazero/          # AlphaZero + Lookahead
│       ├── AlphaZeroMCTS.py
│       ├── SelfPlay.py
│       └── __init__.py
│
├── models/                  # 神经网络模型定义
│   ├── SuikaNet.py         # Policy-Value网络
│   ├── StateConverter.py   # 状态转换器
│   └── __init__.py
│
├── weights/                 # 训练好的模型权重
│   ├── dqn/
│   │   ├── best_model.pdparams      # DQN最佳模型 (183.9分)
│   │   └── checkpoints/             # 各训练阶段
│   │       ├── checkpoint_ep500.pdparams
│   │       ├── checkpoint_ep1000.pdparams
│   │       └── ... (到5000)
│   │
│   ├── alphazero/
│   │   ├── iter_1.pdparams
│   │   ├── ...
│   │   ├── iter_7.pdparams          # 旧reward系统
│   │   ├── iter_8.pdparams          # 新lookahead (训练中)
│   │   └── history.json
│   │
│   └── mcts/
│       └── README.md                # MCTS不需要权重
│
├── training/                # 训练和测试脚本
│   ├── train_alphazero.py          # AlphaZero训练主程序
│   ├── test_dqn_performance.py     # DQN性能测试
│   ├── evaluate.py                 # 通用评估脚本
│   ├── generate_results.py         # 生成对比图表 ✅
│   └── __init__.py
│
├── results/                 # 实验结果和可视化 ⭐
│   ├── figures/
│   │   ├── score_comparison.png    # 分数对比柱状图
│   │   ├── speed_vs_quality.png    # 速度vs质量散点图
│   │   ├── training_cost.png       # 训练成本对比
│   │   └── score_distribution.png  # 分数分布直方图
│   │
│   ├── data/
│   │   ├── comparison.csv          # 对比数据CSV
│   │   └── comparison.json         # 对比数据JSON
│   │
│   ├── SUMMARY.md                  # 结果总结报告
│   └── README.md
│
├── env/                     # 游戏环境
│   ├── Game.py             # 核心游戏逻辑
│   ├── GameInterface.py    # RL接口封装
│   ├── PRNG.py            # 伪随机数生成
│   └── __init__.py
│
├── docs/                    # 完整文档 📖
│   ├── COMPLETE_TRAINING_HISTORY.md    # 完整训练历史
│   ├── LOOKAHEAD_REWARD_UPDATE.md      # Lookahead系统说明
│   ├── CODE_REVIEW_MCTS.md             # MCTS代码审查
│   └── TRAINING_PROCESS_EXPLAINED.md   # AlphaZero训练流程
│
├── README.md                # 项目主README
├── PROJECT_OVERVIEW.md      # 本文档
├── run_tests.sh            # 快速测试脚本
└── __init__.py
```

---

## 🎯 算法性能总结

| 算法 | 平均分 | 标准差 | 最高分 | 速度 | 训练成本 | 状态 |
|-----|-------|-------|-------|------|---------|------|
| **Optimized MCTS** | **255** | ±60 | 350+ | 0.17s | None | ✅ 可用 |
| **DQN** | **183.9** | ±66.4 | 325 | <0.01s | 5000局 | ✅ 可用 |
| **Smart MCTS** | 177.3 | ±26 | 197 | 0.43s | None | ✅ 可用 |
| **AlphaZero (新)** | TBD | - | - | ~1s | 进行中 | 🚀 训练中 |
| AlphaZero (旧) | 96.8 | ±9.3 | 109 | ~1s | 7轮 | ❌ 已弃用 |
| Random | 133.5 | ±40.3 | 243 | 0.001s | None | ✅ Baseline |

**当前冠军:** 🏆 Optimized MCTS (255分，无需训练)

---

## 🚀 快速开始

### 1. 测试所有算法
```bash
cd suika-rl
bash run_tests.sh
```

### 2. 运行DQN (最快)
```bash
cd training
python test_dqn_performance.py
```

### 3. 生成对比结果
```bash
cd training
python generate_results.py
```

查看结果：`results/figures/` 和 `results/SUMMARY.md`

### 4. 继续AlphaZero训练
```bash
cd training
python train_alphazero.py
# (已在后台运行，当前iter 8)
```

---

## 📊 关键文件说明

### 算法实现

**DQN (algorithms/dqn/DQN.py)**
- 3层MLP (640→64→64→16)
- Experience Replay (50K buffer)
- Target Network (每200步更新)
- ε-greedy (0.5→0)
- 优势：速度极快，稳定
- 劣势：需要大量训练

**MCTS (algorithms/mcts_basic/MCTS.py)**
- SimplifiedGameState (16×16网格)
- simulate_lookahead() - 新增10步前瞻
- PUCT选择
- 优势：无需训练，可解释
- 劣势：速度较慢

**AlphaZero (algorithms/alphazero/)**
- AlphaZeroMCTS.py - MCTS搜索
- SelfPlay.py - 自我对弈
- evaluate_with_lookahead() - 混合评估
- 优势：自我学习，持续进化
- 劣势：训练成本高

### 神经网络

**SuikaNet (models/SuikaNet.py)**
```
Input: [13, 16, 16] (状态特征)
  ↓ Conv2d + BatchNorm + ReLU
  ↓ Residual Blocks
  ├─→ Policy Head → [16] (动作概率)
  └─→ Value Head → [1] (状态价值)
```

### 权重文件

**DQN Weights:**
- `weights/dqn/best_model.pdparams` - 最佳模型 (197KB)
- `weights/dqn/checkpoints/` - 训练过程快照

**AlphaZero Weights:**
- `weights/alphazero/iter_7.pdparams` - 旧版本
- `weights/alphazero/iter_8.pdparams` - 新版本 (生成中)

---

## 📈 实验结果

### 已生成的可视化

1. **score_comparison.png** - 所有算法得分对比
   - 柱状图，带误差棒
   - 清晰显示：Optimized MCTS > DQN > Smart MCTS

2. **speed_vs_quality.png** - 速度vs质量权衡
   - 散点图
   - 理想区域：右上角（高分+快速）
   - DQN接近理想，Optimized MCTS质量最高

3. **training_cost.png** - 训练成本对比
   - 分组柱状图
   - 左：需训练 (DQN, AlphaZero)
   - 右：零训练 (MCTS系列)

4. **score_distribution.png** - 分数分布
   - 重叠直方图
   - 显示各算法得分范围

### 数据文件

- **comparison.csv** - 表格数据，适合Excel
- **comparison.json** - 结构化数据，适合程序读取
- **SUMMARY.md** - 文字总结报告

---

## 🔧 开发指南

### 添加新算法

1. 在 `algorithms/` 下创建新目录
2. 实现算法类
3. 创建 `__init__.py` 导出接口
4. 在 `training/` 添加测试脚本
5. 更新 `generate_results.py` 添加对比

### 修改网络结构

1. 编辑 `models/SuikaNet.py`
2. 调整通道数/层数
3. 重新训练并对比

### 重新训练

**DQN:**
```bash
cd algorithms/dqn
python DQN.py
```

**AlphaZero:**
```bash
cd training
python train_alphazero.py
```

---

## 📖 文档索引

### 训练相关
- `docs/COMPLETE_TRAINING_HISTORY.md` - 完整训练历史和Git记录
- `docs/TRAINING_PROCESS_EXPLAINED.md` - AlphaZero训练详解

### 技术说明
- `docs/LOOKAHEAD_REWARD_UPDATE.md` - Lookahead reward系统
- `docs/CODE_REVIEW_MCTS.md` - MCTS代码全面审查

### 结果报告
- `results/SUMMARY.md` - 性能对比总结
- `results/README.md` - 结果目录说明

---

## 🎓 研究价值

本项目适合用于：

1. **算法对比研究**
   - DQN vs MCTS vs AlphaZero
   - 有训练 vs 无训练
   - 速度 vs 质量权衡

2. **教学演示**
   - RL算法实践案例
   - 神经网络应用
   - MCTS搜索原理

3. **论文实验**
   - 完整的实验数据
   - 可视化图表
   - 可复现的结果

4. **进一步研究方向**
   - 更大的神经网络
   - 并行化MCTS
   - 混合算法 (DQN + MCTS)

---

## ⚙️ 环境要求

```bash
# Python 3.11+
paddlepaddle==3.2.1
numpy==1.26.4
opencv-python==4.11.0.86
matplotlib==3.7.2
pymunk==6.5.0
```

安装：
```bash
pip install -r requirements.txt
```

---

## 📝 引用

如果使用本项目，请引用：
```
Suika-RL: A Comprehensive Comparison of Reinforcement Learning Algorithms for Suika Game
https://github.com/RedContritio/DQN_FruitMerger
```

---

## 🤝 贡献

欢迎：
- Bug修复
- 新算法实现
- 性能优化
- 文档改进

---

## 📜 许可证

MIT License

---

**最后更新:** 2025-11-24
**项目状态:** ✅ 主要功能完成，AlphaZero训练进行中
**维护者:** Claude Code Assistant & User
