# DQN_FruitMerger 项目结构

生成时间: 2025-11-25

## 📁 核心目录结构

```
DQN_FruitMerger/
├── 📂 suika-rl/                    # 整理后的RL算法库
│   ├── algorithms/                 # 所有算法实现
│   │   ├── dqn/                   # MLP-DQN (PaddlePaddle)
│   │   ├── cnn_dqn/               # CNN-DQN (PyTorch) ⭐NEW
│   │   ├── mcts_basic/            # 基础MCTS
│   │   ├── mcts_smart/            # 智能MCTS (启发式)
│   │   ├── mcts_optimized/        # 优化MCTS (速度优化)
│   │   └── alphazero/             # AlphaZero
│   │
│   ├── weights/                    # 训练好的模型
│   │   ├── dqn/
│   │   ├── cnn_dqn/
│   │   ├── alphazero/
│   │   └── mcts/
│   │
│   ├── results/                    # 测试结果汇总
│   │   ├── data/
│   │   │   ├── comparison.json    # 算法对比数据
│   │   │   └── comparison.csv
│   │   └── figures/               # 可视化图表
│   │
│   ├── docs/                       # 文档
│   │   ├── CNN_DQN_REPORT.md      # CNN-DQN详细报告
│   │   └── TRAINING_HISTORY.md
│   │
│   └── training/                   # 训练脚本
│
├── 📂 mcts/                        # MCTS原始实现
│   ├── MCTS.py                    # 基础版（正确merge）
│   ├── MCTS_optimized.py          # 优化版（简化merge）
│   └── MCTS_advanced.py           # 高级版（智能启发式）
│
├── 📂 weights_cnn_dqn/             # CNN-DQN训练权重
│   ├── best_model.pth             # 最佳模型 (ep1600)
│   ├── checkpoint_ep500.pth
│   ├── checkpoint_ep1000.pth
│   ├── checkpoint_ep1500.pth
│   └── checkpoint_ep2000.pth
│
├── 📄 核心文件
│   ├── GameInterface.py           # 游戏环境接口
│   ├── CNN_DQN.py                 # CNN-DQN训练脚本
│   │
│   ├── test_optimized_mcts.py     # Optimized MCTS测试 ⭐NEW
│   ├── test_cnn_final.py          # CNN-DQN测试
│   └── test_mcts_*.py             # 其他MCTS测试脚本
│
└── 📊 测试日志/结果
    ├── optimized_mcts_test.log          # Optimized MCTS详细日志 ⭐NEW
    ├── optimized_mcts_test_results.txt  # Optimized MCTS结果 ⭐NEW
    │
    ├── cnn_dqn_full_training.log        # CNN-DQN训练日志
    ├── cnn_final_test.log               # CNN-DQN测试日志
    │
    └── evaluation_results.txt           # DQN vs Random对比
```

---

## 🎯 测试结果位置

### 1. CNN-DQN (205.7分) ✅ 完整
**训练日志**: `cnn_dqn_full_training.log`
- 2000 episodes完整训练记录
- 每100局validation评估
- 最终test set结果: 205.7 ± 51.1

**模型权重**: `weights_cnn_dqn/best_model.pth` (episode 1600)

**报告**: `suika-rl/docs/CNN_DQN_REPORT.md`

---

### 2. Optimized MCTS (152.4分) ✅ **完成 - 低于预期**
**测试日志**: `optimized_mcts_test.log` ✅
**结果文件**: `optimized_mcts_test_results.txt` ✅
- 测试集: seeds 1000-1099 (100局)
- Simulations: 2000/move
- **重要发现**: 简化merge规则严重影响性能，实际分数远低于预期的255分

---

### 3. DQN (MLP) (183.9分) ✅ 有数据
**详细日志**: `evaluation_results.txt`
- 100局完整测试记录
- 每局得分、步数、时间

**汇总数据**: `suika-rl/results/data/comparison.json`

---

### 4. 其他算法
**汇总**: `suika-rl/results/data/comparison.json`
包含所有算法统计：
- Smart MCTS: 177.3 ± 26.0
- Random: 133.5 ± 40.3
- AlphaZero: 96.8 ± 9.3

❌ **没有详细逐局日志**

---

## 📊 关键对比文件

| 文件 | 内容 |
|------|------|
| `suika-rl/results/SUMMARY.md` | 算法排名和推荐 |
| `suika-rl/results/data/comparison.json` | 完整对比数据 |
| `suika-rl/results/data/comparison.csv` | CSV格式 |
| `suika-rl/docs/CNN_DQN_REPORT.md` | CNN-DQN详细分析 |

---

## 🔧 主要训练脚本

| 脚本 | 功能 | 状态 |
|------|------|------|
| `CNN_DQN.py` | CNN-DQN训练 | ✅ 完成 |
| `test_optimized_mcts.py` | Optimized MCTS测试 | ⏳ 运行中 |
| `test_cnn_final.py` | CNN-DQN评估 | ✅ 完成 |
| `test_mcts_basic.py` | Basic MCTS测试 | ❌ 太慢未完成 |

---

## 📝 文档位置

**算法报告**:
- `suika-rl/docs/CNN_DQN_REPORT.md` - CNN-DQN完整分析

**训练记录**:
- `suika-rl/docs/TRAINING_HISTORY.md` - 所有训练历史

**项目说明**:
- `README.md` - 项目概述
- `suika-rl/results/README.md` - 结果说明

---

## 🗑️ 可清理的文件

以下是重复/临时文件，可以删除：

```
cnn_dqn_training.log          # 旧版训练日志
cnn_dqn_v2_training.log       # 中间版本
cnn_dqn_training_old.log      # 旧版本
quick_test.log                # 临时测试
training.log                  # 通用日志
mcts_basic_test.log           # 空文件，未完成
```

**保留**:
- `cnn_dqn_full_training.log` - 最终完整训练
- `cnn_final_test.log` - 最终测试
- `optimized_mcts_test.log` - 新生成的完整测试

---

## 🎯 快速导航

**想看CNN-DQN训练过程**:
```bash
less cnn_dqn_full_training.log
```

**想看CNN-DQN最终测试**:
```bash
tail -50 cnn_dqn_full_training.log
```

**想看所有算法对比**:
```bash
cat suika-rl/results/SUMMARY.md
```

**想看Optimized MCTS测试进度**:
```bash
tail -f optimized_mcts_test.log  # 实时查看
```

---

## 📈 测试进度

- [x] CNN-DQN: **205.7 ± 51.1** (100局完整测试) 🥇 **第一名**
- [x] DQN (MLP): **183.9 ± 66.4** (100局完整测试) 🥈
- [x] Optimized MCTS: **152.4 ± 53.5** (100局完整测试) 🥉 *低于预期*
- [x] Random: 133.5 ± 40.3 (100局)
- [ ] Basic MCTS: 未测试（太慢）

---

## 🔄 更新日志

- 2025-11-25 23:26: ✅ **Optimized MCTS测试完成** - 152.4分（远低于预期255分）
- 2025-11-25 23:30: 📄 创建完整技术报告 (TECHNICAL_REPORT.md) 和结果总结 (RESULTS_SUMMARY.md)
- 2025-11-25 15:17: 启动Optimized MCTS完整测试
- 2025-11-25: 完成CNN-DQN训练和测试 (205.7分)
- 2025-11-24: 整理项目结构
