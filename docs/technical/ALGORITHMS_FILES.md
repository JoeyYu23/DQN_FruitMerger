# 三种算法使用的文件总结

**生成时间:** 2025-12-03

---

## 🎮 **核心游戏环境文件** (所有算法都需要)

### 必需文件 (5个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `Game.py` | 17K | 游戏核心逻辑（pymunk物理引擎） |
| `GameInterface.py` | 5.1K | 环境接口（state/reward/action） |
| `GameEvent.py` | 1.1K | 游戏事件处理 |
| `PRNG.py` | 1.0K | 伪随机数生成器（可复现的种子） |
| `render_utils.py` | 3.5K | 渲染工具 |

### 资源文件夹
- `resources/images/` - 水果图片素材
- `resources/illustrations/` - 插图资源

---

## 1️⃣ **DQN (MLP-DQN with PaddlePaddle)**

### 核心文件 (4个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `DQN.py` | 9.8K | **主文件** - MLP-DQN实现 |
| `SuikaNet.py` | 11K | 神经网络定义（MLP架构） |
| `StateConverter.py` | 7.6K | 状态转换器（可选） |
| `evaluate.py` | 1.7K | 标准评估脚本 |

### 训练脚本
- `train_5000.py` (12K) - 5000 episodes训练
- `train_with_logging.py` (7.6K) - 带日志的训练

### 权重文件
```
weights/
├── best_model.pdparams          # 最佳模型
├── checkpoint_ep500.pdparams
├── checkpoint_ep1000.pdparams
├── ...
└── checkpoint_ep5000.pdparams
```

根目录也有：
- `final.pdparams` (197K) - 最终训练权重
- `final_5000.pdparams` (197K) - 5000轮训练权重

### 依赖
```python
import paddle
from GameInterface import GameInterface
from PRNG import PRNG
```

### 测试/演示
- `test_dqn_performance.py` (3.1K) - 性能测试
- `AIPlay.py` (2.4K) - AI游戏演示
- `AIPlay_Auto.py` (6.1K) - 自动AI演示

---

## 2️⃣ **CNN-DQN (PyTorch)**

### 核心文件 (3个)
| 文件 | 大小 | 说明 |
|------|------|------|
| `CNN_DQN.py` | 19K | **主文件** - CNN-DQN实现（完整） |
| `SuikaNet_torch.py` | 7.9K | PyTorch神经网络（可选，CNN_DQN.py已包含） |
| `test_cnn_final.py` | 1.4K | 最终测试脚本 |

### 权重文件
```
weights_cnn_dqn/
├── best_model.pth              # 🏆 最佳模型 (ep1600, 205.7分)
├── final_model.pth             # 最终模型 (ep2000)
├── checkpoint_ep500.pth
├── checkpoint_ep1000.pth
├── checkpoint_ep1500.pth
└── checkpoint_ep2000.pth
```

### 依赖
```python
import torch
import torch.nn as nn
from GameInterface import GameInterface
```

### 特点
- **CNN架构**：保留空间结构
- **输入格式**：(2, 20, 16) - 2通道20x16网格
- **更高性能**：205.7 ± 51.1 (vs DQN的183.9)
- **更少训练**：1600 episodes (vs DQN的5000)

---

## 3️⃣ **MCTS (Real Physics)**

### 核心文件 (主目录)
| 文件 | 大小 | 说明 |
|------|------|------|
| `mcts/MCTS_real_physics.py` | 21K | **推荐** - 真实物理MCTS |
| `mcts/MCTS_optimized.py` | 11K | 优化版MCTS（简化物理） |
| `mcts/MCTS_tuned.py` | 16K | 调优版MCTS |
| `mcts/MCTS_advanced.py` | 17K | 高级版MCTS（启发式） |
| `mcts/MCTS.py` | 24K | 基础版MCTS（正确merge但慢） |

### 测试文件
- `test_real_physics_mcts.py` (5.8K) - Real Physics MCTS测试
- `evaluate_mcts_real_physics.py` (3.9K) - MCTS评估

### 依赖
```python
import numpy as np
from Game import FRUIT_RADIUS
from GameInterface import GameInterface
```

### 特点
- **无需训练**：纯搜索算法
- **使用真实物理引擎**：完整pymunk模拟
- **两步前瞻**：每步评估当前+未来动作
- **智能奖励**：合并奖励 + 位置优势 - 高度惩罚

### MCTS版本对比
| 版本 | 说明 | 速度 | 准确性 |
|------|------|------|--------|
| `MCTS_real_physics.py` | 真实物理引擎 | 中 | ⭐⭐⭐⭐⭐ |
| `MCTS_optimized.py` | 简化物理（网格） | 快 | ⭐⭐⭐ |
| `MCTS.py` | 基础版（正确但慢） | 慢 | ⭐⭐⭐⭐⭐ |
| `MCTS_advanced.py` | 启发式增强 | 中 | ⭐⭐⭐⭐ |

---

## 📁 **辅助文件/工具**

### 可视化/分析
- `test_model_visual.py` (9.6K) - 模型决策可视化
- `analyze_high_score.py` (9.7K) - 高分游戏分析
- `CompareAgents.py` (11K) - 多算法对比
- `benchmark_all.py` (13K) - 完整benchmark

### 脚本工具 (scripts/)
- `scripts/run_mcts.py` - 运行MCTS
- `scripts/record_top_games.py` - 录制高分游戏
- `scripts/compare_mcts_versions.py` - 对比MCTS版本

### 其他游戏模式
- `InteractivePlay.py` (1.0K) - 人类玩
- `RandomPlay.py` (4.7K) - 随机玩
- `SelfPlay.py` (7.7K) - 自我对弈

---

## 📦 **suika-rl 子项目** (可选，独立版本)

```
suika-rl/
├── algorithms/
│   ├── dqn/              # PaddlePaddle DQN (完整实现)
│   ├── cnn_dqn/          # PyTorch CNN-DQN (完整实现)
│   ├── mcts_basic/       # 基础MCTS
│   ├── mcts_optimized/   # 优化MCTS
│   └── mcts_smart/       # 智能MCTS
│
├── weights/              # 各算法权重
│   ├── dqn/
│   ├── cnn_dqn/
│   └── mcts/
│
├── results/              # 结果汇总
│   ├── data/comparison.json
│   └── figures/
│
└── training/             # 训练脚本
```

**注意**：`suika-rl/` 是项目的整理版本，包含完整算法实现。主目录的文件是原始开发版本。**两者功能相同，可以任选一个使用。**

---

## 🎯 **快速开始指南**

### 运行 DQN
```bash
# 使用训练好的模型
python AIPlay.py  # 使用final.pdparams

# 评估性能
python evaluate.py  # 200次测试
```

### 运行 CNN-DQN
```bash
# 使用最佳模型
python test_cnn_final.py

# 或者修改test_cnn_final.py使用best_model.pth
```

### 运行 MCTS
```bash
# Real Physics MCTS
python test_real_physics_mcts.py --seed 888 --sims 50 --steps 100

# 对比不同MCTS版本
python test_real_physics_mcts.py --compare
```

---

## 📊 **性能对比**

| 算法 | 平均分 | 训练成本 | 推理速度 | 核心文件数 |
|------|--------|----------|----------|-----------|
| **CNN-DQN** | 205.7 | 1600 ep | 0.01s/step | 3 |
| **DQN** | 183.9 | 5000 ep | 0.01s/step | 4 |
| **MCTS (Real)** | 231.92 | 无 | 1.0s/step | 1 |

---

## 🗂️ **文件最小集合**

### 运行 DQN 最少需要:
```
Game.py
GameInterface.py
GameEvent.py
PRNG.py
DQN.py
weights/best_model.pdparams
resources/images/
```

### 运行 CNN-DQN 最少需要:
```
Game.py
GameInterface.py
GameEvent.py
CNN_DQN.py
weights_cnn_dqn/best_model.pth
resources/images/
```

### 运行 MCTS 最少需要:
```
Game.py
GameInterface.py
GameEvent.py
mcts/MCTS_real_physics.py
resources/images/
```

---

## 📝 **依赖包**

### DQN (PaddlePaddle)
```
paddlepaddle
pymunk
numpy
opencv-python
```

### CNN-DQN (PyTorch)
```
torch
pymunk
numpy
opencv-python
```

### MCTS (无机器学习)
```
pymunk
numpy
opencv-python
```

---

**总结:**
- **最简单**: MCTS (无需训练，1个文件)
- **最快速**: CNN-DQN/DQN (0.01s/step)
- **最准确**: CNN-DQN (205.7分)
- **最灵活**: MCTS (可调参数多)
