# 🎮 合成大西瓜 AlphaZero - 快速开始指南

## 📋 目录

1. [环境准备](#环境准备)
2. [快速测试](#快速测试)
3. [本地训练](#本地训练)
4. [云端训练](#云端训练)
5. [可视化结果](#可视化结果)
6. [评估模型](#评估模型)
7. [常见问题](#常见问题)

---

## 环境准备

### 1. 检查Python版本
```bash
python --version  # 需要 Python 3.8+
```

### 2. 安装依赖
```bash
# 进入项目目录
cd /Users/ycy/Downloads/DQN_FruitMerger

# 安装AlphaZero所需依赖
pip install -r requirements_alphazero.txt

# 如果有GPU (可选但推荐):
pip install paddlepaddle-gpu
```

### 3. 验证安装
```bash
python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"
python -c "import pymunk; print('Pymunk: OK')"
python -c "import cv2; print('OpenCV: OK')"
```

---

## 快速测试

### ⚡ 5分钟快速体验
```bash
# 快速训练测试 (2轮迭代, 约5-10分钟)
python run_training.py train --quick
```

这会运行一个简化的训练流程:
- 2轮迭代
- 每轮10局游戏
- MCTS每步50次模拟
- 快速验证环境和代码是否正常

### 查看训练历史
```bash
# 查看训练摘要
python visualize_results.py --summary-only

# 生成可视化图表
python visualize_results.py
```

---

## 本地训练

### 标准训练 (本地推荐配置)
```bash
# 适中的训练强度 (约10-20小时)
python run_training.py train \
    --iterations 10 \
    --games 30 \
    --simulations 100 \
    --batch-size 32 \
    --epochs 5 \
    --eval-games 10
```

### 完整训练 (较高性能)
```bash
# 完整训练 (约30-40小时)
python run_training.py train \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    --batch-size 32 \
    --epochs 5 \
    --eval-games 10
```

### 继续训练
```bash
# 从第10轮继续训练
python run_training.py train \
    --resume 10 \
    --iterations 20 \
    --games 50 \
    --simulations 200
```

---

## 云端训练

### 1. Google Colab

创建新的Colab笔记本:

```python
# 1. 安装依赖
!pip install paddlepaddle pymunk opencv-python matplotlib tqdm

# 2. 克隆项目 (或上传文件)
!git clone https://github.com/your-repo/DQN_FruitMerger.git
%cd DQN_FruitMerger

# 3. 运行训练
!python run_training.py train --iterations 20 --games 50 --simulations 200

# 4. 下载结果
from google.colab import files
!zip -r training_results.zip weights/alphazero/ *.png
files.download('training_results.zip')
```

### 2. AutoDL / 其他云服务器

```bash
# SSH连接服务器后

# 1. 上传项目文件
scp -r DQN_FruitMerger user@server:/workspace/

# 2. SSH登录
ssh user@server

# 3. 进入项目目录
cd /workspace/DQN_FruitMerger

# 4. 安装依赖
pip install -r requirements_alphazero.txt

# 5. 后台运行训练
nohup python run_training.py train \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    > train.log 2>&1 &

# 6. 查看训练进度
tail -f train.log

# 7. 下载结果到本地
# (在本地电脑运行)
scp -r user@server:/workspace/DQN_FruitMerger/weights/alphazero ./
scp user@server:/workspace/DQN_FruitMerger/*.png ./
```

---

## 可视化结果

### 1. 查看训练历史
```bash
# 只显示摘要
python visualize_results.py --summary-only

# 生成并显示图表
python visualize_results.py

# 生成图表但不显示
python visualize_results.py --no-show --save-path my_training.png
```

输出示例:
```
📊 训练历史统计:
   总迭代次数: 20
   最终训练Loss: 1.2345
   最终评估分数: 850.5
   最高评估分数: 920.3
   平均评估分数: 756.2
```

### 2. 图表说明

生成的图表包含4个子图:
- **左上**: 总体训练Loss变化
- **右上**: Policy Loss vs Value Loss对比
- **左下**: 评估分数进度 (最重要!)
- **右下**: 每轮迭代的分数提升

---

## 评估模型

### 1. 评估单个模型
```bash
# 基础评估
python run_training.py evaluate \
    --model-path weights/alphazero/iter_20.pdparams \
    --num-games 20

# 评估并可视化游戏过程
python run_training.py evaluate \
    --model-path weights/alphazero/iter_20.pdparams \
    --num-games 20 \
    --visualize
```

### 2. 直接使用evaluate_model.py
```bash
# 完整评估 + 可视化 + 保存视频
python evaluate_model.py \
    --model-path weights/alphazero/iter_20.pdparams \
    --num-games 30 \
    --simulations 200 \
    --visualize \
    --save-video
```

### 3. 比较多个模型
```bash
python evaluate_model.py \
    --compare \
        weights/alphazero/iter_5.pdparams \
        weights/alphazero/iter_10.pdparams \
        weights/alphazero/iter_20.pdparams \
    --num-games 20
```

### 4. 评估输出示例
```
==============================================================
  AlphaZero 模型评估
==============================================================
  游戏 1/20: 得分=850, 步数=45
  游戏 2/20: 得分=920, 步数=52
  ...

==============================================================
  评估结果
==============================================================
平均得分: 856.3 ± 102.5
最高得分: 1050
最低得分: 650
平均步数: 48.2
==============================================================
```

---

## 常见问题

### Q1: 训练太慢怎么办?

**方案1: 减少计算量**
```bash
python run_training.py train \
    --iterations 10 \
    --games 20 \
    --simulations 50  # 减少MCTS模拟次数
```

**方案2: 使用GPU**
```bash
pip install paddlepaddle-gpu
# 会自动使用GPU加速
```

**方案3: 云端训练**
- 使用Google Colab (免费GPU)
- 使用AutoDL等云服务器

### Q2: 内存不足?

```bash
python run_training.py train \
    --batch-size 16 \  # 减少batch size
    --games 20         # 减少每轮游戏数
```

### Q3: 如何看训练是否正常?

检查几个指标:
1. **Loss应该下降**: `train_losses` 逐渐减小
2. **分数应该提升**: `eval_scores` 总体上升
3. **Value Loss收敛**: 接近0表示网络能准确预测分数

```bash
# 查看训练历史
python visualize_results.py --summary-only
```

### Q4: 训练中断了怎么办?

```bash
# 从上次保存的检查点继续
python run_training.py train \
    --resume 10 \  # 从第10轮继续
    --iterations 20
```

### Q5: 模型性能不提升?

可能的原因和解决方案:
1. **训练轮数不够**: 增加 `--iterations`
2. **MCTS搜索不足**: 增加 `--simulations`
3. **学习率问题**: 修改 `TrainAlphaZero.py` 中的 `learning_rate`
4. **数据量不足**: 增加 `--games`

### Q6: 如何在Jupyter中使用?

```python
# 在Jupyter Notebook中
import sys
sys.path.append('/Users/ycy/Downloads/DQN_FruitMerger')

from TrainAlphaZero import train_alphazero

# 运行训练
train_alphazero(
    num_iterations=5,
    games_per_iteration=20,
    mcts_simulations=100
)

# 可视化
from visualize_results import visualize_training_history
visualize_training_history()
```

---

## 📊 性能预期

根据不同配置的预期性能:

| 配置 | 训练时间 | 预期最终分数 | 说明 |
|------|---------|-------------|------|
| 快速测试 | 5-10分钟 | 100-200 | 仅验证代码 |
| 本地标准 | 10-20小时 | 500-800 | 适合本地训练 |
| 完整训练 | 30-40小时 | 800-1200 | 推荐云端 |
| 高强度 | 60+小时 | 1200-2000+ | 最佳性能 |

---

## 🎯 推荐工作流程

### 初学者
1. 快速测试验证环境: `python run_training.py train --quick`
2. 查看可视化: `python visualize_results.py`
3. 本地小规模训练: 10轮迭代
4. 评估模型: `python run_training.py evaluate`

### 进阶用户
1. 云端完整训练: 20-50轮迭代
2. 定期下载检查点
3. 对比不同迭代的模型
4. 调整超参数优化

### 研究者
1. 修改网络结构 (SuikaNet.py)
2. 调整MCTS参数 (AlphaZeroMCTS.py)
3. 实验不同训练策略
4. 与其他算法对比

---

## 📞 获取帮助

- **查看文档**: `README_ALPHAZERO.md`
- **查看代码**: 所有脚本都有详细注释
- **运行帮助**: `python run_training.py --help`

---

**祝训练顺利! 🚀**
