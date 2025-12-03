# ☁️ 云端同步指南 - 确保版本一致

## 📋 概述

本指南帮助你将本地环境**精确复制**到云端，确保训练结果可复现。

---

## 🎯 快速开始

### 方案1: 使用Git (推荐)

```bash
# 1. 本地：初始化Git仓库并推送
cd /Users/ycy/Downloads/DQN_FruitMerger

# 如果还没有git仓库
git init
git add .
git commit -m "Initial commit with exact environment"
git branch -M main

# 推送到GitHub/GitLab
git remote add origin <你的仓库地址>
git push -u origin main

# 2. 云端：克隆并配置
git clone <你的仓库地址>
cd DQN_FruitMerger
bash setup_cloud.sh gpu  # 或 cpu
python verify_env.py
```

### 方案2: 直接上传文件

```bash
# 1. 本地：打包项目
cd /Users/ycy/Downloads
tar -czf DQN_FruitMerger.tar.gz DQN_FruitMerger/

# 2. 上传到云端
scp DQN_FruitMerger.tar.gz user@server:/workspace/

# 3. 云端：解压并配置
ssh user@server
cd /workspace
tar -xzf DQN_FruitMerger.tar.gz
cd DQN_FruitMerger
bash setup_cloud.sh gpu
python verify_env.py
```

---

## 📦 版本文件说明

已为你生成4个版本配置文件：

### 1. `requirements_exact.txt` ⭐ **推荐**
```
# 精确版本，与你的本地环境完全一致
paddlepaddle==3.2.1
numpy==1.26.4
opencv-python==4.11.0.86
pymunk==6.5.0
matplotlib==3.7.2
tqdm==4.67.1
psutil==5.9.0
```

**适用场景**:
- 确保完全一致的环境
- 复现训练结果
- 调试问题

### 2. `requirements_flexible.txt`
```
# 兼容版本范围
paddlepaddle>=3.0.0,<4.0.0
numpy>=1.20.0,<2.0.0
...
```

**适用场景**:
- 云端环境与本地不完全兼容
- 需要更新的包版本
- 快速部署

### 3. `requirements_full.txt`
```
# 完整的pip freeze输出
# 包含所有依赖和子依赖
```

**适用场景**:
- 最彻底的环境复制
- 解决隐藏的依赖问题

### 4. `setup_cloud.sh`
自动化安装脚本，智能处理CPU/GPU版本。

---

## 🚀 不同云平台部署

### Google Colab

```python
# 新建Colab笔记本

# 1. 克隆项目
!git clone <你的仓库地址>
%cd DQN_FruitMerger

# 2. 安装依赖 (Colab有GPU)
!pip install paddlepaddle-gpu==3.2.1
!pip install numpy==1.26.4 opencv-python==4.11.0.86 pymunk==6.5.0
!pip install matplotlib==3.7.2 tqdm==4.67.1 psutil==5.9.0

# 3. 验证环境
!python verify_env.py

# 4. 开始训练
!python run_training.py train --iterations 20 --games 50 --simulations 200

# 5. 下载结果
from google.colab import files
!zip -r results.zip weights/alphazero/ *.png *.json
files.download('results.zip')
```

### AutoDL / 腾讯云 / 阿里云

```bash
# 1. SSH连接服务器
ssh user@server_ip

# 2. 创建工作目录
mkdir -p /workspace/suikagame
cd /workspace/suikagame

# 3. 上传项目 (本地运行)
scp -r /Users/ycy/Downloads/DQN_FruitMerger/* user@server:/workspace/suikagame/

# 4. 回到服务器，配置环境
cd /workspace/suikagame
bash setup_cloud.sh gpu

# 5. 验证环境
python verify_env.py

# 6. 后台训练
nohup python run_training.py train \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    > train.log 2>&1 &

# 7. 监控进度
tail -f train.log
```

### Kaggle

```python
# 在Kaggle Notebook中

# 1. 上传项目文件到Kaggle Dataset
# 或者从GitHub克隆

# 2. 安装依赖
!pip install -q paddlepaddle-gpu==3.2.1
!pip install -q -r requirements_exact.txt

# 3. 训练
!python run_training.py train --iterations 10

# 4. 保存结果
import shutil
shutil.make_archive('training_results', 'zip', 'weights/alphazero')
```

---

## 🔧 版本冲突解决

### 问题1: PaddlePaddle版本不兼容

```bash
# 卸载旧版本
pip uninstall paddlepaddle paddlepaddle-gpu -y

# 安装精确版本
# CPU:
pip install paddlepaddle==3.2.1

# GPU:
pip install paddlepaddle-gpu==3.2.1
```

### 问题2: NumPy版本冲突

```bash
# NumPy 2.0有breaking changes
pip install "numpy<2.0" --force-reinstall
```

### 问题3: OpenCV找不到

```bash
# 尝试不同的opencv包
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-python==4.8.1.78
```

### 问题4: 云端没有GPU但安装了GPU版本

```bash
# 切换到CPU版本
pip uninstall paddlepaddle-gpu -y
pip install paddlepaddle==3.2.1
```

---

## ✅ 环境验证清单

部署到云端后，务必运行验证：

```bash
# 1. 检查环境
python verify_env.py

# 2. 快速测试训练
python run_training.py train --quick

# 3. 检查生成的文件
ls -lh weights/alphazero/
```

期望输出：
```
✅ 环境验证通过！可以开始训练

核心依赖:
✓ paddlepaddle==3.2.1
✓ numpy==1.26.4
✓ opencv-python==4.11.0.86
✓ pymunk==6.5.0
...

功能测试:
✓ PaddlePaddle CPU
✓ Pymunk物理引擎
✓ OpenCV图像处理
✓ 游戏核心模块
✓ AlphaZero模块
```

---

## 📥 云端结果下载

### 方法1: SCP直接下载

```bash
# 下载训练权重
scp -r user@server:/workspace/DQN_FruitMerger/weights/alphazero ./weights/

# 下载可视化图表
scp user@server:/workspace/DQN_FruitMerger/*.png ./

# 下载训练日志
scp user@server:/workspace/DQN_FruitMerger/train.log ./
```

### 方法2: 打包下载

```bash
# 云端打包
cd /workspace/DQN_FruitMerger
zip -r results.zip \
    weights/alphazero/ \
    *.png \
    *.log \
    weights/alphazero/history.json

# 本地下载
scp user@server:/workspace/DQN_FruitMerger/results.zip ./
```

### 方法3: Git同步

```bash
# 云端提交
cd /workspace/DQN_FruitMerger
git add weights/alphazero/*.pdparams
git add weights/alphazero/history.json
git add *.png
git commit -m "Training iteration 20 completed"
git push

# 本地拉取
cd /Users/ycy/Downloads/DQN_FruitMerger
git pull
```

---

## 🔄 持续同步工作流

### 推荐工作流程

```
本地开发 → Git推送 → 云端拉取 → 云端训练 → Git推送 → 本地拉取
```

### 实践示例

```bash
# 1. 本地修改代码
vim AlphaZeroMCTS.py

# 2. 提交并推送
git add .
git commit -m "Improved MCTS exploration"
git push

# 3. 云端拉取更新
ssh user@server
cd /workspace/DQN_FruitMerger
git pull

# 4. 云端训练
nohup python run_training.py train --iterations 20 > train.log 2>&1 &

# 5. 训练完成后提交结果
git add weights/alphazero/
git commit -m "Training results iter 20"
git push

# 6. 本地拉取结果
exit  # 退出SSH
cd /Users/ycy/Downloads/DQN_FruitMerger
git pull

# 7. 本地分析结果
python visualize_results.py
```

---

## 🛠️ 实用工具脚本

### 快速上传脚本 `sync_to_cloud.sh`

```bash
#!/bin/bash
# 快速同步到云端

SERVER="user@server_ip"
REMOTE_PATH="/workspace/DQN_FruitMerger"

echo "同步代码到云端..."
rsync -avz --exclude 'weights/' \
           --exclude '__pycache__/' \
           --exclude '*.pyc' \
           ./ $SERVER:$REMOTE_PATH/

echo "✅ 同步完成"
```

### 快速下载脚本 `sync_from_cloud.sh`

```bash
#!/bin/bash
# 快速下载云端结果

SERVER="user@server_ip"
REMOTE_PATH="/workspace/DQN_FruitMerger"

echo "下载训练结果..."
rsync -avz $SERVER:$REMOTE_PATH/weights/alphazero/ ./weights/alphazero/
rsync -avz $SERVER:$REMOTE_PATH/*.png ./
rsync -avz $SERVER:$REMOTE_PATH/*.log ./

echo "✅ 下载完成"
```

---

## 📊 版本对照表

记录你的环境版本，便于问题追踪：

| 包名 | 本地版本 | 云端版本 | 状态 |
|------|---------|---------|------|
| Python | 3.11.5 | _____ | 待填写 |
| PaddlePaddle | 3.2.1 | _____ | 待填写 |
| NumPy | 1.26.4 | _____ | 待填写 |
| OpenCV | 4.11.0.86 | _____ | 待填写 |
| Pymunk | 6.5.0 | _____ | 待填写 |

填写方法：
```bash
# 云端运行
python -c "import sys; print('Python:', sys.version)"
python -c "import paddle; print('PaddlePaddle:', paddle.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

## 💡 最佳实践建议

1. **使用Git**: 代码和小文件用Git管理
2. **大文件分离**: 模型权重用SCP或云盘传输
3. **版本锁定**: 优先使用`requirements_exact.txt`
4. **定期验证**: 每次环境变更后运行`verify_env.py`
5. **备份权重**: 训练中途定期下载checkpoint
6. **日志监控**: 使用`tail -f`实时查看训练进度
7. **资源监控**: 云端运行`nvidia-smi`检查GPU使用

---

## 🆘 常见错误排查

### 错误: "No module named 'paddle'"
```bash
# 检查pip安装的位置
pip show paddlepaddle
# 确保使用正确的python
which python
python -m pip install paddlepaddle==3.2.1
```

### 错误: CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi
# 安装匹配的PaddlePaddle版本
# 参考: https://www.paddlepaddle.org.cn/install/quick
```

### 错误: 权重文件损坏
```bash
# 重新下载
rm -rf weights/alphazero/*
scp -r user@server:/workspace/DQN_FruitMerger/weights/alphazero/* ./weights/alphazero/
```

---

**祝云端训练顺利！🚀**

有问题随时参考本指南或运行 `python verify_env.py`
