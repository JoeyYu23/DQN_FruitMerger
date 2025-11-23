# 云服务器部署指南

**服务器配置信息**
```
GPU:       RTX 3080 Ti (12GB) × 1
CPU:       12 vCPU Intel Xeon Silver 4214R @ 2.40GHz
内存:      90GB
系统盘:    30GB
数据盘:    50GB
系统:      Ubuntu 22.04
Python:    3.10
CUDA:      11.8
PyTorch:   2.1.2
端口映射:  6006 (http), 6008 (http)
计费:      ¥1.08/时 - ¥1.14/时
```

---

## 目录

1. [快速开始](#快速开始)
2. [详细部署步骤](#详细部署步骤)
3. [训练配置优化](#训练配置优化)
4. [监控和调试](#监控和调试)
5. [常见问题](#常见问题)
6. [成本估算](#成本估算)

---

## 快速开始

### 前置要求

在本地Mac上准备:
```bash
cd /Users/ycy/Downloads/DQN_FruitMerger

# 打包项目
tar -czf DQN_FruitMerger.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='weights' \
    --exclude='videos' \
    .
```

### 一键部署

**步骤1: 上传到服务器**
```bash
# 替换为你的服务器IP和用户名
SERVER_IP="your.server.ip"
SERVER_USER="root"

# 上传项目
scp DQN_FruitMerger.tar.gz ${SERVER_USER}@${SERVER_IP}:/root/
```

**步骤2: SSH登录服务器**
```bash
ssh ${SERVER_USER}@${SERVER_IP}
```

**步骤3: 解压并运行部署脚本**
```bash
# 解压
cd /root
tar -xzf DQN_FruitMerger.tar.gz
cd DQN_FruitMerger

# 添加执行权限并运行部署脚本
chmod +x deploy_server.sh
./deploy_server.sh
```

部署脚本会自动完成:
- ✓ 检查系统环境 (GPU, CUDA, Python)
- ✓ 创建虚拟环境
- ✓ 安装 PyTorch 2.1.2 (CUDA 11.8)
- ✓ 安装所有依赖
- ✓ 运行环境测试

预计时间: **3-5分钟**

---

## 详细部署步骤

### 方法A: 使用自动化脚本 (推荐)

参考上面的"快速开始"部分。

### 方法B: 手动部署

如果自动脚本遇到问题,可以手动执行以下步骤:

#### 1. 检查GPU和CUDA

```bash
# 检查GPU
nvidia-smi

# 期望输出:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 520.xx       Driver Version: 520.xx       CUDA Version: 11.8     |
# +-----------------------------------------------------------------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |   0  RTX 3080 Ti     Off  |   ...             |                  N/A |
# |                              |                       |        12288MiB      |
```

#### 2. 创建虚拟环境

```bash
cd /root/DQN_FruitMerger

# 创建虚拟环境
python3 -m venv venv

# 激活
source venv/bin/activate

# 升级pip
pip install --upgrade pip
```

#### 3. 安装PyTorch (CUDA 11.8)

```bash
# 安装PyTorch 2.1.2 for CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

期望输出:
```
PyTorch: 2.1.2+cu118
CUDA Available: True
```

#### 4. 安装项目依赖

```bash
# 使用清华镜像源加速
pip install pymunk pygame opencv-python numpy tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple

# 可选: 安装TensorBoard用于监控
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 5. 验证环境

```bash
python -c "
import torch
import pymunk
import pygame
import cv2
import numpy as np
print('✓ 所有依赖导入成功')
print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
"
```

---

## 训练配置优化

### RTX 3080 Ti (12GB) 推荐配置

你的GPU显存充足,可以使用更大的batch_size来加速训练。

#### DQN训练 (原版PaddlePaddle)

**注意**: 项目原版使用PaddlePaddle,如果要使用需要额外安装:

```bash
# 安装PaddlePaddle GPU版本 (CUDA 11.8)
pip install paddlepaddle-gpu==2.5.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

然后运行训练:
```bash
# 标准配置
python DQN.py
```

#### PyTorch版本 (开发中)

当前PyTorch版本只实现了网络部分,完整训练脚本开发中:

```bash
# 测试PyTorch网络
python SuikaNet_torch.py
```

### 性能优化建议

基于你的硬件配置:

```python
# 推荐的训练参数
BATCH_SIZE = 128          # RTX 3080 Ti可以支持更大的batch
LEARNING_RATE = 0.001
GAMMA = 0.99
MEMORY_SIZE = 100000      # 90GB内存可以存更多经验
EPISODES = 5000           # 增加训练轮数
UPDATE_FREQUENCY = 4
TARGET_UPDATE = 1000
```

### 多GPU训练 (未来扩展)

虽然当前只有1张GPU,但代码可以为多GPU做准备:

```python
# PyTorch多GPU示例 (需要修改训练代码)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 监控和调试

### 1. TensorBoard监控

启动TensorBoard (使用端口6006):

```bash
# 后台运行
nohup tensorboard --logdir=./logs --host=0.0.0.0 --port=6006 > tensorboard.log 2>&1 &
```

然后在浏览器访问:
```
http://your.server.ip:6006
```

### 2. GPU使用监控

实时查看GPU状态:
```bash
# 每秒刷新
watch -n 1 nvidia-smi

# 或者使用更详细的监控
nvidia-smi dmon -s pucvmet
```

### 3. 系统资源监控

```bash
# CPU和内存
htop

# 磁盘IO
iotop

# 网络
iftop
```

### 4. 训练日志

```bash
# 实时查看训练日志
tail -f training.log

# 查看最后100行
tail -n 100 training.log
```

### 5. 后台训练

使用screen或tmux防止SSH断开导致训练中断:

```bash
# 方法1: 使用tmux (推荐)
tmux new -s train
python DQN.py
# 按 Ctrl+B 然后按 D 分离会话

# 重新连接
tmux attach -s train

# 方法2: 使用nohup
nohup python DQN.py > train.log 2>&1 &

# 查看进程
ps aux | grep python
```

---

## 常见问题

### Q1: CUDA Out of Memory

**问题**: 显存不足

**解决方案**:
```python
# 减小batch_size
BATCH_SIZE = 64  # 从128降到64

# 或清理GPU缓存
import torch
torch.cuda.empty_cache()
```

### Q2: PyTorch版本不完整

**问题**: PyTorch训练脚本还在开发中

**解决方案**: 使用PaddlePaddle版本
```bash
pip install paddlepaddle-gpu==2.5.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
python DQN.py
```

### Q3: SSH断开导致训练中断

**解决方案**: 使用tmux或screen
```bash
tmux new -s training
# 运行训练
# Ctrl+B, D 分离
```

### Q4: 依赖安装失败

**解决方案**: 使用国内镜像源
```bash
pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q5: 端口6006无法访问

**检查**:
```bash
# 1. 检查TensorBoard是否运行
ps aux | grep tensorboard

# 2. 检查端口是否监听
netstat -tulpn | grep 6006

# 3. 检查防火墙
sudo ufw status
sudo ufw allow 6006/tcp
```

### Q6: 训练速度慢

**优化建议**:
1. 增加batch_size (利用12GB显存)
2. 使用混合精度训练
3. 减少日志输出频率
4. 使用更高效的数据加载

---

## 成本估算

### 训练时间预估

基于RTX 3080 Ti性能:

| 任务 | Episodes | 预计时间 | 费用 (¥1.14/时) |
|------|----------|----------|----------------|
| 快速测试 | 100 | 0.5小时 | ¥0.57 |
| 标准训练 | 2000 | 8小时 | ¥9.12 |
| 完整训练 | 5000 | 20小时 | ¥22.80 |
| 长期训练 | 10000 | 40小时 | ¥45.60 |

### 省钱技巧

1. **及时关机**: 训练完立即关闭实例
2. **批量训练**: 一次训练多个模型
3. **使用checkpoint**: 支持断点续训
4. **监控告警**: 设置训练完成通知

### 自动关机脚本

训练完成后自动关机:

```bash
# train_and_shutdown.sh
#!/bin/bash
python DQN.py
echo "训练完成,60秒后关机..."
sleep 60
sudo shutdown -h now
```

---

## 快速命令参考

### 常用命令

```bash
# 激活环境
source venv/bin/activate

# 开始训练
python DQN.py

# 后台训练
nohup python DQN.py > train.log 2>&1 &

# 查看日志
tail -f train.log

# 监控GPU
watch -n 1 nvidia-smi

# 启动TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006

# 退出环境
deactivate
```

### 文件传输

```bash
# 上传到服务器
scp local_file user@server:/path/

# 从服务器下载
scp user@server:/path/file local_path/

# 下载训练好的模型
scp user@server:/root/DQN_FruitMerger/weights/* ./models/
```

---

## 下一步

1. **完成部署**: 运行 `./deploy_server.sh`
2. **测试环境**: 确保所有测试通过
3. **开始训练**: 运行 `python DQN.py`
4. **监控进度**: 使用TensorBoard查看训练曲线
5. **下载模型**: 训练完成后下载权重文件
6. **关闭实例**: 避免继续计费

---

## 技术支持

- **GitHub仓库**: https://github.com/JoeyYu23/DQN_FruitMerger
- **问题反馈**: 创建GitHub Issue
- **文档**: 查看项目README.md

---

**部署愉快！** 🚀

如有问题,请参考常见问题部分或提交Issue。
