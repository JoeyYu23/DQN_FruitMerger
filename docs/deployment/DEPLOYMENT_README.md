# 云服务器部署文件说明

本文档说明为你的云服务器配置创建的部署文件。

## 服务器配置

```
GPU:       RTX 3080 Ti (12GB) × 1
CPU:       12 vCPU Intel Xeon Silver 4214R @ 2.40GHz
内存:      90GB
CUDA:      11.8
Python:    3.10
PyTorch:   2.1.2
系统:      Ubuntu 22.04
费用:      ¥1.08-1.14/时
```

## 新增文件列表

| 文件名 | 用途 | 说明 |
|--------|------|------|
| `SERVER_DEPLOYMENT_GUIDE.md` | 完整部署指南 | 详细的分步骤部署说明 |
| `deploy_server.sh` | 自动部署脚本 | 一键安装所有依赖 |
| `upload_to_server.sh` | 上传脚本 | 从Mac上传到服务器 |
| `requirements_server.txt` | 服务器依赖 | PyTorch 2.1.2 + CUDA 11.8 |
| `DEPLOYMENT_README.md` | 本文件 | 部署文件说明 |

## 快速开始 (3步完成部署)

### 步骤1: 上传到服务器

在Mac上运行:

```bash
cd /Users/ycy/Downloads/DQN_FruitMerger

# 添加执行权限
chmod +x upload_to_server.sh

# 上传 (替换为你的服务器IP和用户名)
./upload_to_server.sh <服务器IP> root

# 例如:
# ./upload_to_server.sh 123.456.789.0 root
```

### 步骤2: SSH登录服务器

```bash
ssh root@<服务器IP>
cd /root/DQN_FruitMerger
```

### 步骤3: 运行自动部署

```bash
./deploy_server.sh
```

部署脚本会自动完成:
- ✓ 检查GPU和CUDA环境
- ✓ 创建Python虚拟环境
- ✓ 安装PyTorch 2.1.2 (CUDA 11.8)
- ✓ 安装所有项目依赖
- ✓ 运行环境测试

**预计时间**: 3-5分钟

## 详细使用说明

### 方案A: 自动上传 (推荐)

使用 `upload_to_server.sh` 脚本:

```bash
# 基本用法
./upload_to_server.sh <IP> <用户名>

# 指定远程路径
./upload_to_server.sh <IP> <用户名> /home/user

# 示例
./upload_to_server.sh 123.456.789.0 root /root
```

脚本功能:
1. 自动打包项目 (排除不必要文件)
2. 通过scp上传到服务器
3. 自动解压并设置权限
4. 清理临时文件

### 方案B: 手动上传

```bash
# 在Mac上打包
cd /Users/ycy/Downloads
tar -czf DQN_FruitMerger.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='venv' \
    DQN_FruitMerger/

# 上传
scp DQN_FruitMerger.tar.gz root@<服务器IP>:/root/

# SSH登录
ssh root@<服务器IP>

# 解压
cd /root
tar -xzf DQN_FruitMerger.tar.gz
cd DQN_FruitMerger
```

### 自动部署脚本详解

`deploy_server.sh` 执行的任务:

1. **环境检查**
   - Python版本
   - CUDA和GPU状态
   - 磁盘空间

2. **虚拟环境**
   - 创建venv
   - 升级pip

3. **安装PyTorch**
   - PyTorch 2.1.2
   - 对应CUDA 11.8版本
   - 验证GPU可用性

4. **安装依赖**
   - pymunk (物理引擎)
   - pygame (游戏界面)
   - opencv-python (图形处理)
   - numpy (数值计算)
   - tqdm (进度条)
   - tensorboard (监控)

5. **环境测试**
   - 导入测试
   - GPU计算测试
   - 网络架构测试

### 手动部署

如果自动脚本失败,可以手动执行:

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装PyTorch
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖
pip install -r requirements_server.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

## 训练和监控

### 启动训练

```bash
# 激活环境
source venv/bin/activate

# DQN训练 (需要PaddlePaddle)
# 注意: 需要先安装PaddlePaddle
pip install paddlepaddle-gpu==2.5.1.post118 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
python DQN.py

# PyTorch版本 (开发中,仅网络测试)
python SuikaNet_torch.py
```

### 后台训练

```bash
# 使用tmux (推荐)
tmux new -s train
source venv/bin/activate
python DQN.py
# Ctrl+B, D 分离

# 重新连接
tmux attach -s train

# 使用nohup
nohup python DQN.py > train.log 2>&1 &
tail -f train.log
```

### TensorBoard监控

```bash
# 启动TensorBoard (端口6006)
source venv/bin/activate
nohup tensorboard --logdir=./logs \
    --host=0.0.0.0 --port=6006 > tensorboard.log 2>&1 &

# 浏览器访问
# http://<服务器IP>:6006
```

### GPU监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看详细信息
nvidia-smi dmon -s pucvmet
```

## 优化建议

基于你的RTX 3080 Ti (12GB)配置:

### 训练参数优化

```python
# 可以使用更大的batch_size
BATCH_SIZE = 128  # 原版是32

# 更大的经验回放池 (90GB内存)
MEMORY_SIZE = 100000  # 原版是50000

# 更多训练轮数
EPISODES = 5000  # 原版是2000
```

### 性能预估

| 配置 | Episodes | 预计时间 | 费用 |
|------|----------|----------|------|
| 快速测试 | 100 | 0.5小时 | ¥0.57 |
| 标准训练 | 2000 | 8小时 | ¥9.12 |
| 完整训练 | 5000 | 20小时 | ¥22.80 |

## 常见问题

### Q1: 上传失败

**原因**: SSH密钥未配置

**解决**:
```bash
# 生成SSH密钥
ssh-keygen -t rsa

# 复制公钥到服务器
ssh-copy-id root@<服务器IP>
```

### Q2: 部署脚本权限错误

**解决**:
```bash
chmod +x deploy_server.sh
chmod +x upload_to_server.sh
```

### Q3: PyTorch安装失败

**解决**: 确认CUDA版本
```bash
nvidia-smi  # 查看CUDA版本

# 如果是CUDA 12.x
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Q4: 端口6006无法访问

**检查防火墙**:
```bash
sudo ufw allow 6006/tcp
sudo ufw allow 6008/tcp
```

### Q5: 训练中断

**使用tmux**:
```bash
tmux new -s train
# 运行训练
# Ctrl+B, D 分离会话
```

## 下载训练结果

训练完成后,下载模型到Mac:

```bash
# 在Mac上运行
scp -r root@<服务器IP>:/root/DQN_FruitMerger/weights ./models/
scp root@<服务器IP>:/root/DQN_FruitMerger/train.log ./logs/
```

## 关闭服务器

**重要**: 训练完成后立即关闭,避免继续计费!

```bash
# 在服务器上
sudo shutdown -h now

# 或在云服务商控制台关闭实例
```

## 成本控制

1. 训练完立即关机
2. 使用tmux防止断开
3. 设置训练完成通知
4. 定期下载checkpoint

## 技术支持

- **详细指南**: `SERVER_DEPLOYMENT_GUIDE.md`
- **GitHub**: https://github.com/JoeyYu23/DQN_FruitMerger
- **原始README**: `README.md`
- **PyTorch说明**: `README_PYTORCH.md`

## 下一步

1. ✓ 上传项目到服务器
2. ✓ 运行自动部署脚本
3. ☐ 开始训练
4. ☐ 监控训练进度
5. ☐ 下载训练结果
6. ☐ 关闭服务器实例

---

**祝部署顺利！** 🚀

如有问题,请查看 `SERVER_DEPLOYMENT_GUIDE.md` 获取详细帮助。
