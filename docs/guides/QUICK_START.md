# 快速开始 - GitHub + AutoDL 部署

最简单的云端部署方式：本地Mac → GitHub → AutoDL服务器

---

## 🎯 三步完成部署

### 步骤1: 推送到GitHub (在Mac上)

```bash
cd /Users/ycy/Downloads/DQN_FruitMerger

# 一键推送
./push_to_github.sh
```

**预计时间**: 1-2分钟

### 步骤2: 创建AutoDL实例

访问 https://www.autodl.com

```
GPU: RTX 3080 Ti / RTX 3090 / RTX 4090
镜像: PyTorch 2.1.0 + Python 3.10 + CUDA 11.8
存储: 50GB系统盘
价格: ~1.5-2.5元/小时
```

点击"立即创建"

### 步骤3: 在AutoDL上部署 (在JupyterLab或SSH)

**方法A: 使用一键脚本**

```bash
# 下载并运行一键部署脚本
cd /root
wget https://raw.githubusercontent.com/JoeyYu23/DQN_FruitMerger/main/autodl_quick_deploy.sh
chmod +x autodl_quick_deploy.sh
./autodl_quick_deploy.sh
```

**方法B: 手动部署**

```bash
cd /root
git clone https://github.com/JoeyYu23/DQN_FruitMerger.git
cd DQN_FruitMerger
./deploy_server.sh
```

**预计时间**: 3-5分钟

---

## 🚀 开始训练

```bash
# 激活环境
source venv/bin/activate

# 安装PaddlePaddle (DQN训练需要)
pip install paddlepaddle-gpu==2.5.1.post118 \
    -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# 后台训练（推荐）
tmux new -s train
python DQN.py
# 按 Ctrl+B, 然后按 D 分离
```

---

## 📊 监控训练

### GPU监控
```bash
watch -n 1 nvidia-smi
```

### TensorBoard
```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006 &
# 访问: http://你的实例链接:6006
```

### 查看日志
```bash
tail -f training.log
```

---

## 💾 下载模型

训练完成后，在Mac上运行：

```bash
# 方法1: 从AutoDL JupyterLab下载
# 在文件浏览器中右键点击文件 -> Download

# 方法2: 使用scp
scp -P <端口> -r root@connect.autodl.com:/root/DQN_FruitMerger/weights ./models/
```

---

## 💰 关闭实例（重要！）

训练完成后立即关闭避免计费：

```
AutoDL控制台 → 容器实例 → 关机
```

---

## 📚 详细文档

| 文档 | 说明 |
|------|------|
| `GITHUB_AUTODL_GUIDE.md` | GitHub + AutoDL完整指南 |
| `SERVER_DEPLOYMENT_GUIDE.md` | 服务器部署详细说明 |
| `DEPLOYMENT_README.md` | 部署文件总览 |

---

## 🛠️ 创建的工具脚本

| 脚本 | 用途 |
|------|------|
| `push_to_github.sh` | 一键推送到GitHub |
| `autodl_quick_deploy.sh` | AutoDL一键部署 |
| `deploy_server.sh` | 自动环境配置 |
| `upload_to_server.sh` | 直接上传到服务器 |

---

## ⚡ 快速命令参考

### Mac本地
```bash
# 推送到GitHub
./push_to_github.sh

# 或手动
git add .
git commit -m "your message"
git push origin main
```

### AutoDL服务器
```bash
# 克隆
git clone https://github.com/JoeyYu23/DQN_FruitMerger.git

# 部署
cd DQN_FruitMerger && ./deploy_server.sh

# 训练
source venv/bin/activate && python DQN.py

# 监控
watch -n 1 nvidia-smi
```

---

## 💡 成本估算

| 训练规模 | Episodes | 时间 | 费用(¥2/时) |
|---------|----------|------|------------|
| 快速测试 | 100 | 0.5h | ¥1 |
| 标准训练 | 2000 | 8h | ¥16 |
| 完整训练 | 5000 | 20h | ¥40 |

---

## ❓ 遇到问题？

**克隆速度慢:**
```bash
# 使用ZIP下载
wget https://github.com/JoeyYu23/DQN_FruitMerger/archive/refs/heads/main.zip
unzip main.zip
```

**训练中断:**
```bash
# 使用tmux
tmux new -s train
# 运行训练后按 Ctrl+B, D 分离
# 重连: tmux attach -s train
```

**查看更多帮助:**
- 查看 `GITHUB_AUTODL_GUIDE.md`
- 查看 `SERVER_DEPLOYMENT_GUIDE.md`

---

**就是这么简单！** 🎉

3步完成部署，开始你的云端训练之旅！
