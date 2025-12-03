# 🚀 云端部署完整教程

**目标**: 将AlphaZero训练部署到云端GPU服务器，享受2-4倍加速

**总耗时**: 15-20分钟（首次配置）

---

## 📋 准备清单

在开始之前，确保你有：

- [x] 项目代码（已在 `/Users/ycy/Downloads/DQN_FruitMerger`）
- [ ] 云服务账号（阿里云/腾讯云/AWS等）
- [ ] 信用卡或支付宝（用于付费）
- [ ] SSH工具（Mac自带Terminal即可）

**预计成本**: 1-5元（20轮完整训练）

---

## 🎯 方案选择

### 推荐方案对比

| 云服务商 | GPU类型 | 价格 | 优势 | 适合人群 |
|---------|---------|------|------|---------|
| **阿里云** | Tesla T4 | ~8元/时 | 国内快，中文界面 | 国内用户 ⭐⭐⭐ |
| **腾讯云** | Tesla T4 | ~7元/时 | 便宜，学生优惠 | 学生用户 ⭐⭐⭐ |
| **AutoDL** | RTX 3090 | ~1.5元/时 | 超便宜，专为深度学习 | 预算有限 ⭐⭐⭐⭐⭐ |
| **AWS** | Tesla T4 | ~$0.5/时 | 全球覆盖 | 海外用户 ⭐⭐ |
| **Google Colab** | 免费GPU | 免费 | 完全免费！ | 快速试用 ⭐⭐⭐⭐ |

**我的推荐**:
1. 🥇 **AutoDL** - 性价比最高（1.5元/时）
2. 🥈 **Google Colab** - 完全免费（有时间限制）
3. 🥉 **阿里云/腾讯云** - 稳定可靠（新用户有优惠）

---

## 📖 详细部署步骤

我会提供3个最常用方案的详细步骤：

---

## 🎯 方案1: AutoDL（最推荐，超便宜）

### 优势
- ✅ 专为深度学习优化
- ✅ 价格超低（1.5元/时）
- ✅ 镜像预装好环境
- ✅ 按秒计费，用多少付多少

### 步骤

#### 1. 注册AutoDL账号
```
网址: https://www.autodl.com
- 手机号注册
- 实名认证（需要）
- 充值50元即可（够用很久）
```

#### 2. 创建实例

登录后，点击"算力市场"：

```
GPU选择: RTX 3090 (24GB)  ← 推荐
         或 Tesla T4 (16GB)

地区: 任意（选延迟低的）

镜像: PyTorch 1.12.0 + Python 3.8 + CUDA 11.3
     （会预装好conda和基础环境）

存储: 50GB系统盘 + 0GB数据盘（够用）

价格: ~1.5元/小时

点击"立即创建"
```

#### 3. 连接服务器

创建成功后，在"容器实例"页面：

```
点击 "JupyterLab" 按钮
→ 会打开一个网页版终端

或者使用SSH:
点击 "SSH" 按钮查看连接命令，类似：
ssh -p 12345 root@connect.autodl.com
```

#### 4. 上传代码

**方法A: 使用JupyterLab上传（推荐）**

```
1. 在JupyterLab界面，点击左上角"上传"按钮
2. 选择本地的整个 DQN_FruitMerger 文件夹压缩包
3. 上传后在终端解压:
   cd /root
   unzip DQN_FruitMerger.zip
```

**方法B: 使用scp命令**

在你的Mac终端：
```bash
# 先打包
cd /Users/ycy/Downloads
tar -czf DQN_FruitMerger.tar.gz DQN_FruitMerger/

# 上传（替换端口和地址）
scp -P 12345 DQN_FruitMerger.tar.gz root@connect.autodl.com:/root/

# SSH登录服务器
ssh -p 12345 root@connect.autodl.com

# 解压
cd /root
tar -xzf DQN_FruitMerger.tar.gz
```

#### 5. 安装依赖

在服务器终端：

```bash
cd /root/DQN_FruitMerger

# AutoDL已经有conda，直接安装依赖
pip install -r requirements_alphazero.txt

# 安装GPU版本PaddlePaddle
pip install paddlepaddle-gpu

# 验证GPU可用
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"
# 输出 True 表示成功
```

#### 6. 测试

```bash
# 快速测试
python test_pipeline.py

# 期望输出: All Pipeline Tests Passed! ✓
```

#### 7. 开始训练！

```bash
# 方式1: 快速测试（2分钟）
python TrainAlphaZero.py --iterations 2 --games 10 --simulations 50

# 方式2: 标准训练（8-15分钟）
./train_cloud.sh 20 50 200 32 5 10

# 方式3: 后台运行（推荐）
nohup ./train_cloud.sh 20 50 200 32 5 10 > train.log 2>&1 &

# 查看进度
tail -f train.log
```

#### 8. 监控训练

**实时查看GPU使用**:
```bash
watch -n 1 nvidia-smi
```

**查看训练历史**:
```bash
cat weights/alphazero/history.json | python -m json.tool
```

#### 9. 下载结果

训练完成后，下载模型：

```bash
# 在你的Mac上运行
scp -P 12345 root@connect.autodl.com:/root/DQN_FruitMerger/weights/alphazero/* ./alphazero_models/

# 下载训练日志
scp -P 12345 root@connect.autodl.com:/root/DQN_FruitMerger/train.log ./
```

#### 10. 关闭实例（省钱！）

训练完成后，**立即关闭实例**避免继续计费：

```
在AutoDL网页 → "容器实例" → 点击 "关机"
```

**费用计算**:
```
20轮训练 × 8分钟 = ~0.2小时
0.2小时 × 1.5元/小时 = 0.3元！

超便宜！🎉
```

---

## 🎯 方案2: Google Colab（完全免费）

### 优势
- ✅ 完全免费！
- ✅ 提供免费GPU（Tesla T4）
- ✅ 无需配置环境
- ✅ 浏览器直接使用

### 限制
- ❌ 连续使用12小时会断开
- ❌ 空闲90分钟会断开
- ❌ 需要手动上传代码

### 步骤

#### 1. 打开Google Colab

```
网址: https://colab.research.google.com
使用Google账号登录
```

#### 2. 创建新笔记本

```
文件 → 新建笔记本
```

#### 3. 启用GPU

```
运行时 → 更改运行时类型 → 硬件加速器 → GPU → 保存
```

#### 4. 上传代码

在第一个代码单元格：

```python
# 方法1: 从Google Drive上传（推荐）
from google.colab import drive
drive.mount('/content/drive')

# 先在Google Drive上传压缩包，然后:
!cp /content/drive/MyDrive/DQN_FruitMerger.tar.gz /content/
!tar -xzf DQN_FruitMerger.tar.gz
```

```python
# 方法2: 直接上传（适合小文件）
from google.colab import files
uploaded = files.upload()  # 会弹出上传对话框

# 解压
!tar -xzf DQN_FruitMerger.tar.gz
```

#### 5. 安装依赖

新建代码单元格：

```python
!cd DQN_FruitMerger && pip install -r requirements_alphazero.txt
!pip install paddlepaddle-gpu

# 验证GPU
import paddle
print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")
```

#### 6. 运行训练

```python
!cd DQN_FruitMerger && python TrainAlphaZero.py \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    --batch-size 32 \
    --epochs 5 \
    --eval-games 10
```

#### 7. 查看进度

```python
# 实时查看日志
!tail -20 DQN_FruitMerger/logs/train_*.log

# 查看训练历史
!cat DQN_FruitMerger/weights/alphazero/history.json
```

#### 8. 下载结果

训练完成后：

```python
from google.colab import files

# 下载模型
!cd DQN_FruitMerger && tar -czf models.tar.gz weights/

# 触发下载
files.download('DQN_FruitMerger/models.tar.gz')
```

**提示**: Colab会话最多12小时，如果训练超时，需要分批训练：
```python
# 第一次
!python TrainAlphaZero.py --iterations 10 --checkpoint-dir weights/alphazero

# 断开后，第二次从检查点恢复
!python TrainAlphaZero.py --resume 10 --iterations 20 --checkpoint-dir weights/alphazero
```

---

## 🎯 方案3: 阿里云/腾讯云（稳定可靠）

### 优势
- ✅ 国内网络快
- ✅ 稳定可靠
- ✅ 新用户有优惠券

### 步骤

#### 1. 注册并登录

**阿里云**: https://www.aliyun.com
**腾讯云**: https://cloud.tencent.com

注册并完成实名认证

#### 2. 购买GPU实例

以阿里云为例：

```
产品 → 云服务器ECS → 立即购买

地域: 选择离你近的（如华东、华北）

实例规格:
  计算型 → GPU计算型
  → ecs.gn6v-c8g1.2xlarge (Tesla T4)

镜像:
  公共镜像 → Ubuntu 20.04 64位

网络:
  默认即可

购买时长:
  按量付费（用多少付多少）

价格: ~8元/小时
```

#### 3. 连接服务器

购买成功后，在"实例列表"找到你的实例：

```
获取公网IP: 比如 47.123.45.67

设置密码:
  实例 → 更多 → 密码/密钥 → 重置实例密码

连接:
ssh root@47.123.45.67
输入密码
```

#### 4. 配置环境

首次登录后：

```bash
# 更新系统
apt update && apt upgrade -y

# 安装conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="/root/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 创建环境
conda create -n alphazero python=3.8 -y
conda activate alphazero

# 安装NVIDIA驱动（如果没有）
apt install nvidia-driver-470 -y

# 重启验证GPU
nvidia-smi
```

#### 5. 上传代码

在你的Mac：

```bash
# 打包
cd /Users/ycy/Downloads
tar -czf DQN_FruitMerger.tar.gz DQN_FruitMerger/

# 上传
scp DQN_FruitMerger.tar.gz root@47.123.45.67:/root/

# 登录服务器
ssh root@47.123.45.67

# 解压
tar -xzf DQN_FruitMerger.tar.gz
cd DQN_FruitMerger
```

#### 6. 安装依赖

```bash
conda activate alphazero
pip install -r requirements_alphazero.txt
pip install paddlepaddle-gpu
```

#### 7-10步同AutoDL

训练、监控、下载步骤与AutoDL相同。

#### 11. 释放实例

**重要**: 训练完成后立即释放实例！

```
阿里云控制台 → ECS实例 → 更多 → 实例状态 → 释放
```

---

## 🎓 完整流程示意图

```
┌─────────────────────────────────────────────┐
│  第1步: 选择云服务                            │
│  AutoDL / Colab / 阿里云 / 腾讯云             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第2步: 创建GPU实例                          │
│  选择: Tesla T4 / RTX 3090                   │
│  区域: 离你近的数据中心                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第3步: 上传代码                             │
│  scp / Web上传 / Git clone                   │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第4步: 安装环境                             │
│  pip install requirements                    │
│  pip install paddlepaddle-gpu               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第5步: 测试验证                             │
│  python test_pipeline.py                    │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第6步: 开始训练！                           │
│  nohup ./train_cloud.sh 20 50 200 &         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第7步: 监控进度                             │
│  tail -f train.log                          │
│  watch nvidia-smi                           │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第8步: 下载结果                             │
│  scp weights/ 到本地                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  第9步: 关闭实例（省钱！）                    │
│  释放/关机                                   │
└─────────────────────────────────────────────┘
```

---

## 💰 成本估算

### AutoDL (最便宜)
```
GPU: RTX 3090 (1.5元/小时)
训练时间: 8分钟 = 0.13小时
成本: 0.13 × 1.5 = 0.2元 ✅
```

### Google Colab (免费)
```
GPU: Tesla T4
训练时间: 15分钟
成本: 免费！✅✅✅
```

### 阿里云/腾讯云
```
GPU: Tesla T4 (8元/小时)
训练时间: 10分钟 = 0.17小时
成本: 0.17 × 8 = 1.4元 ✅
```

---

## ⚠️ 注意事项

### 1. 及时关闭实例！

```
训练完成后立即关闭，否则会一直计费！

设置提醒:
- 在手机上设置闹钟
- 或使用云监控自动关闭
```

### 2. 数据备份

```
定期下载检查点:
scp root@server:/root/DQN_FruitMerger/weights/alphazero/* ./backup/

万一实例意外关闭，可以从检查点恢复
```

### 3. 网络问题

```
如果上传很慢:
- 使用Git: 先push到GitHub，服务器上git clone
- 使用对象存储: 上传到阿里云OSS，服务器下载
```

### 4. GPU Out of Memory

```
如果显存不够:
- 减小batch_size: --batch-size 16
- 减小网络: hidden_channels=32
- 减小模拟次数: --simulations 100
```

---

## 🎯 快速对比表

| 方案 | 难度 | 成本 | 速度 | 推荐度 |
|------|------|------|------|--------|
| **AutoDL** | ⭐ | 0.2元 | 快 | ⭐⭐⭐⭐⭐ |
| **Colab** | ⭐ | 免费 | 中 | ⭐⭐⭐⭐ |
| **阿里云** | ⭐⭐ | 1.4元 | 快 | ⭐⭐⭐ |
| **腾讯云** | ⭐⭐ | 1.2元 | 快 | ⭐⭐⭐ |

**我的建议**:
1. 首选 **AutoDL** - 最简单、最便宜、最快
2. 备选 **Colab** - 免费，但有限制
3. 如果上面两个不行，用 **阿里云/腾讯云**

---

## 📞 遇到问题？

### 常见问题快速解决

**Q: GPU不可用？**
```bash
# 检查CUDA
nvidia-smi

# 重装paddlepaddle-gpu
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

**Q: 依赖安装失败？**
```bash
# 换国内源
pip install -r requirements_alphazero.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Q: 连接超时？**
```bash
# 检查安全组规则（云服务商控制台）
# 确保开放SSH端口（22）
```

---

## ✅ 检查清单

部署前确认：
- [ ] 云服务账号已注册
- [ ] 余额充足（至少10元）
- [ ] 代码已打包
- [ ] SSH工具准备好

部署后确认：
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 环境测试通过 (`python test_pipeline.py`)
- [ ] 训练日志正常
- [ ] 定时检查进度

训练完成：
- [ ] 下载模型到本地
- [ ] 下载训练日志
- [ ] **关闭实例**（重要！）
- [ ] 验证费用

---

**你现在可以开始了！建议从AutoDL开始，5分钟内就能开始训练！** 🚀
