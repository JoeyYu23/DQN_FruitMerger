# 🔧 版本同步完成！

## ✅ 已生成的文件

为确保本地和云端环境完全一致，已为你生成以下文件：

### 📦 环境配置文件

1. **`requirements_exact.txt`** ⭐ **推荐使用**
   - 精确版本，与你的本地环境100%一致
   - Python 3.11.5
   - PaddlePaddle 3.2.1
   - NumPy 1.26.4
   - OpenCV 4.11.0.86
   - Pymunk 6.5.0

2. **`requirements_flexible.txt`**
   - 兼容版本范围
   - 适合不同云平台

3. **`requirements_full.txt`**
   - 完整的pip freeze输出
   - 包含所有依赖和子依赖

### 🚀 自动化脚本

4. **`setup_cloud.sh`**
   - 云端一键安装脚本
   - 自动检测CPU/GPU
   - 使用国内镜像加速

5. **`export_env.py`**
   - 导出本地环境工具
   - 可重新生成配置文件

6. **`verify_env.py`**
   - 环境验证工具
   - 检查所有依赖是否正确安装

### 📚 文档

7. **`CLOUD_SYNC_GUIDE.md`**
   - 完整的云端同步指南
   - Google Colab使用教程
   - AutoDL/阿里云/腾讯云部署教程
   - 常见问题解决方案

---

## 🎯 快速使用

### 场景1: 使用Google Colab (推荐新手)

```python
# 在Colab中新建笔记本

# 1. 上传项目文件或从GitHub克隆
!git clone https://github.com/your-username/DQN_FruitMerger.git
%cd DQN_FruitMerger

# 2. 安装精确版本的依赖
!bash setup_cloud.sh gpu

# 3. 验证环境
!python verify_env.py

# 4. 开始训练
!python run_training.py train --iterations 20

# 5. 可视化结果
!python visualize_results.py

# 6. 下载结果
from google.colab import files
!zip -r results.zip weights/alphazero/ *.png
files.download('results.zip')
```

### 场景2: 使用云服务器 (推荐长时间训练)

```bash
# 步骤1: 上传项目到云服务器
# 在本地执行:
cd /Users/ycy/Downloads
scp -r DQN_FruitMerger user@your-server:/workspace/

# 步骤2: SSH连接到服务器
ssh user@your-server

# 步骤3: 配置环境
cd /workspace/DQN_FruitMerger
bash setup_cloud.sh gpu  # 如果有GPU
# 或
bash setup_cloud.sh cpu  # 如果只有CPU

# 步骤4: 验证环境
python verify_env.py

# 步骤5: 后台训练
nohup python run_training.py train \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    > train.log 2>&1 &

# 步骤6: 监控进度
tail -f train.log

# 或查看GPU使用情况
watch -n 1 nvidia-smi

# 步骤7: 下载结果到本地
# 在本地执行:
scp -r user@your-server:/workspace/DQN_FruitMerger/weights/alphazero ./weights/
```

### 场景3: 本地继续训练

```bash
# 本地已经配置好环境，直接使用
cd /Users/ycy/Downloads/DQN_FruitMerger

# 验证环境
python verify_env.py

# 继续训练
python run_training.py train --resume 2 --iterations 10

# 可视化
python visualize_results.py
```

---

## 📊 当前环境信息

### 本地环境 (macOS)
- Python: 3.11.5
- PaddlePaddle: 3.2.1 (CPU)
- NumPy: 1.26.4
- OpenCV: 4.8.1 / 4.11.0.86
- Pymunk: 6.5.0
- 状态: ✅ 已验证通过

### 训练进度
- 已完成: 2轮迭代
- 评估分数: 136.2 → 104.3
- 已生成权重:
  - `weights/alphazero/iter_1.pdparams`
  - `weights/alphazero/iter_2.pdparams`
  - `weights/alphazero/history.json`

---

## 🔍 版本同步检查清单

部署到云端后，请按此清单验证：

- [ ] 上传项目文件到云端
- [ ] 运行 `bash setup_cloud.sh gpu` (或cpu)
- [ ] 运行 `python verify_env.py` 检查环境
- [ ] 检查PaddlePaddle版本: `python -c "import paddle; print(paddle.__version__)"`
- [ ] 快速测试: `python run_training.py train --quick`
- [ ] 检查GPU可用: `nvidia-smi` (如果有GPU)
- [ ] 查看生成的checkpoint: `ls -lh weights/alphazero/`

---

## 🛠️ 常用命令速查

### 环境管理
```bash
# 重新生成版本文件
python export_env.py

# 验证环境
python verify_env.py

# 云端安装 (GPU)
bash setup_cloud.sh gpu

# 云端安装 (CPU)
bash setup_cloud.sh cpu
```

### 训练相关
```bash
# 快速测试 (5-10分钟)
python run_training.py train --quick

# 标准训练 (10-20小时)
python run_training.py train --iterations 10 --games 30 --simulations 100

# 完整训练 (30-40小时)
python run_training.py train --iterations 20 --games 50 --simulations 200

# 继续训练
python run_training.py train --resume 2 --iterations 10

# 后台训练
nohup python run_training.py train --iterations 20 > train.log 2>&1 &
```

### 可视化和评估
```bash
# 查看训练摘要
python visualize_results.py --summary-only

# 生成可视化图表
python visualize_results.py

# 评估模型
python run_training.py evaluate --model-path weights/alphazero/iter_20.pdparams

# 评估并可视化
python evaluate_model.py --model-path weights/alphazero/iter_20.pdparams --visualize
```

### 文件传输
```bash
# 上传到云端
scp -r /Users/ycy/Downloads/DQN_FruitMerger user@server:/workspace/

# 从云端下载
scp -r user@server:/workspace/DQN_FruitMerger/weights/alphazero ./weights/

# 同步(推荐)
rsync -avz --exclude '__pycache__' ./ user@server:/workspace/DQN_FruitMerger/
```

---

## 📖 详细文档

- **训练使用**: 查看 `QUICK_START_GUIDE.md`
- **云端部署**: 查看 `CLOUD_SYNC_GUIDE.md`
- **AlphaZero原理**: 查看 `README_ALPHAZERO.md`
- **DQN原理**: 查看 `README.md`

---

## 💡 最佳实践

### 推荐工作流程

```
1. 本地开发和测试
   ↓
2. 导出环境版本 (python export_env.py)
   ↓
3. 上传到云端
   ↓
4. 云端验证环境 (python verify_env.py)
   ↓
5. 云端训练 (后台运行)
   ↓
6. 下载结果到本地
   ↓
7. 本地可视化和分析
   ↓
8. 继续训练或调整参数
```

### 版本管理建议

1. **代码**: 使用Git管理
2. **大文件**: 使用SCP/rsync传输
3. **依赖**: 锁定精确版本
4. **权重**: 定期备份checkpoint

---

## 🆘 遇到问题？

### 1. 版本不一致
```bash
# 重新安装精确版本
pip install -r requirements_exact.txt --force-reinstall
```

### 2. GPU不可用
```bash
# 检查CUDA
nvidia-smi

# 重装GPU版本
pip uninstall paddlepaddle paddlepaddle-gpu -y
pip install paddlepaddle-gpu==3.2.1
```

### 3. 依赖冲突
```bash
# 使用完整环境
pip install -r requirements_full.txt

# 或使用灵活版本
pip install -r requirements_flexible.txt
```

### 4. 训练中断
```bash
# 从最后的checkpoint继续
python run_training.py train --resume <最后的迭代次数>
```

---

## 📞 获取帮助

- 运行 `python verify_env.py` 检查环境
- 查看 `CLOUD_SYNC_GUIDE.md` 详细文档
- 运行 `python run_training.py --help` 查看命令帮助

---

**环境同步配置完成！现在你可以安心地在云端训练了 🚀**

下一步:
1. 选择一个云平台 (推荐Google Colab或AutoDL)
2. 按照上面的快速使用指南部署
3. 开始训练!
