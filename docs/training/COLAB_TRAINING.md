# 🚀 Google Colab 训练代码 - 直接复制粘贴运行

## 使用方法
1. 打开 [Google Colab](https://colab.research.google.com/)
2. 新建笔记本
3. 依次复制下面的代码块到Colab的cell中运行

---

## 📦 Step 1: 安装依赖 (约2-3分钟)

```python
# Cell 1: 安装所需依赖
!pip install -q paddlepaddle-gpu==3.2.1
!pip install -q numpy==1.26.4 opencv-python pymunk matplotlib tqdm psutil

print("✅ 依赖安装完成！")
```

---

## 📁 Step 2: 上传项目文件 (2选1)

### 方式A: 从GitHub克隆 (如果你已推送到GitHub)

```python
# Cell 2A: 从GitHub克隆项目
!git clone https://github.com/你的用户名/DQN_FruitMerger.git
%cd DQN_FruitMerger

print("✅ 项目克隆完成！")
```

### 方式B: 手动上传文件

```python
# Cell 2B: 手动上传项目文件
from google.colab import files
import zipfile
import os

print("📤 请上传项目ZIP文件...")
uploaded = files.upload()

# 解压
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"✅ 已解压: {filename}")

# 进入项目目录
%cd DQN_FruitMerger

print("✅ 文件上传完成！")
```

---

## ✅ Step 3: 验证环境 (约10秒)

```python
# Cell 3: 验证环境配置
!python verify_env.py
```

---

## 🎮 Step 4: 开始训练 (选择配置)

### 配置A: 快速测试 (约10-15分钟)

```python
# Cell 4A: 快速测试训练
!python TrainAlphaZero.py \
    --iterations 2 \
    --games 10 \
    --simulations 50 \
    --batch-size 16 \
    --epochs 3 \
    --eval-games 5

print("✅ 快速测试完成！")
```

### 配置B: 标准训练 (约3-4小时，推荐)

```python
# Cell 4B: 标准训练配置
!python TrainAlphaZero.py \
    --iterations 10 \
    --games 30 \
    --simulations 100 \
    --batch-size 32 \
    --epochs 5 \
    --eval-games 10

print("✅ 标准训练完成！")
```

### 配置C: 完整训练 (约8-10小时，最佳效果)

```python
# Cell 4C: 完整训练配置
!python TrainAlphaZero.py \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    --batch-size 64 \
    --epochs 5 \
    --eval-games 10

print("✅ 完整训练完成！")
```

---

## 📊 Step 5: 可视化训练结果

```python
# Cell 5: 查看训练摘要
!python visualize_results.py --summary-only
```

```python
# Cell 6: 生成可视化图表
!python visualize_results.py --no-show

# 显示图表
from IPython.display import Image, display
display(Image('training_visualization.png'))
```

---

## 💾 Step 6: 下载训练结果

```python
# Cell 7: 打包并下载结果
!zip -r training_results.zip \
    weights/alphazero/*.pdparams \
    weights/alphazero/history.json \
    *.png \
    *.log

from google.colab import files
files.download('training_results.zip')

print("✅ 结果已打包下载！")
```

---

## 🔍 额外功能

### 监控GPU使用情况

```python
# 查看GPU信息
!nvidia-smi
```

### 查看训练历史数据

```python
# 查看训练历史JSON
import json

with open('weights/alphazero/history.json', 'r') as f:
    history = json.load(f)

print("训练迭代:", history['iterations'])
print("评估分数:", history['eval_scores'])
print("训练Loss:", history['train_losses'])
```

### 继续训练

```python
# 如果训练中断，从上次继续
!python TrainAlphaZero.py \
    --resume 10 \
    --iterations 20 \
    --games 50 \
    --simulations 200 \
    --batch-size 64 \
    --epochs 5 \
    --eval-games 10
```

### 评估已训练的模型

```python
# 评估特定模型
!python evaluate_model.py \
    --model-path weights/alphazero/iter_20.pdparams \
    --num-games 20 \
    --simulations 200

# 显示得分分布图
from IPython.display import Image, display
display(Image('score_distribution.png'))
```

---

## ⚠️ 重要提示

1. **Colab会话限制**: 免费版12小时后会断开，建议：
   - 定期运行下载代码保存checkpoint
   - 使用`--resume`参数继续训练

2. **GPU加速**: 确保启用GPU
   - 菜单: 运行时 → 更改运行时类型 → GPU

3. **磁盘空间**: 训练会生成约500MB-1GB文件
   - 定期清理不需要的checkpoint

4. **保存进度**: 每完成几轮迭代就下载一次
   ```python
   !zip -r checkpoint_iter10.zip weights/alphazero/
   files.download('checkpoint_iter10.zip')
   ```

---

## 📈 预期训练时间 (GPU)

| 配置 | 迭代次数 | 预计时间 | 最终分数 |
|------|---------|---------|---------|
| 快速测试 | 2 | 10-15分钟 | 100-150 |
| 标准训练 | 10 | 3-4小时 | 500-800 |
| 完整训练 | 20 | 8-10小时 | 1000-1500 |

---

## 🎯 完整运行顺序

```
Cell 1: 安装依赖
  ↓
Cell 2: 上传项目 (选A或B)
  ↓
Cell 3: 验证环境
  ↓
Cell 4: 开始训练 (选A/B/C)
  ↓
Cell 5-6: 可视化结果
  ↓
Cell 7: 下载结果
```

---

**祝训练顺利！🚀**

有问题随时查看 `CLOUD_SYNC_GUIDE.md` 或 `QUICK_START_GUIDE.md`
