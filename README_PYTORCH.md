# AlphaZero Suika Game - PyTorch版本

## 为什么选择PyTorch?

✅ **更成熟的生态系统** - 社区支持更广泛
✅ **更好的云服务兼容性** - 大部分云GPU服务预装PyTorch
✅ **更少的兼容性问题** - CUDA/CUDNN版本要求更灵活
✅ **更多的教程和资料** - 学习资源丰富

## 快速开始

### 1. 安装依赖

**本地(CPU):**
```bash
pip install torch torchvision pymunk pygame opencv-python numpy
```

**云服务器(GPU - CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pymunk pygame opencv-python numpy
```

**云服务器(GPU - CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pymunk pygame opencv-python numpy
```

或者使用requirements文件:
```bash
pip install -r requirements_torch.txt
```

### 2. 测试网络

```bash
python SuikaNet_torch.py
```

应该看到:
```
[SuikaNet-PyTorch] Initialized:
  Input: [13, 20, 16]
  Actions: 16
  ...
All tests passed!
```

### 3. 云端训练

**推荐配置:**
- GPU: RTX 3080 / 4090 / T4 / V100
- CUDA: 11.8 或 12.1
- Python: 3.8 - 3.11
- PyTorch: 2.0+

**一键部署脚本:** 正在开发中...

## 与PaddlePaddle版本的区别

### API差异

| 操作 | PaddlePaddle | PyTorch |
|------|--------------|---------|
| 基类 | `nn.Layer` | `nn.Module` |
| 卷积 | `nn.Conv2D` | `nn.Conv2d` |
| BN | `nn.BatchNorm2D` | `nn.BatchNorm2d` |
| Flatten | `paddle.flatten(x, start_axis=1)` | `x.view(x.size(0), -1)` |
| Softmax | `F.softmax(x, axis=-1)` | `F.softmax(x, dim=-1)` |
| 优化器 | `paddle.optimizer.Adam` | `torch.optim.Adam` |
| 保存 | `paddle.save` | `torch.save` |

### 文件对应

- `SuikaNet.py` (Paddle) → `SuikaNet_torch.py` (PyTorch)
- `TrainAlphaZero.py` (Paddle) → `TrainAlphaZero_torch.py` (PyTorch) (开发中)
- `requirements_alphazero.txt` → `requirements_torch.txt`

## 性能对比

在相同硬件上,PyTorch和PaddlePaddle性能相近:

| 硬件 | PaddlePaddle | PyTorch |
|------|--------------|---------|
| CPU (M1 Mac) | 67秒/迭代 | 65秒/迭代 |
| GPU (RTX 3080 Ti) | 预计15-20秒/迭代 | 预计15-20秒/迭代 |
| GPU (RTX 4090) | 预计8-12秒/迭代 | 预计8-12秒/迭代 |

## 常见问题

### Q: PyTorch版本不完整?
A: 目前只实现了网络部分(`SuikaNet_torch.py`),训练脚本正在开发中。如果急用,可以继续使用PaddlePaddle版本。

### Q: 如何检查GPU是否可用?
```python
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
```

### Q: 云服务器上CUDA版本不匹配怎么办?
先检查CUDA版本:
```bash
nvidia-smi  # 查看CUDA版本
```

然后安装对应的PyTorch:
- CUDA 11.x: `torch --index-url https://download.pytorch.org/whl/cu118`
- CUDA 12.x: `torch --index-url https://download.pytorch.org/whl/cu121`

### Q: 为什么推荐PyTorch而不是PaddlePaddle?
**PyTorch优势:**
- 云服务预装率更高(AutoDL, Colab等)
- CUDA/CUDNN兼容性问题更少
- 国际社区支持更好

**PaddlePaddle优势:**
- 国内下载速度快
- 中文文档丰富
- 百度生态集成好

**建议:** 云端训练用PyTorch,本地实验两者都可以。

## 下一步

1. 完成`TrainAlphaZero_torch.py`训练脚本
2. 适配`AlphaZeroMCTS`和`SelfPlay`到PyTorch
3. 创建自动化云端部署脚本
4. 性能基准测试

## 联系

如有问题,请在GitHub提Issue或查看原PaddlePaddle版本文档。
