#!/bin/bash

# ========================================
# AlphaZero 云端部署 - 5分钟快速开始
# ========================================

echo "🚀 AlphaZero Cloud Deployment Guide"
echo "===================================="
echo ""

# 检查是否在项目目录
if [ ! -f "SuikaNet.py" ]; then
    echo "❌ 错误: 请在项目根目录运行此脚本"
    echo "   cd /Users/ycy/Downloads/DQN_FruitMerger"
    exit 1
fi

echo "📦 第1步: 打包代码"
echo "------------------------------------"
cd ..
tar -czf DQN_FruitMerger.tar.gz DQN_FruitMerger/ \
    --exclude='DQN_FruitMerger/weights' \
    --exclude='DQN_FruitMerger/output' \
    --exclude='DQN_FruitMerger/*.log'
echo "✅ 打包完成: DQN_FruitMerger.tar.gz"
ls -lh DQN_FruitMerger.tar.gz
cd DQN_FruitMerger

echo ""
echo "🌐 第2步: 选择云平台"
echo "------------------------------------"
echo "推荐方案 (从易到难):"
echo ""
echo "1. AutoDL (最推荐) ⭐⭐⭐⭐⭐"
echo "   - 超便宜: 1.5元/小时"
echo "   - 超简单: 网页直接上传"
echo "   - 注册: https://www.autodl.com"
echo ""
echo "2. Google Colab (免费) ⭐⭐⭐⭐"
echo "   - 完全免费"
echo "   - 浏览器使用"
echo "   - 网址: https://colab.research.google.com"
echo ""
echo "3. 阿里云/腾讯云 (稳定) ⭐⭐⭐"
echo "   - 国内快"
echo "   - 8元/小时"
echo ""

echo ""
echo "📋 第3步: 上传代码到云端"
echo "------------------------------------"
echo "压缩包位置: /Users/ycy/Downloads/DQN_FruitMerger.tar.gz"
echo ""
echo "AutoDL上传方式:"
echo "  1. 登录 AutoDL 控制台"
echo "  2. 创建实例后，点击 JupyterLab"
echo "  3. 点击上传按钮，选择 DQN_FruitMerger.tar.gz"
echo "  4. 在终端运行: tar -xzf DQN_FruitMerger.tar.gz"
echo ""
echo "或使用 scp 命令:"
echo "  scp -P [端口] DQN_FruitMerger.tar.gz root@[服务器IP]:/root/"
echo ""

echo ""
echo "🔧 第4步: 在云端运行以下命令"
echo "------------------------------------"
cat << 'EOF'

# 解压代码
cd /root
tar -xzf DQN_FruitMerger.tar.gz
cd DQN_FruitMerger

# 安装依赖
pip install -r requirements_alphazero.txt
pip install paddlepaddle-gpu

# 验证GPU
python -c "import paddle; print('GPU:', paddle.device.is_compiled_with_cuda())"

# 快速测试
python test_pipeline.py

# 开始训练！
nohup ./train_cloud.sh 20 50 200 32 5 10 > train.log 2>&1 &

# 查看进度
tail -f train.log

EOF

echo ""
echo "✅ 准备完成！"
echo "===================================="
echo ""
echo "下一步:"
echo "1. 访问云平台网站并注册"
echo "2. 创建GPU实例（推荐 Tesla T4 或 RTX 3090）"
echo "3. 上传 DQN_FruitMerger.tar.gz"
echo "4. 运行上面的命令"
echo "5. 等待8-15分钟完成训练"
echo "6. 下载结果并关闭实例"
echo ""
echo "💰 预计成本: 0.2-2元"
echo "⏱️  预计时间: 8-15分钟"
echo ""
echo "详细教程: cat CLOUD_DEPLOY_TUTORIAL.md"
echo ""
