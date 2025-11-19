#!/bin/bash
# 训练监控脚本 - 实时查看训练进度

echo "🔍 DQN训练监控"
echo "============================================"
echo ""

# 检查训练是否在运行
if pgrep -f "train_5000.py" > /dev/null; then
    echo "✅ 训练正在运行"
    echo ""

    # 显示最新的进度信息
    echo "📊 最新进度:"
    echo "--------------------------------------------"
    tail -20 training_5000.log 2>/dev/null || echo "日志文件尚未创建"

    echo ""
    echo "--------------------------------------------"
    echo "💡 提示:"
    echo "  - 按 Ctrl+C 退出监控（不会停止训练）"
    echo "  - 训练日志: training_5000.log"
    echo "  - 检查点: weights/checkpoint_ep*.pdparams"
    echo "  - 最佳模型: weights/best_model.pdparams"
    echo ""
    echo "🔄 每5秒自动刷新..."
    echo ""

    # 持续监控
    while pgrep -f "train_5000.py" > /dev/null; do
        sleep 5
        clear
        echo "🔍 DQN训练监控 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================"
        echo ""
        tail -15 training_5000.log 2>/dev/null || echo "等待日志..."
        echo ""
        echo "--------------------------------------------"
        echo "💡 按 Ctrl+C 退出监控（不会停止训练）"
    done

    echo ""
    echo "✅ 训练已完成！"

else
    echo "❌ 训练未在运行"
    echo ""
    echo "要启动训练，运行:"
    echo "  python3 train_5000.py"
fi
