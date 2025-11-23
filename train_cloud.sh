#!/bin/bash
# AlphaZero云端训练脚本

echo "========================================"
echo "AlphaZero Training Script"
echo "========================================"

# 配置参数
ITERATIONS=${1:-20}           # 迭代次数
GAMES=${2:-50}                # 每轮游戏数
SIMULATIONS=${3:-200}         # MCTS模拟次数
BATCH_SIZE=${4:-32}           # 批量大小
EPOCHS=${5:-5}                # 每轮epoch数
EVAL_GAMES=${6:-10}           # 评估游戏数

echo "Configuration:"
echo "  Iterations: $ITERATIONS"
echo "  Games per iteration: $GAMES"
echo "  MCTS simulations: $SIMULATIONS"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs per iteration: $EPOCHS"
echo "  Eval games: $EVAL_GAMES"
echo "========================================"

# 创建必要的目录
mkdir -p weights/alphazero
mkdir -p logs

# 设置日志文件
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""

# 运行训练（输出到日志和终端）
python TrainAlphaZero.py \
    --iterations $ITERATIONS \
    --games $GAMES \
    --simulations $SIMULATIONS \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --eval-games $EVAL_GAMES \
    --checkpoint-dir weights/alphazero \
    2>&1 | tee $LOG_FILE

echo ""
echo "========================================"
echo "Training completed!"
echo "Check log: $LOG_FILE"
echo "Models saved in: weights/alphazero"
echo "========================================"
