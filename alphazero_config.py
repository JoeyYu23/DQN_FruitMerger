"""
AlphaZero配置文件 - 统一所有模块的参数
"""

class AlphaZeroConfig:
    """AlphaZero全局配置"""

    # ============================================
    # Grid尺寸 (关键！所有模块必须一致)
    # ============================================
    # 使用GameInterface的特征尺寸
    GRID_HEIGHT = 20  # GameInterface.FEATURE_MAP_HEIGHT
    GRID_WIDTH = 16   # GameInterface.FEATURE_MAP_WIDTH

    # ============================================
    # 网络参数
    # ============================================
    INPUT_CHANNELS = 13  # 11个水果 + 1个当前水果 + 1个高度
    NUM_ACTIONS = 16     # GameInterface.ACTION_NUM
    HIDDEN_CHANNELS = 64

    # ============================================
    # MCTS参数
    # ============================================
    C_PUCT = 1.5
    NUM_SIMULATIONS = 200
    TEMPERATURE = 1.0
    TEMPERATURE_THRESHOLD = 30  # 前N步使用温度采样
    ADD_DIRICHLET_NOISE = True
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25

    # ============================================
    # Self-Play参数
    # ============================================
    GAMES_PER_ITERATION = 50
    SCORE_NORMALIZE_SCALE = 1000.0  # 用于tanh归一化

    # ============================================
    # 训练参数
    # ============================================
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 32
    EPOCHS_PER_ITERATION = 5
    NUM_ITERATIONS = 20

    # ============================================
    # 评估参数
    # ============================================
    EVAL_GAMES = 10
    EVAL_SIMULATIONS = 200

    # ============================================
    # 路径配置
    # ============================================
    CHECKPOINT_DIR = "weights/alphazero"
    LOG_DIR = "logs"
