"""
快速训练脚本 - 训练一个基础DQN模型
"""
import os
import numpy as np
from DQN import Agent, build_model, ReplayMemory, run_episode, compare_with_random
from GameInterface import GameInterface

# 减少训练规模，加速训练
MEMORY_SIZE = 10000
MEMORY_WARMUP_SIZE = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 训练参数
MAX_EPISODE = 500  # 减少到500局
EVALUATE_INTERVAL = 50  # 每50局评估一次

FINAL_PARAM_PATH = "final.pdparams"

def quick_train():
    print("=" * 60)
    print("🎮 快速训练DQN水果合成AI")
    print("=" * 60)

    # 初始化环境
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    print(f"特征维度: {feature_dim}, 动作数: {action_dim}")

    env = GameInterface()
    memory = ReplayMemory(MEMORY_SIZE)

    # 创建智能体
    e_greed = 0.9  # 初期多探索
    e_greed_decrement = 2e-6
    agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)

    # 检查是否有已有模型
    if os.path.exists(FINAL_PARAM_PATH):
        print(f"⚠️  发现已存在的模型: {FINAL_PARAM_PATH}")
        response = input("是否继续训练？(y/n): ")
        if response.lower() != 'y':
            print("训练取消")
            return

    # 预热经验池
    print(f"\n📦 预热经验池 (目标: {MEMORY_WARMUP_SIZE} 条经验)...")
    while len(memory) < MEMORY_WARMUP_SIZE:
        run_episode(env, agent, memory, -1)
        if len(memory) % 200 == 0:
            print(f"   已收集 {len(memory)} / {MEMORY_WARMUP_SIZE} 条经验")

    print(f"✅ 经验池预热完成，共 {len(memory)} 条经验\n")

    # 开始训练
    print(f"🚀 开始训练 (共 {MAX_EPISODE} 局)...")
    print("-" * 60)

    best_score = 0

    for episode_id in range(0, MAX_EPISODE + 1):
        total_reward = run_episode(env, agent, memory, episode_id)

        # 定期评估
        if episode_id % EVALUATE_INTERVAL == 0:
            print(f"\n📊 Episode {episode_id}/{MAX_EPISODE}")
            print(f"   ε-greedy: {agent.e_greed:.4f}")
            print(f"   最近奖励: {total_reward:.1f}")

            # 与随机agent比较
            compare_with_random(env, agent, action_dim)

            # 快速测试当前性能
            test_score, _ = evaluate_quick(env, agent)
            print(f"   测试分数: {test_score:.1f}")

            if test_score > best_score:
                best_score = test_score
                print(f"   🏆 新最佳分数!")

            print("-" * 60)

        # 显示训练进度
        elif episode_id % 10 == 0:
            progress = episode_id / MAX_EPISODE * 100
            print(f"[{progress:5.1f}%] Episode {episode_id:4d}, "
                  f"Reward: {total_reward:6.1f}, "
                  f"ε: {agent.e_greed:.4f}", end='\r')

    print("\n")

    # 保存模型
    import paddle
    paddle.save(agent.policy_net.state_dict(), FINAL_PARAM_PATH)
    print(f"✅ 模型已保存到: {FINAL_PARAM_PATH}")
    print(f"🎯 最佳测试分数: {best_score:.1f}")
    print("\n现在可以运行 'python3 AIPlay.py' 观看AI玩游戏!")

def evaluate_quick(env, agent):
    """快速评估（单局）"""
    env.reset(seed=12345)
    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)
    rewards_sum = 0

    while alive:
        action = agent.predict(feature)
        feature, reward, alive = env.next(action)
        rewards_sum += np.sum(reward)

    return env.game.score, rewards_sum

if __name__ == "__main__":
    import sys

    # 检查是否安装了paddle
    try:
        import paddle
        print(f"PaddlePaddle 版本: {paddle.__version__}")
    except ImportError:
        print("❌ 错误: 需要安装 PaddlePaddle")
        print("请运行: pip install paddlepaddle")
        sys.exit(1)

    try:
        quick_train()
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        response = input("是否保存当前模型? (y/n): ")
        if response.lower() == 'y':
            import paddle
            paddle.save(agent.policy_net.state_dict(), FINAL_PARAM_PATH)
            print(f"✅ 模型已保存")
