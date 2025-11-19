"""
带详细日志记录的DQN训练脚本
记录训练过程中的各种指标，用于后续可视化分析
"""
import os
import json
import numpy as np
import paddle
from datetime import datetime
from DQN import (
    Agent, build_model, ReplayMemory, MEMORY_SIZE, MEMORY_WARMUP_SIZE,
    BATCH_SIZE, LEARNING_RATE, GAMMA, set_global_seed
)
from GameInterface import GameInterface
from PRNG import PRNG

# 训练配置
TRAINING_SEED = 42
MAX_EPISODES = 500  # 快速演示：500轮
EVAL_INTERVAL = 25  # 每25轮评估一次
EVAL_EPISODES = 10  # 每次评估10局
LOG_FILE = "training_metrics.json"
MODEL_SAVE_DIR = "weights"

# 验证集和测试集的固定种子
VAL_SEEDS = list(range(10000, 10000 + EVAL_EPISODES))
TEST_SEEDS = list(range(20000, 20000 + EVAL_EPISODES))


def evaluate_agent(env, agent, seeds, num_episodes=None):
    """
    在固定种子上评估agent

    Args:
        env: 游戏环境
        agent: 要评估的agent
        seeds: 评估使用的种子列表
        num_episodes: 评估局数（如果为None，使用seeds的长度）

    Returns:
        dict: 包含平均分数、奖励等统计信息
    """
    if num_episodes is None:
        num_episodes = len(seeds)

    scores = []
    rewards = []

    for i in range(num_episodes):
        seed = seeds[i]
        env.reset(seed=seed)

        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        episode_reward = 0
        while alive:
            action = agent.predict(feature)
            feature, reward, alive = env.next(action)
            episode_reward += np.sum(reward)

        scores.append(env.game.score)
        rewards.append(episode_reward)

    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
    }


def run_training_episode(env, agent, memory):
    """
    运行一个训练episode

    Returns:
        dict: 包含该episode的各种指标
    """
    env.reset()

    step = 0
    rewards_sum = 0
    losses = []

    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    while alive:
        step += 1

        action = agent.sample(feature)
        next_feature, reward, alive = env.next(action)

        reward = reward if alive else -1000
        memory.append((feature, action, reward, next_feature, alive))

        # 学习
        if len(memory) >= MEMORY_WARMUP_SIZE and agent.global_step % 1 == 0:
            batch = memory.sample(BATCH_SIZE)
            loss = agent.learn(*batch)
            losses.append(loss)

        rewards_sum += np.sum(reward)
        feature = next_feature
        agent.global_step += 1

    return {
        'score': int(env.game.score),
        'reward': float(rewards_sum),
        'steps': int(step),
        'mean_loss': float(np.mean(losses)) if losses else 0.0,
        'epsilon': float(agent.e_greed)
    }


def main():
    print("=" * 70)
    print("DQN训练 - 带详细指标记录")
    print("=" * 70)

    # 设置随机种子
    set_global_seed(TRAINING_SEED)

    # 初始化环境和agent
    env = GameInterface()
    feature_dim = env.FEATURE_MAP_HEIGHT * env.FEATURE_MAP_WIDTH * 2
    action_dim = env.ACTION_NUM

    memory = ReplayMemory(MEMORY_SIZE)
    agent = Agent(
        build_model,
        feature_dim,
        action_dim,
        e_greed=0.5,
        e_greed_decrement=1e-6
    )

    print(f"特征维度: {feature_dim}")
    print(f"动作空间: {action_dim}")
    print(f"训练种子: {TRAINING_SEED}")
    print(f"最大训练轮数: {MAX_EPISODES}")
    print(f"评估间隔: 每{EVAL_INTERVAL}轮")
    print()

    # 预热经验池
    print(f"预热经验池 (目标: {MEMORY_WARMUP_SIZE})...")
    warmup_count = 0
    while len(memory) < MEMORY_WARMUP_SIZE:
        run_training_episode(env, agent, memory)
        warmup_count += 1
        if warmup_count % 100 == 0:
            print(f"  预热进度: {len(memory)}/{MEMORY_WARMUP_SIZE}")
    print(f"✅ 经验池预热完成，共 {len(memory)} 条经验\n")

    # 训练数据记录
    training_log = {
        'config': {
            'seed': TRAINING_SEED,
            'max_episodes': MAX_EPISODES,
            'memory_size': MEMORY_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'gamma': GAMMA,
            'eval_interval': EVAL_INTERVAL,
        },
        'training': [],
        'validation': [],
        'test': []
    }

    # 开始训练
    print("开始训练...")
    print("-" * 70)

    best_val_score = 0

    for episode in range(MAX_EPISODES + 1):
        # 训练一个episode
        episode_metrics = run_training_episode(env, agent, memory)

        # 记录训练指标
        training_log['training'].append({
            'episode': episode,
            **episode_metrics
        })

        # 定期评估
        if episode % EVAL_INTERVAL == 0:
            # 验证集评估
            val_metrics = evaluate_agent(env, agent, VAL_SEEDS, EVAL_EPISODES)
            training_log['validation'].append({
                'episode': episode,
                **val_metrics
            })

            print(f"\nEpisode {episode}/{MAX_EPISODES}")
            print(f"  训练 - 分数: {episode_metrics['score']:.0f}, "
                  f"奖励: {episode_metrics['reward']:.1f}, "
                  f"Loss: {episode_metrics['mean_loss']:.4f}")
            print(f"  验证 - 分数: {val_metrics['mean_score']:.1f}±{val_metrics['std_score']:.1f}, "
                  f"奖励: {val_metrics['mean_reward']:.1f}±{val_metrics['std_reward']:.1f}")
            print(f"  ε-greedy: {agent.e_greed:.4f}")

            # 保存最佳模型
            if val_metrics['mean_score'] > best_val_score:
                best_val_score = val_metrics['mean_score']
                best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pdparams")
                paddle.save(agent.policy_net.state_dict(), best_model_path)
                print(f"  🏆 新最佳验证分数! 模型已保存")

            print("-" * 70)

        # 显示进度
        elif episode % 10 == 0:
            progress = episode / MAX_EPISODES * 100
            print(f"[{progress:5.1f}%] Episode {episode:4d}, "
                  f"Score: {episode_metrics['score']:3.0f}, "
                  f"Reward: {episode_metrics['reward']:6.1f}, "
                  f"Loss: {episode_metrics['mean_loss']:.4f}, "
                  f"ε: {agent.e_greed:.4f}",
                  end='\r')

    print("\n")

    # 最终测试集评估
    print("最终测试集评估...")
    test_metrics = evaluate_agent(env, agent, TEST_SEEDS, len(TEST_SEEDS))
    training_log['test'].append({
        'episode': MAX_EPISODES,
        **test_metrics
    })

    print(f"测试集结果 - 分数: {test_metrics['mean_score']:.1f}±{test_metrics['std_score']:.1f}, "
          f"奖励: {test_metrics['mean_reward']:.1f}±{test_metrics['std_reward']:.1f}")

    # 保存最终模型
    final_model_path = "final.pdparams"
    paddle.save(agent.policy_net.state_dict(), final_model_path)
    print(f"✅ 最终模型已保存: {final_model_path}")

    # 保存训练日志
    with open(LOG_FILE, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"✅ 训练日志已保存: {LOG_FILE}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print(f"最佳验证分数: {best_val_score:.1f}")
    print(f"最终测试分数: {test_metrics['mean_score']:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
