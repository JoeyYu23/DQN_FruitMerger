"""
5000轮DQN训练 - 带实时进度显示
"""
import os
import json
import numpy as np
import paddle
from datetime import datetime, timedelta
import time
from DQN import (
    Agent, build_model, ReplayMemory, MEMORY_SIZE, MEMORY_WARMUP_SIZE,
    BATCH_SIZE, LEARNING_RATE, GAMMA, set_global_seed
)
from GameInterface import GameInterface

# ==================== 训练配置 ====================
TRAINING_SEED = 42
MAX_EPISODES = 5000
EVAL_INTERVAL = 100  # 每100轮评估一次（5000轮太多，25轮太频繁）
EVAL_EPISODES = 10
LOG_FILE = "training_metrics_5000.json"
MODEL_SAVE_DIR = "weights"
CHECKPOINT_INTERVAL = 500  # 每500轮保存checkpoint

# 验证集和测试集的固定种子
VAL_SEEDS = list(range(10000, 10000 + EVAL_EPISODES))
TEST_SEEDS = list(range(20000, 20000 + EVAL_EPISODES))


class ProgressTracker:
    """训练进度跟踪器"""

    def __init__(self, total_episodes):
        self.total_episodes = total_episodes
        self.start_time = None
        self.episode_times = []
        self.last_print_time = time.time()

    def start(self):
        """开始训练"""
        self.start_time = time.time()

    def update(self, episode, metrics):
        """更新进度"""
        current_time = time.time()

        # 记录episode时间
        if len(self.episode_times) > 0:
            episode_time = current_time - self.last_update_time
            self.episode_times.append(episode_time)
            # 只保留最近100个episode的时间
            if len(self.episode_times) > 100:
                self.episode_times.pop(0)

        self.last_update_time = current_time

        # 计算统计信息
        elapsed = current_time - self.start_time
        progress = episode / self.total_episodes

        # 估计剩余时间
        if len(self.episode_times) > 0:
            avg_episode_time = np.mean(self.episode_times)
            remaining_episodes = self.total_episodes - episode
            eta_seconds = avg_episode_time * remaining_episodes
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = "计算中..."

        # 只在满足条件时打印（避免刷屏）
        should_print = (
            episode % 10 == 0 or  # 每10个episode
            current_time - self.last_print_time > 5  # 或每5秒
        )

        if should_print:
            self.print_progress(episode, progress, elapsed, eta, metrics)
            self.last_print_time = current_time

    def print_progress(self, episode, progress, elapsed, eta, metrics):
        """打印进度信息"""
        # 进度条
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # 格式化时间
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # 清除当前行并打印
        print(f'\r', end='')  # 回到行首
        print(
            f"[{bar}] {progress*100:5.1f}% | "
            f"Ep {episode:4d}/{self.total_episodes} | "
            f"⏱️ {elapsed_str} | "
            f"⏳ ETA {eta} | "
            f"Score {metrics['score']:3.0f} | "
            f"Loss {metrics['mean_loss']:6.1f} | "
            f"ε {metrics['epsilon']:.4f}",
            end='', flush=True
        )

    def print_eval(self, episode, train_metrics, val_metrics):
        """打印评估信息"""
        print()  # 换行
        print("=" * 100)
        print(f"📊 Episode {episode}/{self.total_episodes} - 评估结果")
        print("-" * 100)

        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"⏰ 当前时间: {current_time} | 已训练时间: {elapsed_str}")
        print()

        print(f"训练集 (本轮):")
        print(f"  分数: {train_metrics['score']:6.1f} | "
              f"奖励: {train_metrics['reward']:8.1f} | "
              f"Loss: {train_metrics['mean_loss']:8.2f} | "
              f"步数: {train_metrics['steps']:4d}")

        print(f"验证集 (10局平均):")
        print(f"  分数: {val_metrics['mean_score']:6.1f} ± {val_metrics['std_score']:5.1f} | "
              f"奖励: {val_metrics['mean_reward']:8.1f} ± {val_metrics['std_reward']:5.1f} | "
              f"最高: {val_metrics['max_score']:3d} | "
              f"最低: {val_metrics['min_score']:3d}")

        print(f"探索率: ε = {train_metrics['epsilon']:.6f}")
        print("=" * 100)
        print()


def evaluate_agent(env, agent, seeds, num_episodes=None):
    """评估agent"""
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
    """运行一个训练episode"""
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


def save_checkpoint(agent, episode, filename):
    """保存训练checkpoint"""
    checkpoint = {
        'episode': episode,
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epsilon': agent.e_greed,
        'global_step': agent.global_step,
    }
    paddle.save(checkpoint, filename)


def main():
    print("=" * 100)
    print("🚀 DQN训练 - 5000轮完整训练")
    print("=" * 100)

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

    print(f"\n📋 训练配置:")
    print(f"  特征维度: {feature_dim}")
    print(f"  动作空间: {action_dim}")
    print(f"  训练种子: {TRAINING_SEED}")
    print(f"  最大训练轮数: {MAX_EPISODES}")
    print(f"  评估间隔: 每{EVAL_INTERVAL}轮")
    print(f"  Checkpoint间隔: 每{CHECKPOINT_INTERVAL}轮")
    print()

    # 预热经验池
    print(f"🔥 预热经验池 (目标: {MEMORY_WARMUP_SIZE})...")
    warmup_start = time.time()
    warmup_count = 0

    while len(memory) < MEMORY_WARMUP_SIZE:
        run_training_episode(env, agent, memory)
        warmup_count += 1
        if warmup_count % 100 == 0:
            print(f"  预热进度: {len(memory):5d}/{MEMORY_WARMUP_SIZE} ({len(memory)/MEMORY_WARMUP_SIZE*100:.1f}%)", end='\r')

    warmup_time = time.time() - warmup_start
    print(f"\n✅ 经验池预热完成，共 {len(memory)} 条经验 (耗时: {warmup_time:.1f}秒)\n")

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
            'checkpoint_interval': CHECKPOINT_INTERVAL,
        },
        'training': [],
        'validation': [],
        'test': []
    }

    # 创建进度跟踪器
    progress = ProgressTracker(MAX_EPISODES)
    progress.start()

    best_val_score = 0

    print("🎯 开始训练...")
    print("-" * 100)
    print()

    for episode in range(MAX_EPISODES + 1):
        # 训练一个episode
        episode_metrics = run_training_episode(env, agent, memory)

        # 记录训练指标
        training_log['training'].append({
            'episode': episode,
            **episode_metrics
        })

        # 更新进度
        progress.update(episode, episode_metrics)

        # 定期评估
        if episode % EVAL_INTERVAL == 0:
            # 验证集评估
            val_metrics = evaluate_agent(env, agent, VAL_SEEDS, EVAL_EPISODES)
            training_log['validation'].append({
                'episode': episode,
                **val_metrics
            })

            # 打印评估信息
            progress.print_eval(episode, episode_metrics, val_metrics)

            # 保存最佳模型
            if val_metrics['mean_score'] > best_val_score:
                best_val_score = val_metrics['mean_score']
                best_model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pdparams")
                paddle.save(agent.policy_net.state_dict(), best_model_path)
                print(f"🏆 新最佳验证分数: {best_val_score:.1f} - 模型已保存到 {best_model_path}\n")

        # 定期保存checkpoint
        if episode > 0 and episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, f"checkpoint_ep{episode}.pdparams")
            save_checkpoint(agent, episode, checkpoint_path)
            print(f"\n💾 Checkpoint已保存: {checkpoint_path}\n")

    print("\n")
    print("=" * 100)

    # 最终测试集评估
    print("🎓 最终测试集评估...")
    test_metrics = evaluate_agent(env, agent, TEST_SEEDS, len(TEST_SEEDS))
    training_log['test'].append({
        'episode': MAX_EPISODES,
        **test_metrics
    })

    print(f"\n测试集结果:")
    print(f"  分数: {test_metrics['mean_score']:.1f} ± {test_metrics['std_score']:.1f}")
    print(f"  奖励: {test_metrics['mean_reward']:.1f} ± {test_metrics['std_reward']:.1f}")
    print(f"  最高: {test_metrics['max_score']} | 最低: {test_metrics['min_score']}")

    # 保存最终模型
    final_model_path = "final_5000.pdparams"
    paddle.save(agent.policy_net.state_dict(), final_model_path)
    print(f"\n✅ 最终模型已保存: {final_model_path}")

    # 保存训练日志
    with open(LOG_FILE, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"✅ 训练日志已保存: {LOG_FILE}")

    # 训练总结
    total_time = time.time() - progress.start_time
    total_time_str = str(timedelta(seconds=int(total_time)))

    print("\n" + "=" * 100)
    print("🎉 训练完成！")
    print("=" * 100)
    print(f"总训练时间: {total_time_str}")
    print(f"平均每轮时间: {total_time/MAX_EPISODES:.2f}秒")
    print(f"最佳验证分数: {best_val_score:.1f}")
    print(f"最终测试分数: {test_metrics['mean_score']:.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
