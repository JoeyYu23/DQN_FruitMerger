"""
可视化训练过程
从training_metrics.json读取数据，生成训练曲线图
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import uniform_filter1d

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

LOG_FILE = "training_metrics.json"
OUTPUT_DIR = "output"


def smooth_curve(data, window_size=20):
    """使用移动平均平滑曲线"""
    if len(data) < window_size:
        return data
    return uniform_filter1d(data, size=window_size, mode='nearest')


def plot_training_curves(log_file=LOG_FILE):
    """绘制训练曲线"""

    # 读取训练日志
    with open(log_file, 'r') as f:
        data = json.load(f)

    training = data['training']
    validation = data['validation']
    test = data.get('test', [])

    # 提取训练数据
    train_episodes = [x['episode'] for x in training]
    train_scores = [x['score'] for x in training]
    train_rewards = [x['reward'] for x in training]
    train_losses = [x['mean_loss'] for x in training]
    train_epsilon = [x['epsilon'] for x in training]

    # 提取验证数据
    val_episodes = [x['episode'] for x in validation]
    val_scores = [x['mean_score'] for x in validation]
    val_scores_std = [x['std_score'] for x in validation]
    val_rewards = [x['mean_reward'] for x in validation]
    val_rewards_std = [x['std_reward'] for x in validation]

    # 平滑训练曲线
    train_scores_smooth = smooth_curve(train_scores, window_size=50)
    train_rewards_smooth = smooth_curve(train_rewards, window_size=50)
    train_losses_smooth = smooth_curve(train_losses, window_size=50)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN训练过程可视化', fontsize=16, fontweight='bold')

    # 1. 分数对比（训练集 vs 验证集）
    ax1 = axes[0, 0]
    ax1.plot(train_episodes, train_scores, alpha=0.3, color='blue', linewidth=0.5, label='训练分数（原始）')
    ax1.plot(train_episodes, train_scores_smooth, color='blue', linewidth=2, label='训练分数（平滑）')
    ax1.plot(val_episodes, val_scores, color='red', linewidth=2, marker='o', markersize=4, label='验证分数')
    ax1.fill_between(val_episodes,
                      np.array(val_scores) - np.array(val_scores_std),
                      np.array(val_scores) + np.array(val_scores_std),
                      color='red', alpha=0.2, label='验证标准差')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('游戏分数', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. 奖励对比（训练集 vs 验证集）
    ax2 = axes[0, 1]
    ax2.plot(train_episodes, train_rewards, alpha=0.3, color='green', linewidth=0.5, label='训练奖励（原始）')
    ax2.plot(train_episodes, train_rewards_smooth, color='green', linewidth=2, label='训练奖励（平滑）')
    ax2.plot(val_episodes, val_rewards, color='orange', linewidth=2, marker='s', markersize=4, label='验证奖励')
    ax2.fill_between(val_episodes,
                      np.array(val_rewards) - np.array(val_rewards_std),
                      np.array(val_rewards) + np.array(val_rewards_std),
                      color='orange', alpha=0.2, label='验证标准差')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('累积奖励', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 3. Loss曲线
    ax3 = axes[1, 0]
    ax3.plot(train_episodes, train_losses, alpha=0.3, color='purple', linewidth=0.5, label='Loss（原始）')
    ax3.plot(train_episodes, train_losses_smooth, color='purple', linewidth=2, label='Loss（平滑）')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('训练损失', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # 4. Epsilon衰减曲线
    ax4 = axes[1, 1]
    ax4.plot(train_episodes, train_epsilon, color='brown', linewidth=2, label='ε-greedy')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Epsilon', fontsize=12)
    ax4.set_title('探索率（ε-greedy）', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    output_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 训练曲线已保存: {output_path}")

    # 显示图表
    plt.show()

    # 打印统计信息
    print("\n" + "=" * 70)
    print("训练统计信息")
    print("=" * 70)

    print(f"\n训练集（最后100个episode平均）:")
    print(f"  分数: {np.mean(train_scores[-100:]):.1f} ± {np.std(train_scores[-100:]):.1f}")
    print(f"  奖励: {np.mean(train_rewards[-100:]):.1f} ± {np.std(train_rewards[-100:]):.1f}")
    print(f"  Loss: {np.mean(train_losses[-100:]):.4f}")

    print(f"\n验证集（最终）:")
    print(f"  分数: {val_scores[-1]:.1f} ± {val_scores_std[-1]:.1f}")
    print(f"  奖励: {val_rewards[-1]:.1f} ± {val_rewards_std[-1]:.1f}")

    if test:
        test_score = test[0]['mean_score']
        test_score_std = test[0]['std_score']
        test_reward = test[0]['mean_reward']
        test_reward_std = test[0]['std_reward']
        print(f"\n测试集:")
        print(f"  分数: {test_score:.1f} ± {test_score_std:.1f}")
        print(f"  奖励: {test_reward:.1f} ± {test_reward_std:.1f}")

    print("\n" + "=" * 70)


def plot_comparison(log_file=LOG_FILE):
    """绘制训练集、验证集、测试集的对比图"""

    with open(log_file, 'r') as f:
        data = json.load(f)

    validation = data['validation']
    test = data.get('test', [])

    val_episodes = [x['episode'] for x in validation]
    val_scores = [x['mean_score'] for x in validation]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(val_episodes, val_scores, color='blue', linewidth=2, marker='o', label='验证集')

    if test:
        test_episode = test[0]['episode']
        test_score = test[0]['mean_score']
        ax.axhline(y=test_score, color='red', linestyle='--', linewidth=2, label=f'测试集 ({test_score:.1f})')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('验证集和测试集性能对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, 'val_test_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_path}")

    plt.show()


if __name__ == "__main__":
    import os
    import sys

    if not os.path.exists(LOG_FILE):
        print(f"❌ 错误：找不到训练日志文件 {LOG_FILE}")
        print(f"请先运行 train_with_logging.py 进行训练")
        sys.exit(1)

    print("=" * 70)
    print("可视化训练过程")
    print("=" * 70)
    print()

    plot_training_curves()
    print()
    plot_comparison()

    print("\n所有图表已生成！")
