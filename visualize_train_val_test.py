"""
可视化训练、验证、测试的时间线
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib
import json

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def visualize_train_val_test_timeline():
    """绘制训练/验证/测试的时间线"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 读取实际数据
    with open('training_metrics.json', 'r') as f:
        data = json.load(f)

    max_episodes = data['config']['max_episodes']
    eval_interval = data['config']['eval_interval']

    # ==================== 子图1：训练流程时间线 ====================
    ax1.set_xlim(0, max_episodes + 50)
    ax1.set_ylim(0, 4)

    # 训练集：每个episode
    train_episodes = list(range(0, max_episodes + 1, 10))  # 每10个画一个点，避免太密集
    ax1.scatter(train_episodes, [1] * len(train_episodes),
               color='blue', s=20, alpha=0.5, label='训练集评估点')

    # 验证集：每25个episode
    val_episodes = list(range(0, max_episodes + 1, eval_interval))
    ax1.scatter(val_episodes, [2] * len(val_episodes),
               color='red', s=150, marker='o', label='验证集评估点', zorder=5)

    # 测试集：最后一次
    ax1.scatter([max_episodes], [3],
               color='green', s=300, marker='*', label='测试集评估点', zorder=5)

    # 添加箭头和说明
    ax1.annotate('训练每个Episode', xy=(250, 1), xytext=(250, 0.5),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax1.annotate(f'每{eval_interval}轮验证一次', xy=(250, 2), xytext=(250, 2.5),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.annotate('训练完成后测试一次', xy=(max_episodes, 3), xytext=(max_episodes - 100, 3.5),
                fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))

    # 设置
    ax1.set_xlabel('Episode', fontsize=14)
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['训练集\n(每局)', '验证集\n(每25局)', '测试集\n(最后)'], fontsize=12)
    ax1.set_title('训练/验证/测试时间线', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, axis='x', alpha=0.3)

    # ==================== 子图2：数据集统计对比 ====================
    datasets = ['训练集', '验证集', '测试集']

    # 评估次数
    eval_counts = [
        len(data['training']),
        len(data['validation']),
        len(data['test'])
    ]

    # 每次局数
    games_per_eval = [1, 10, 10]

    # 总局数
    total_games = [eval_counts[i] * games_per_eval[i] for i in range(3)]

    x = np.arange(len(datasets))
    width = 0.25

    bars1 = ax2.bar(x - width, eval_counts, width, label='评估次数', color='skyblue')
    bars2 = ax2.bar(x, games_per_eval, width, label='每次局数', color='orange')
    bars3 = ax2.bar(x + width, total_games, width, label='总局数', color='lightgreen')

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_xlabel('数据集类型', fontsize=14)
    ax2.set_ylabel('数量', fontsize=14)
    ax2.set_title('数据集统计对比', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = 'output/train_val_test_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时间线图已保存: {output_path}")

    plt.show()


def print_summary():
    """打印训练/验证/测试流程总结"""

    with open('training_metrics.json', 'r') as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("训练/验证/测试流程总结")
    print("=" * 80)

    print("\n【配置信息】")
    print(f"  全局随机种子: {data['config']['seed']}")
    print(f"  最大训练轮数: {data['config']['max_episodes']}")
    print(f"  验证评估间隔: 每{data['config']['eval_interval']}轮")

    print("\n【数据集划分】")
    print(f"\n  1️⃣  训练集 (Training Set)")
    print(f"     - 用途: 训练模型，更新网络参数")
    print(f"     - 频率: 每个episode都训练")
    print(f"     - 评估次数: {len(data['training'])}次")
    print(f"     - 种子策略: 随机（受全局种子42控制）")
    print(f"     - 探索策略: ε-greedy（有探索）")

    print(f"\n  2️⃣  验证集 (Validation Set)")
    print(f"     - 用途: 监控训练，选择最佳模型")
    print(f"     - 频率: 每{data['config']['eval_interval']}个episode评估一次")
    print(f"     - 评估次数: {len(data['validation'])}次")
    print(f"     - 每次局数: 10局")
    print(f"     - 种子策略: 固定种子 [10000-10009]")
    print(f"     - 探索策略: 纯贪婪（无探索）")

    print(f"\n  3️⃣  测试集 (Test Set)")
    print(f"     - 用途: 最终性能评估")
    print(f"     - 频率: 训练完成后评估1次")
    print(f"     - 评估次数: {len(data['test'])}次")
    print(f"     - 每次局数: 10局")
    print(f"     - 种子策略: 固定种子 [20000-20009]（与验证集不同）")
    print(f"     - 探索策略: 纯贪婪（无探索）")

    print("\n【性能结果】")

    # 训练集最后100个episode的平均
    train_scores = [x['score'] for x in data['training'][-100:]]
    print(f"\n  训练集（最后100个episode平均）:")
    print(f"     平均分数: {np.mean(train_scores):.1f} ± {np.std(train_scores):.1f}")

    # 验证集最终结果
    val_final = data['validation'][-1]
    print(f"\n  验证集（最终）:")
    print(f"     平均分数: {val_final['mean_score']:.1f} ± {val_final['std_score']:.1f}")
    print(f"     平均奖励: {val_final['mean_reward']:.1f} ± {val_final['std_reward']:.1f}")

    # 验证集最佳结果
    best_val = max(data['validation'], key=lambda x: x['mean_score'])
    print(f"\n  验证集（历史最佳）:")
    print(f"     平均分数: {best_val['mean_score']:.1f} ± {best_val['std_score']:.1f}")
    print(f"     出现在: Episode {best_val['episode']}")

    # 测试集结果
    test_final = data['test'][0]
    print(f"\n  测试集（最终评估）:")
    print(f"     平均分数: {test_final['mean_score']:.1f} ± {test_final['std_score']:.1f}")
    print(f"     平均奖励: {test_final['mean_reward']:.1f} ± {test_final['std_reward']:.1f}")
    print(f"     最高分数: {test_final['max_score']}")
    print(f"     最低分数: {test_final['min_score']}")

    print("\n【关键洞察】")

    # 对比验证集和测试集
    val_test_diff = abs(val_final['mean_score'] - test_final['mean_score'])
    val_test_ratio = val_test_diff / val_final['mean_score'] * 100

    if val_test_ratio < 5:
        print(f"  ✅ 验证集和测试集性能接近（差异{val_test_ratio:.1f}%），泛化能力良好")
    elif val_test_ratio < 10:
        print(f"  ⚠️  验证集和测试集有一定差异（{val_test_ratio:.1f}%），可能存在轻微过拟合")
    else:
        print(f"  ❌ 验证集和测试集差异较大（{val_test_ratio:.1f}%），可能存在过拟合")

    # 检查训练曲线
    val_scores = [x['mean_score'] for x in data['validation']]
    if val_scores[-1] > val_scores[len(val_scores)//2]:
        print(f"  ✅ 验证集性能持续提升，训练有效")
    else:
        print(f"  ⚠️  验证集性能后期下降，可能训练过久")

    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    import os

    if not os.path.exists('training_metrics.json'):
        print("❌ 错误：找不到 training_metrics.json")
        print("请先运行 train_with_logging.py")
        exit(1)

    visualize_train_val_test_timeline()
    print_summary()
