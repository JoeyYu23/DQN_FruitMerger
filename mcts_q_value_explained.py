#!/usr/bin/env python3
"""
MCTS Q值计算详解
展示Q值是如何一步步计算出来的
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

def visualize_q_calculation():
    """可视化Q值计算流程"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ==========================================
    # 1. MCTS搜索树结构
    # ==========================================
    ax1 = axes[0, 0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    ax1.set_title('MCTS Search Tree Structure', fontsize=16, fontweight='bold')

    # Root节点
    root = FancyBboxPatch((4, 8), 2, 1, boxstyle="round,pad=0.1",
                          edgecolor='blue', facecolor='lightblue', linewidth=3)
    ax1.add_patch(root)
    ax1.text(5, 8.5, 'Root\nvisits=200', ha='center', va='center', fontsize=10, fontweight='bold')

    # 子节点（不同动作）
    children_x = [1, 3, 5, 7, 9]
    children_labels = ['Col 0', 'Col 2', 'Col 4', 'Col 6', 'Col 8']
    children_visits = [3, 5, 163, 10, 1]
    children_values = [85, 75, 105, 32, 84]

    for i, (x, label, visits, value) in enumerate(zip(children_x, children_labels,
                                                        children_visits, children_values)):
        # 连线
        ax1.plot([5, x+0.5], [8, 6.5], 'k-', alpha=0.5, linewidth=2)

        # 节点
        color = 'lightgreen' if visits > 100 else 'lightyellow'
        child = FancyBboxPatch((x, 5.5), 1, 1, boxstyle="round,pad=0.05",
                              edgecolor='green' if visits > 100 else 'gray',
                              facecolor=color, linewidth=2)
        ax1.add_patch(child)

        # 文字
        ax1.text(x+0.5, 6.2, label, ha='center', va='center', fontsize=8)
        ax1.text(x+0.5, 5.9, f'V={visits}', ha='center', va='center', fontsize=7)

    # 说明
    ax1.text(5, 3.5, 'Each node stores:', ha='center', fontsize=11, fontweight='bold')
    ax1.text(5, 3.0, '• visit_count: 访问次数', ha='center', fontsize=9)
    ax1.text(5, 2.5, '• total_value: 累计价值', ha='center', fontsize=9)
    ax1.text(5, 2.0, '• Q = total_value / visit_count', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    # ==========================================
    # 2. 单次模拟（Rollout）流程
    # ==========================================
    ax2 = axes[0, 1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    ax2.set_title('Single Simulation (Rollout)', fontsize=16, fontweight='bold')

    steps_y = [9, 7.5, 6, 4.5, 3, 1.5]
    steps_labels = [
        '1. Selection\n选择最佳子节点',
        '2. Expansion\n扩展新节点',
        '3. Simulation\n快速模拟到终局',
        '4. Evaluation\n计算最终价值',
        '5. Backpropagation\n回传更新',
        'Result: Q值更新'
    ]

    for i, (y, label) in enumerate(zip(steps_y, steps_labels)):
        color = 'lightblue' if i < 3 else 'lightcoral' if i < 5 else 'lightgreen'
        box = FancyBboxPatch((0.5, y-0.3), 9, 0.6, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax2.add_patch(box)
        ax2.text(5, y, label, ha='center', va='center', fontsize=10)

        if i < len(steps_y) - 1:
            ax2.annotate('', xy=(5, y-0.9), xytext=(5, y-0.4),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # ==========================================
    # 3. Value计算公式
    # ==========================================
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Value Calculation Formula', fontsize=16, fontweight='bold')

    # 公式展示
    formulas = [
        ('Rollout终局评估:', 'darkblue', 8.5),
        ('value = score', 'black', 7.8),
        ('        - death_penalty  (if game over)', 'red', 7.2),
        ('        - height_penalty', 'orange', 6.6),
        ('', 'black', 6.0),
        ('其中:', 'darkblue', 5.5),
        ('• score: 当前游戏得分', 'black', 4.9),
        ('• death_penalty = 500', 'red', 4.3),
        ('• height_penalty = 5.0 × (h/H) × H', 'orange', 3.7),
        ('  (h: 最高水果行, H: 总高度)', 'gray', 3.2),
        ('', 'black', 2.6),
        ('更新Q值:', 'darkblue', 2.1),
        ('total_value += value', 'black', 1.5),
        ('visit_count += 1', 'black', 0.9),
        ('Q = total_value / visit_count', 'green', 0.3),
    ]

    for text, color, y in formulas:
        weight = 'bold' if color in ['darkblue', 'green'] else 'normal'
        size = 11 if color in ['darkblue', 'green'] else 10
        ax3.text(1, y, text, fontsize=size, color=color,
                fontfamily='monospace', fontweight=weight)

    # ==========================================
    # 4. 实际例子
    # ==========================================
    ax4 = axes[1, 1]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    ax4.set_title('Example: Column 4 at Step 30', fontsize=16, fontweight='bold')

    example_data = [
        ('初始状态:', 'darkblue', 9.0),
        ('visit_count = 0', 'black', 8.4),
        ('total_value = 0', 'black', 7.9),
        ('', 'black', 7.4),
        ('经过200次模拟后:', 'darkblue', 6.9),
        ('visit_count = 163', 'green', 6.3),
        ('', 'black', 5.8),
        ('每次模拟的value示例:', 'darkblue', 5.3),
        ('  Sim 1: score=95, height_penalty=12 → v=83', 'gray', 4.7),
        ('  Sim 2: score=110, height_penalty=15 → v=95', 'gray', 4.2),
        ('  Sim 3: score=120, death! → v=-380', 'gray', 3.7),
        ('  ...', 'gray', 3.2),
        ('  Sim 163: score=105, height_penalty=10 → v=95', 'gray', 2.7),
        ('', 'black', 2.2),
        ('total_value = 17082 (累计163次)', 'black', 1.7),
        ('Q = 17082 / 163 = 104.8 ✓', 'green', 1.0, 'bold', 12),
    ]

    for item in example_data:
        if len(item) == 3:
            text, color, y = item
            weight = 'normal'
            size = 10
        else:
            text, color, y, weight, size = item

        ax4.text(1, y, text, fontsize=size, color=color,
                fontfamily='monospace', fontweight=weight)

    plt.tight_layout()
    plt.savefig('mcts_q_value_explanation.png', dpi=150, bbox_inches='tight')
    print("✅ Q值计算可视化已保存: mcts_q_value_explanation.png")
    plt.close()


def create_detailed_example():
    """创建详细的数值示例"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 左图：模拟过程
    ax1.set_title('200 Simulations Process', fontsize=14, fontweight='bold')

    # 模拟一些rollout的value分布
    np.random.seed(42)
    simulations = 163

    # 大部分模拟得到正常分数（80-120）
    normal_values = np.random.normal(95, 15, int(simulations * 0.85))
    # 少数模拟game over（负分）
    failed_values = np.random.uniform(-500, -300, int(simulations * 0.1))
    # 少数模拟特别好（高分）
    good_values = np.random.uniform(120, 150, int(simulations * 0.05))

    all_values = np.concatenate([normal_values, failed_values, good_values])
    all_values = all_values[:simulations]  # 确保正好163个

    # 计算累计
    cumulative_values = np.cumsum(all_values)
    cumulative_q = cumulative_values / np.arange(1, simulations + 1)

    # 绘制
    ax1.plot(cumulative_q, linewidth=2, color='blue', label='Q value')
    ax1.axhline(y=cumulative_q[-1], color='red', linestyle='--',
               label=f'Final Q = {cumulative_q[-1]:.1f}')
    ax1.set_xlabel('Simulation Number', fontsize=12)
    ax1.set_ylabel('Q Value (Averaged)', fontsize=12)
    ax1.set_title('Q Value Convergence', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 添加注释
    ax1.text(simulations//2, cumulative_q[-1] + 10,
            f'After {simulations} simulations:\nQ = {cumulative_q[-1]:.1f}',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=11, ha='center')

    # 右图：value分布
    ax2.hist(all_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=np.mean(all_values), color='red', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(all_values):.1f}')
    ax2.set_xlabel('Value from Rollout', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Distribution of Rollout Values', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 标注区域
    ax2.axvspan(-500, -200, alpha=0.2, color='red', label='Game Over')
    ax2.axvspan(70, 120, alpha=0.2, color='green', label='Normal')
    ax2.axvspan(120, 150, alpha=0.2, color='gold', label='Excellent')

    plt.tight_layout()
    plt.savefig('mcts_q_value_detailed.png', dpi=150, bbox_inches='tight')
    print("✅ Q值详细示例已保存: mcts_q_value_detailed.png")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("MCTS Q值计算详解")
    print("="*70)

    visualize_q_calculation()
    create_detailed_example()

    print("\n" + "="*70)
    print("Q值计算总结:")
    print("="*70)
    print("""
    1. MCTS通过200次模拟（simulation）评估每个动作

    2. 每次模拟的步骤:
       ① Selection: 从根节点向下选择最有希望的路径
       ② Expansion: 扩展一个新节点
       ③ Simulation: 从新节点快速模拟到游戏结束（rollout）
       ④ Evaluation: 计算这次模拟的价值
          value = score - death_penalty - height_penalty
       ⑤ Backpropagation: 将value回传给路径上的所有节点

    3. 每个节点维护两个数值:
       • visit_count: 被访问的次数
       • total_value: 所有模拟的value之和

    4. Q值 = total_value / visit_count
       → 表示这个动作的"平均预期价值"

    5. 访问次数越多的动作，Q值越可信
       → Step 30的Column 4: 163次访问 → Q=104.8
       → Step 46的Column 6: 190次访问 → Q=168.7

    6. MCTS最终选择访问次数最多的动作
       （因为访问次数多 = MCTS认为它最有价值）
    """)
    print("="*70)
