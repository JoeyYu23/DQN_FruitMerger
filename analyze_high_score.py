"""
分析高分局 - 可视化AI的决策过程
"""
import numpy as np
import paddle
import cv2
from DQN import Agent, build_model
from GameInterface import GameInterface
from Game import visualize_feature
from render_utils import cover
import time

def replay_game_with_analysis(agent, env, seed, save_video=False):
    """重放游戏并分析每一步"""
    env.reset(seed=seed)

    decisions = []  # 记录每一步的决策

    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step_count = 0
    reward_sum = 0

    print(f"\n{'='*60}")
    print(f"重放游戏 (seed={seed})")
    print(f"{'='*60}")

    while alive:
        step_count += 1

        # 获取所有动作的Q值
        with paddle.no_grad():
            q_values = agent.policy_net(paddle.to_tensor(feature)).numpy()

        # 选择最佳动作
        action = agent.predict(feature)
        if isinstance(action, np.ndarray):
            action = action.item()

        # 记录决策信息
        decision = {
            'step': step_count,
            'action': action,
            'q_values': q_values.copy(),
            'best_q': q_values[action],
            'current_fruit': env.game.current_fruit_type,
            'feature': feature.copy(),
            'score_before': env.game.score
        }

        # 执行动作
        next_feature, reward, alive = env.next(action)
        reward_sum += np.sum(reward)

        decision['reward'] = np.sum(reward)
        decision['score_after'] = env.game.score
        decision['alive'] = alive

        decisions.append(decision)

        # 打印关键步骤
        if np.sum(reward) > 0:  # 有正奖励（合成了）
            fruit_name = ['', '葡萄', '樱桃', '草莓', '橙子', '柿子', '桃子', '菠萝', '椰子', '西瓜半', '西瓜', '大西瓜'][decision['current_fruit']]
            print(f"  步骤 {step_count:3d}: 放置{fruit_name} 在位置{action:2d} "
                  f"→ 奖励={reward:5.1f}, 分数={env.game.score:3d}, "
                  f"Q值={decision['best_q']:.2f}")

        feature = next_feature

    print(f"\n最终结果: 分数={env.game.score}, 总奖励={reward_sum:.1f}, 步数={step_count}")

    return decisions, env.game.score, reward_sum

def analyze_decisions(decisions):
    """分析决策模式"""
    print(f"\n{'='*60}")
    print("决策分析")
    print(f"{'='*60}")

    # 动作分布
    actions = [d['action'] for d in decisions]
    action_counts = np.bincount(actions, minlength=16)

    print("\n📊 动作位置分布:")
    for i in range(16):
        if action_counts[i] > 0:
            bar = '█' * int(action_counts[i] / max(action_counts) * 30)
            print(f"  位置 {i:2d}: {action_counts[i]:3d}次 {bar}")

    # 找出最常用的位置
    top_positions = np.argsort(action_counts)[::-1][:3]
    print(f"\n🎯 最常用位置: {', '.join([str(p) for p in top_positions if action_counts[p] > 0])}")

    # Q值统计
    q_values = [d['best_q'] for d in decisions]
    print(f"\n📈 Q值统计:")
    print(f"  平均Q值: {np.mean(q_values):.2f}")
    print(f"  最高Q值: {np.max(q_values):.2f}")
    print(f"  最低Q值: {np.min(q_values):.2f}")

    # 奖励分布
    rewards = [d['reward'] for d in decisions if d['reward'] > 0]
    if rewards:
        print(f"\n💰 正奖励统计:")
        print(f"  获得奖励次数: {len(rewards)}")
        print(f"  平均奖励: {np.mean(rewards):.2f}")
        print(f"  最高奖励: {np.max(rewards):.0f}")

    # 找出关键决策（高奖励）
    high_reward_steps = [d for d in decisions if d['reward'] >= 10]
    if high_reward_steps:
        print(f"\n🌟 关键决策（奖励≥10）:")
        for d in high_reward_steps[:5]:  # 显示前5个
            print(f"  步骤 {d['step']:3d}: 位置{d['action']:2d}, "
                  f"奖励={d['reward']:.0f}, Q值={d['best_q']:.2f}")

def visualize_game_step(agent, env, seed, target_step, action_dim):
    """可视化特定步骤的游戏状态和AI决策"""
    env.reset(seed=seed)

    action = np.random.randint(0, env.action_num)
    feature, _, alive = env.next(action)

    step = 0

    while alive and step < target_step:
        step += 1
        action = agent.predict(feature)
        if isinstance(action, np.ndarray):
            action = action.item()
        feature, _, alive = env.next(action)

    if not alive or step != target_step:
        print(f"无法到达步骤 {target_step}")
        return

    # 获取Q值
    with paddle.no_grad():
        q_values = agent.policy_net(paddle.to_tensor(feature)).numpy().flatten()

    # 可视化
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    # 绘制游戏画面
    screen = env.game.draw()

    # 绘制所有动作的Q值（热力图）
    unit_w = 1.0 * env.game.width / action_dim

    # 归一化Q值用于颜色映射
    q_min, q_max = q_values.min(), q_values.max()
    q_norm = (q_values - q_min) / (q_max - q_min + 1e-8)

    for i in range(action_dim):
        color_intensity = int(q_norm[i] * 255)
        rect = np.zeros_like(screen, dtype=np.uint8)
        # 绿色->黄色->红色 表示Q值从低到高
        if q_norm[i] < 0.5:
            color = (0, int(255 * q_norm[i] * 2), int(255 * (1 - q_norm[i] * 2)), 100)
        else:
            color = (0, 255, 0, 100)

        cv2.rectangle(rect,
                     (int(i * unit_w), 0),
                     (int((i + 1) * unit_w), env.game.height),
                     color, -1)
        cover(screen, rect, 0.3)

    # 标记最佳动作
    best_action = np.argmax(q_values)
    best_rect = np.zeros_like(screen, dtype=np.uint8)
    cv2.rectangle(best_rect,
                 (int(best_action * unit_w), 0),
                 (int((best_action + 1) * unit_w), env.game.height),
                 (0, 0, 255, 150), 3)
    cover(screen, best_rect, 1)

    # 添加Q值文本
    for i in range(action_dim):
        x = int((i + 0.5) * unit_w)
        cv2.putText(screen, f"{q_values[i]:.1f}",
                   (x - 15, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                   (255, 255, 255), 1)

    # 添加信息文本
    info = f"Step:{step} Score:{env.game.score} Best:{best_action} Q:{q_values[best_action]:.2f}"
    cv2.putText(screen, info, (5, env.game.height - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # 显示特征图
    reshaped_feature = feature.reshape((feature_map_height, feature_map_width, 2))
    feature_img = visualize_feature(reshaped_feature, env.game.resolution).astype(np.uint8)

    # 合并显示
    combined = np.hstack([cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR), feature_img])

    return combined, q_values, best_action

def find_top_games(num_games=100, top_k=3):
    """找出分数最高的几局游戏"""
    print("🔍 寻找高分局...")

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    env = GameInterface()
    agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    agent.policy_net.set_state_dict(paddle.load("final.pdparams"))

    game_results = []

    for seed in range(num_games):
        env.reset(seed=seed)
        step_count = 0
        reward_sum = 0

        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        while alive:
            step_count += 1
            action = agent.predict(feature)
            if isinstance(action, np.ndarray):
                action = action.item()
            feature, reward, alive = env.next(action)
            reward_sum += np.sum(reward)

        game_results.append({
            'seed': seed,
            'score': env.game.score,
            'reward': reward_sum,
            'steps': step_count
        })

        if (seed + 1) % 20 == 0:
            print(f"  已扫描 {seed + 1}/{num_games} 局")

    # 排序找出前k名
    game_results.sort(key=lambda x: x['score'], reverse=True)

    print(f"\n🏆 Top {top_k} 高分局:")
    for i, game in enumerate(game_results[:top_k]):
        print(f"  #{i+1}: Seed={game['seed']:3d}, "
              f"分数={game['score']:3d}, "
              f"奖励={game['reward']:6.1f}, "
              f"步数={game['steps']:3d}")

    return game_results[:top_k], agent, env

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 DQN高分局分析")
    print("=" * 60)

    # 找出高分局
    top_games, agent, env = find_top_games(num_games=100, top_k=5)

    # 详细分析最高分局
    best_game = top_games[0]
    print(f"\n\n{'='*60}")
    print(f"🎯 详细分析最高分局 (Seed={best_game['seed']})")
    print(f"{'='*60}")

    decisions, final_score, total_reward = replay_game_with_analysis(
        agent, env, best_game['seed']
    )

    analyze_decisions(decisions)

    # 可视化关键步骤
    print(f"\n\n{'='*60}")
    print("📸 生成关键步骤可视化")
    print(f"{'='*60}")

    key_steps = [1, len(decisions)//4, len(decisions)//2, len(decisions)*3//4, len(decisions)-1]

    for step in key_steps:
        if step < len(decisions):
            img, q_values, best_action = visualize_game_step(
                agent, env, best_game['seed'], step, GameInterface.ACTION_NUM
            )
            filename = f"high_score_step_{step:03d}.png"
            cv2.imwrite(filename, img)
            print(f"  保存步骤 {step:3d}: {filename}")

    print(f"\n✅ 分析完成!")
    print(f"\n💡 关键发现:")
    print(f"  - 最高分: {best_game['score']} (Seed {best_game['seed']})")
    print(f"  - 可视化图片已保存到当前目录")
    print(f"  - 查看图片了解AI的决策过程")
