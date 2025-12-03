"""
对比MCTS和DQN的表现
"""
import sys
import os
import numpy as np
import time
import paddle

from GameInterface import GameInterface
from DQN import Agent, build_model, RandomAgent

# MCTS暂时用随机代替，如果能加载就用MCTS
USE_MCTS = False
try:
    from mcts.MCTS_optimized import FastMCTSAgent
    USE_MCTS = True
    print("✓ MCTS模块加载成功")
except:
    print("✗ MCTS模块加载失败，将使用随机Agent对比")


def evaluate_agent(agent, env, num_games=10, agent_name="Agent"):
    """评估一个agent"""
    scores = []
    steps_list = []
    times_per_move = []

    for game_idx in range(num_games):
        env.reset(seed=game_idx * 100)

        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        step = 0
        total_time = 0

        while alive:
            step += 1

            start = time.time()
            if hasattr(agent, 'predict'):
                if agent_name == "MCTS":
                    action = agent.predict(env)
                else:
                    action = agent.predict(feature)
            else:
                action = agent.sample(feature)
            elapsed = time.time() - start
            total_time += elapsed

            # 处理action：可能是numpy标量、numpy数组或整数
            if isinstance(action, np.ndarray):
                if action.ndim == 0:  # 0维数组（标量）
                    action_val = int(action)
                else:
                    action_val = int(action[0])
            else:
                action_val = int(action)

            feature, reward, alive = env.next(action_val)

        final_score = env.game.score
        avg_time = total_time / step if step > 0 else 0

        scores.append(final_score)
        steps_list.append(step)
        times_per_move.append(avg_time)

        print(f"  游戏 {game_idx + 1}/{num_games}: 得分={final_score:4d}, 步数={step:3d}, 时间={avg_time:.4f}s/步")

    return {
        'agent': agent_name,
        'scores': scores,
        'steps': steps_list,
        'times': times_per_move,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_time': np.mean(times_per_move)
    }


def main():
    print("=" * 70)
    print("🎮 DQN vs MCTS 对比测试")
    print("=" * 70)

    # 初始化环境
    env = GameInterface()
    feature_dim = GameInterface.FEATURE_MAP_WIDTH * GameInterface.FEATURE_MAP_HEIGHT * 2
    action_dim = env.action_num

    print(f"环境配置: feature_dim={feature_dim}, action_dim={action_dim}")

    num_games = 5  # 每个agent测试5局

    results = {}

    # 1. 测试随机Agent（基准）
    print("\n[1/3] 测试随机Agent（基准）")
    print("-" * 70)
    random_agent = RandomAgent(action_dim)
    results['Random'] = evaluate_agent(random_agent, env, num_games, "Random")

    # 2. 测试DQN Agent
    print("\n[2/3] 测试DQN Agent")
    print("-" * 70)
    dqn_agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)

    # 尝试加载模型
    model_paths = ['final_5000.pdparams', 'final.pdparams']
    loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"  加载模型: {model_path}")
            dqn_agent.policy_net.set_state_dict(paddle.load(model_path))
            loaded = True
            break

    if not loaded:
        print("  ⚠️  未找到训练好的模型，使用未训练的DQN")

    results['DQN'] = evaluate_agent(dqn_agent, env, num_games, "DQN")

    # 3. 测试MCTS Agent
    print("\n[3/3] 测试MCTS Agent")
    print("-" * 70)
    if USE_MCTS:
        mcts_agent = FastMCTSAgent(num_simulations=100)
        results['MCTS'] = evaluate_agent(mcts_agent, env, num_games, "MCTS")
    else:
        print("  跳过MCTS测试（模块未加载）")

    # 打印对比结果
    print("\n" + "=" * 70)
    print("📊 对比结果")
    print("=" * 70)

    print(f"\n{'Agent':<12} {'平均得分':<15} {'最高分':<10} {'最低分':<10} {'平均时间(s/步)':<15}")
    print("-" * 70)

    for agent_name in ['Random', 'DQN', 'MCTS']:
        if agent_name not in results:
            continue
        r = results[agent_name]
        print(f"{r['agent']:<12} {r['mean_score']:>6.1f} ± {r['std_score']:<5.1f} "
              f"{r['max_score']:>10} {r['min_score']:>10} {r['mean_time']:>15.4f}")

    # 对比分析
    print("\n" + "=" * 70)
    print("🎯 对比分析")
    print("=" * 70)

    baseline = results['Random']['mean_score']

    for agent_name in ['DQN', 'MCTS']:
        if agent_name not in results:
            continue
        r = results[agent_name]
        improvement = ((r['mean_score'] - baseline) / baseline * 100) if baseline > 0 else 0

        print(f"\n{agent_name} vs Random:")
        print(f"  平均得分提升: {r['mean_score'] - baseline:+.1f} ({improvement:+.1f}%)")
        print(f"  最高分提升: {r['max_score'] - results['Random']['max_score']:+d}")
        print(f"  计算速度: {r['mean_time']:.4f}秒/步")

        if r['mean_score'] > baseline * 1.5:
            print(f"  ✅ {agent_name}显著优于随机策略")
        elif r['mean_score'] > baseline:
            print(f"  🔸 {agent_name}略优于随机策略")
        else:
            print(f"  ❌ {agent_name}未超越随机策略")

    # DQN vs MCTS
    if 'DQN' in results and 'MCTS' in results:
        print(f"\nDQN vs MCTS:")
        dqn_score = results['DQN']['mean_score']
        mcts_score = results['MCTS']['mean_score']
        diff = dqn_score - mcts_score

        if abs(diff) < 5:
            print(f"  🤝 性能接近 (差距: {abs(diff):.1f}分)")
        elif dqn_score > mcts_score:
            print(f"  🏆 DQN胜出 (+{diff:.1f}分, {diff/mcts_score*100:+.1f}%)")
        else:
            print(f"  🏆 MCTS胜出 (+{abs(diff):.1f}分, {abs(diff)/dqn_score*100:+.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
