"""
多局游戏统计分析 - 评估DQN AI性能
"""
import numpy as np
import paddle
from DQN import Agent, build_model, RandomAgent
from GameInterface import GameInterface
import time
from datetime import datetime

def evaluate_agent(agent, env, num_games=100, show_progress=True):
    """评估agent在多局游戏中的表现"""
    scores = []
    rewards = []
    steps = []

    print(f"\n开始评估 {num_games} 局游戏...")
    start_time = time.time()

    for game_id in range(num_games):
        env.reset(seed=game_id)  # 使用固定种子确保可重复

        step_count = 0
        reward_sum = 0

        # 随机选择第一个动作
        action = np.random.randint(0, env.action_num)
        feature, _, alive = env.next(action)

        while alive:
            step_count += 1

            # 使用predict确保使用最佳策略
            action = agent.predict(feature)

            # 确保action是标量
            if isinstance(action, np.ndarray):
                action = action.item()

            feature, reward, alive = env.next(action)
            reward_sum += np.sum(reward)

        # 记录统计数据
        scores.append(env.game.score)
        rewards.append(reward_sum)
        steps.append(step_count)

        if show_progress and (game_id + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (game_id + 1)
            remaining = avg_time * (num_games - game_id - 1)
            print(f"  进度: {game_id + 1}/{num_games} "
                  f"| 最近分数: {env.game.score} "
                  f"| 预计剩余: {remaining:.1f}秒")

    elapsed_time = time.time() - start_time

    return {
        'scores': np.array(scores),
        'rewards': np.array(rewards),
        'steps': np.array(steps),
        'elapsed_time': elapsed_time
    }

def print_statistics(name, data, color=""):
    """打印统计信息"""
    scores = data['scores']
    rewards = data['rewards']
    steps = data['steps']

    print(f"\n{'=' * 70}")
    print(f"📊 {name} 统计结果")
    print(f"{'=' * 70}")

    print(f"\n🎯 分数统计:")
    print(f"  平均分数: {np.mean(scores):.2f}")
    print(f"  最高分数: {np.max(scores):.0f}")
    print(f"  最低分数: {np.min(scores):.0f}")
    print(f"  标准差:   {np.std(scores):.2f}")
    print(f"  中位数:   {np.median(scores):.2f}")

    print(f"\n🏆 奖励统计:")
    print(f"  平均奖励: {np.mean(rewards):.2f}")
    print(f"  最高奖励: {np.max(rewards):.0f}")
    print(f"  最低奖励: {np.min(rewards):.0f}")

    print(f"\n👣 步数统计:")
    print(f"  平均步数: {np.mean(steps):.2f}")
    print(f"  最多步数: {np.max(steps):.0f}")
    print(f"  最少步数: {np.min(steps):.0f}")

    print(f"\n⏱️  耗时: {data['elapsed_time']:.2f}秒")

    # 分数分布
    print(f"\n📈 分数分布:")
    bins = [0, 100, 150, 200, 250, 300, 400, 1000]
    for i in range(len(bins) - 1):
        count = np.sum((scores >= bins[i]) & (scores < bins[i+1]))
        percentage = count / len(scores) * 100
        bar = '█' * int(percentage / 2)
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {count:3d}局 ({percentage:5.1f}%) {bar}")

def compare_agents(dqn_data, random_data):
    """对比DQN和随机agent"""
    print(f"\n{'=' * 70}")
    print(f"⚔️  DQN vs 随机Agent 对比")
    print(f"{'=' * 70}")

    dqn_avg = np.mean(dqn_data['scores'])
    random_avg = np.mean(random_data['scores'])
    improvement = (dqn_avg - random_avg) / random_avg * 100

    print(f"\n平均分数对比:")
    print(f"  DQN Agent:    {dqn_avg:.2f}")
    print(f"  Random Agent: {random_avg:.2f}")
    print(f"  提升:         {improvement:+.1f}%")

    dqn_max = np.max(dqn_data['scores'])
    random_max = np.max(random_data['scores'])

    print(f"\n最高分数对比:")
    print(f"  DQN Agent:    {dqn_max:.0f}")
    print(f"  Random Agent: {random_max:.0f}")
    print(f"  差距:         {dqn_max - random_max:+.0f}")

    # 胜率统计
    wins = 0
    for i in range(len(dqn_data['scores'])):
        if dqn_data['scores'][i] > random_data['scores'][i]:
            wins += 1

    win_rate = wins / len(dqn_data['scores']) * 100
    print(f"\n🏅 DQN胜率: {win_rate:.1f}% ({wins}/{len(dqn_data['scores'])}局)")

def save_results(dqn_data, random_data, filename="evaluation_results.txt"):
    """保存结果到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估局数: {len(dqn_data['scores'])}\n\n")

        f.write("DQN Agent 详细数据:\n")
        for i, (score, reward, step) in enumerate(zip(
            dqn_data['scores'], dqn_data['rewards'], dqn_data['steps']
        )):
            f.write(f"  第{i+1:3d}局: 分数={score:3.0f}, 奖励={reward:6.1f}, 步数={step:3.0f}\n")

        f.write("\n随机Agent 详细数据:\n")
        for i, (score, reward, step) in enumerate(zip(
            random_data['scores'], random_data['rewards'], random_data['steps']
        )):
            f.write(f"  第{i+1:3d}局: 分数={score:3.0f}, 奖励={reward:6.1f}, 步数={step:3.0f}\n")

    print(f"\n💾 详细结果已保存到: {filename}")

if __name__ == "__main__":
    print("=" * 70)
    print("🎮 DQN水果合成AI - 多局性能评估")
    print("=" * 70)

    # 设置评估参数
    NUM_GAMES = 100  # 评估局数

    print(f"\n设置:")
    print(f"  评估局数: {NUM_GAMES}")

    # 初始化环境
    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH
    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2

    env = GameInterface()

    # 加载DQN Agent
    print(f"\n📦 加载DQN模型...")
    dqn_agent = Agent(build_model, feature_dim, action_dim, e_greed=0.0)
    dqn_agent.policy_net.set_state_dict(paddle.load("final.pdparams"))
    print("✅ DQN模型加载成功!")

    # 创建随机Agent
    random_agent = RandomAgent(action_dim)

    # 评估DQN Agent
    print(f"\n{'=' * 70}")
    print("🤖 评估DQN Agent")
    print(f"{'=' * 70}")
    dqn_data = evaluate_agent(dqn_agent, env, NUM_GAMES)
    print_statistics("DQN Agent", dqn_data)

    # 评估随机Agent
    print(f"\n{'=' * 70}")
    print("🎲 评估随机Agent")
    print(f"{'=' * 70}")
    random_data = evaluate_agent(random_agent, env, NUM_GAMES)
    print_statistics("随机Agent", random_data)

    # 对比结果
    compare_agents(dqn_data, random_data)

    # 保存结果
    save_results(dqn_data, random_data)

    print(f"\n{'=' * 70}")
    print("✅ 评估完成!")
    print(f"{'=' * 70}")
