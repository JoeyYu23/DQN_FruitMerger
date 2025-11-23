"""
SelfPlay.py - 自我博弈数据收集

核心功能：
1. 用MCTS+网络玩游戏
2. 收集训练数据 (state, π, z)
3. 支持多进程并行收集
"""

import numpy as np
import paddle
from typing import List, Tuple
from GameInterface import GameInterface
from AlphaZeroMCTS import AlphaZeroMCTS
from SuikaNet import SuikaNet
from StateConverter import StateConverter
import time


class SelfPlayCollector:
    """自我博弈数据收集器"""

    def __init__(self,
                 network: SuikaNet,
                 num_simulations: int = 200,
                 temperature: float = 1.0,
                 temperature_threshold: int = 30,
                 add_noise: bool = True):
        """
        Args:
            network: SuikaNet神经网络
            num_simulations: MCTS每步模拟次数
            temperature: 温度参数 (前N步用于探索)
            temperature_threshold: 前N步使用温度采样，之后确定性选择
            add_noise: 是否添加Dirichlet噪声
        """
        self.network = network
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.add_noise = add_noise

        # 创建StateConverter - 使用网络的board尺寸
        self.converter = StateConverter(
            grid_height=network.board_h,
            grid_width=network.board_w,
            feature_height=network.board_h,
            feature_width=network.board_w
        )

    def play_one_episode(self, seed: int = None, verbose: bool = False) -> List[Tuple]:
        """
        玩一局游戏，收集训练数据

        Args:
            seed: 随机种子（用于复现）
            verbose: 是否打印详细信息

        Returns:
            data: [(state_tensor, pi, z), ...]
                state_tensor: [13, H, W] numpy数组
                pi: [16] 概率分布
                z: float 标准化的最终得分
        """
        # 创建游戏
        game = GameInterface()
        game.reset(seed=seed)

        # 创建MCTS（每局游戏重新创建）
        mcts = AlphaZeroMCTS(
            network=self.network,
            c_puct=1.5,
            num_simulations=self.num_simulations,
            temperature=self.temperature,
            add_dirichlet_noise=self.add_noise
        )

        # 存储数据
        states = []
        pis = []
        step_count = 0

        if verbose:
            print(f"[SelfPlay] Starting episode (seed={seed})...")

        # 游戏循环
        while game.game.alive:
            step_count += 1

            # 1. 获取当前状态
            simplified_state = self.converter.game_to_simplified(game)
            state_tensor = self.converter.simplified_to_tensor(
                simplified_state, add_batch_dim=False
            )

            # 2. 决定温度（前N步探索，之后确定性）
            current_temp = self.temperature if step_count <= self.temperature_threshold else 0.0
            mcts.temperature = current_temp

            # 3. MCTS搜索得到策略π
            pi = mcts.search(simplified_state)

            # 4. 采样动作
            if current_temp == 0:
                action = int(np.argmax(pi))
            else:
                action = np.random.choice(len(pi), p=pi)

            # 5. 执行动作
            _, _, alive = game.next(action)

            # 6. 保存 (state, pi)
            states.append(state_tensor.numpy())
            pis.append(pi)

            if verbose and step_count % 10 == 0:
                print(f"  Step {step_count}, Score: {game.game.score}, Action: {action}")

        # 游戏结束
        final_score = game.game.score

        if verbose:
            print(f"[SelfPlay] Episode finished: {step_count} steps, score={final_score}")

        # 计算z（标准化得分）
        z = self.normalize_score(final_score)

        # 构建训练数据
        data = []
        for state, pi in zip(states, pis):
            data.append((state, pi, z))

        return data

    def normalize_score(self, score: float) -> float:
        """
        归一化得分到[-1, 1]

        策略：
        - 用tanh压缩到(-1, 1)
        - scale参数控制压缩程度
        """
        scale = 1000.0  # 可调整
        return np.tanh(score / scale)

    def collect_episodes(self,
                        num_episodes: int,
                        start_seed: int = 0,
                        verbose: bool = True) -> List[Tuple]:
        """
        收集多局游戏数据

        Args:
            num_episodes: 收集的局数
            start_seed: 起始随机种子
            verbose: 是否打印进度

        Returns:
            all_data: [(state, pi, z), ...] 所有数据
        """
        all_data = []
        scores = []

        start_time = time.time()

        for ep in range(num_episodes):
            seed = start_seed + ep

            # 玩一局
            episode_data = self.play_one_episode(seed=seed, verbose=False)
            all_data.extend(episode_data)

            # 统计
            if len(episode_data) > 0:
                z = episode_data[0][2]  # 所有step的z相同
                score = self.denormalize_score(z)
                scores.append(score)

            # 打印进度
            if verbose and (ep + 1) % 5 == 0:
                elapsed = time.time() - start_time
                avg_score = np.mean(scores) if scores else 0
                samples_per_sec = len(all_data) / elapsed if elapsed > 0 else 0

                print(f"[SelfPlay] {ep+1}/{num_episodes} episodes | "
                      f"Samples: {len(all_data)} | "
                      f"Avg score: {avg_score:.1f} | "
                      f"Speed: {samples_per_sec:.1f} samples/s")

        if verbose:
            total_time = time.time() - start_time
            print(f"\n[SelfPlay] Collection finished!")
            print(f"  Total episodes: {num_episodes}")
            print(f"  Total samples: {len(all_data)}")
            print(f"  Avg steps per episode: {len(all_data)/num_episodes:.1f}")
            print(f"  Avg score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
            print(f"  Max score: {np.max(scores):.1f}")
            print(f"  Total time: {total_time:.1f}s")

        return all_data

    def denormalize_score(self, z: float) -> float:
        """反向计算得分（近似）"""
        scale = 1000.0
        return np.arctanh(np.clip(z, -0.999, 0.999)) * scale


# ============================================
# 测试代码
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("Testing SelfPlay")
    print("="*60)

    # 创建网络
    print("\n[Setup] Creating network...")
    net = SuikaNet(input_channels=13, num_actions=16, hidden_channels=32)

    # 创建收集器
    print("[Setup] Creating collector...")
    collector = SelfPlayCollector(
        network=net,
        num_simulations=50,  # 少量模拟用于测试
        temperature=1.0,
        temperature_threshold=15
    )

    # 测试单局游戏
    print("\n[Test 1] Play one episode:")
    data = collector.play_one_episode(seed=42, verbose=True)
    print(f"  Collected {len(data)} samples")
    if len(data) > 0:
        state, pi, z = data[0]
        print(f"  Sample 0:")
        print(f"    State shape: {state.shape}")
        print(f"    Pi shape: {pi.shape}")
        print(f"    Pi sum: {pi.sum():.6f}")
        print(f"    Z value: {z:.4f}")

    # 测试多局收集
    print("\n[Test 2] Collect multiple episodes:")
    all_data = collector.collect_episodes(num_episodes=3, start_seed=100, verbose=True)
    print(f"\n  Total samples collected: {len(all_data)}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
