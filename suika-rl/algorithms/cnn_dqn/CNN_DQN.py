"""
CNN-based DQN for Suika Game (PyTorch version)
改进：使用CNN提取空间特征，替代MLP的flatten输入
修正：GameInterface返回(640,)需要reshape到(20,16,2)
"""
import os
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 超参数
# ============================================
MEMORY_SIZE = 50000
MEMORY_WARMUP_SIZE = 5000
BATCH_SIZE = 32
LEARNING_RATE = 0.00005  # 🔧 降低学习率提高稳定性 (0.0001 → 0.00005)
GAMMA = 0.99
TARGET_UPDATE = 200

# ============================================
# CNN Q-Network
# ============================================
class CNN_QNet(nn.Module):
    """
    CNN-based Q-Network for Suika Game

    Input: (batch, 2, 20, 16) - 2通道的20×16特征图
    Output: (batch, 16) - 16个动作的Q值
    """
    def __init__(self, action_dim=16):
        super(CNN_QNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # MaxPooling (optional, 减少维度)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算flatten后的维度
        # After conv: (2, 20, 16)
        # After pool1: (16, 10, 8)
        # After pool2: (32, 5, 4)
        # After pool3: (64, 5, 4) → no pool
        # Flatten: 64 * 5 * 4 = 1280
        self.fc1 = nn.Linear(64 * 5 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape: (batch, 2, 20, 16)

        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # → (batch, 16, 10, 8)

        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # → (batch, 32, 5, 4)

        # Conv block 3 (no pool to preserve info)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)  # → (batch, 64, 5, 4)

        # Flatten
        x = x.view(x.size(0), -1)  # → (batch, 1280)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # → (batch, 16)

        return x


# ============================================
# Replay Buffer
# ============================================
class ReplayMemory:
    def __init__(self, max_size=MEMORY_SIZE):
        self.buffer = collections.deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """采样batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为numpy数组
        states = np.array(states, dtype=np.float32)      # (batch, 640)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================
# CNN DQN Agent
# ============================================
class CNN_DQN_Agent:
    def __init__(self, action_dim=16, learning_rate=LEARNING_RATE,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 创建Q网络和目标网络
        self.policy_net = CNN_QNet(action_dim).to(device)
        self.target_net = CNN_QNet(action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 计数器
        self.global_step = 0
        self.target_update = TARGET_UPDATE

    def preprocess_state(self, state):
        """
        预处理状态：(640,) → (20, 16, 2) → (2, 20, 16)

        Args:
            state: numpy array of shape (640,) or (batch, 640)

        Returns:
            torch tensor of shape (2, 20, 16) or (batch, 2, 20, 16)
        """
        if state.ndim == 1:  # Single state (640,)
            # (640,) → (20, 16, 2)
            state = state.reshape(20, 16, 2)
            # (20, 16, 2) → (2, 20, 16)
            state = np.transpose(state, (2, 0, 1))
            state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add batch dim
        else:  # Batch of states (batch, 640)
            # (batch, 640) → (batch, 20, 16, 2)
            state = state.reshape(-1, 20, 16, 2)
            # (batch, 20, 16, 2) → (batch, 2, 20, 16)
            state = np.transpose(state, (0, 3, 1, 2))
            state = torch.FloatTensor(state).to(device)

        return state

    def select_action(self, state, training=True):
        """
        选择动作（ε-greedy）

        Args:
            state: numpy array (640,)
            training: 是否训练模式

        Returns:
            action: int
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()

        return action

    def learn(self, memory, batch_size=BATCH_SIZE):
        """从replay buffer学习 (with numerical stability)"""
        if len(memory) < batch_size:
            return None

        # 采样batch
        states, actions, rewards, next_states, dones = memory.sample(batch_size)

        # 预处理状态 (batch, 640) → (batch, 2, 20, 16)
        states = self.preprocess_state(states)
        next_states = self.preprocess_state(next_states)

        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # 🔧 归一化reward (防止Q值爆炸)
        rewards = rewards / 100.0

        # 计算当前Q值
        q_values = self.policy_net(states)
        # 🔧 裁剪Q值到合理范围
        q_values = torch.clamp(q_values, -10, 10)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # 🔧 裁剪目标网络的Q值
            next_q_values = torch.clamp(next_q_values, -10, 10)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
            # 🔧 裁剪目标Q值
            target_q = torch.clamp(target_q, -10, 10)

        # 🔧 使用Huber loss替代MSE (对异常值更鲁棒)
        loss = F.smooth_l1_loss(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 🔧 更强的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # 更新target network
        self.global_step += 1
        if self.global_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        """衰减epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'global_step': self.global_step
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.global_step = checkpoint['global_step']
        print(f"Model loaded from {path}")


# ============================================
# 训练函数
# ============================================
def train_cnn_dqn(env, agent, memory, num_episodes=2000,
                  save_interval=500, model_dir="weights_cnn_dqn"):
    """
    训练CNN-DQN

    Args:
        env: 游戏环境 (GameInterface)
        agent: CNN_DQN_Agent
        memory: ReplayMemory
        num_episodes: 训练轮数
        save_interval: 保存间隔
        model_dir: 模型保存目录
    """
    os.makedirs(model_dir, exist_ok=True)

    print("="*60)
    print("Training CNN-DQN")
    print("="*60)
    print(f"Device: {device}")
    print(f"Episodes: {num_episodes}")
    print(f"Memory size: {MEMORY_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print("="*60)

    # Warmup: 填充replay buffer
    print("\nWarming up replay buffer...")
    while len(memory) < MEMORY_WARMUP_SIZE:
        env.reset()
        action = random.randint(0, agent.action_dim - 1)
        state, _, alive = env.next(action)

        while alive:
            action = random.randint(0, agent.action_dim - 1)
            next_state, reward, alive = env.next(action)
            memory.push(state, action, reward, next_state, not alive)
            state = next_state

        if len(memory) % 500 == 0:
            print(f"  Buffer size: {len(memory)}/{MEMORY_WARMUP_SIZE}")

    print(f"Warmup complete! Buffer size: {len(memory)}\n")

    # 训练循环
    scores = []
    losses = []

    for episode in range(num_episodes):
        env.reset()
        action = random.randint(0, agent.action_dim - 1)
        state, _, alive = env.next(action)

        episode_reward = 0
        episode_loss = []
        steps = 0

        while alive:
            # 选择动作
            action = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, alive = env.next(action)

            # 存储经验
            memory.push(state, action, reward, next_state, not alive)

            # 学习
            loss = agent.learn(memory)
            if loss is not None:
                episode_loss.append(loss)

            episode_reward += np.sum(reward) if hasattr(reward, '__len__') else reward
            state = next_state
            steps += 1

        # 衰减epsilon
        agent.decay_epsilon()

        # 记录
        score = env.game.score
        scores.append(score)
        if episode_loss:
            losses.append(np.mean(episode_loss))

        # 打印进度
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {episode+1:4d} | "
                  f"Score: {score:3d} | "
                  f"Avg Score: {avg_score:6.1f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Steps: {steps:3d}")

        # 保存模型
        if (episode + 1) % save_interval == 0:
            save_path = os.path.join(model_dir, f"checkpoint_ep{episode+1}.pth")
            agent.save(save_path)

    # 保存最终模型
    final_path = os.path.join(model_dir, "final_model.pth")
    agent.save(final_path)

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Final average score: {np.mean(scores[-100:]):.1f}")
    print("="*60)

    return scores, losses


# ============================================
# 测试函数
# ============================================
def test_cnn_dqn(env, agent, num_episodes=10):
    """测试CNN-DQN"""
    agent.policy_net.eval()
    scores = []

    for episode in range(num_episodes):
        env.reset(seed=1000 + episode)
        action = random.randint(0, agent.action_dim - 1)
        state, _, alive = env.next(action)

        while alive:
            action = agent.select_action(state, training=False)
            state, _, alive = env.next(action)

        score = env.game.score
        scores.append(score)
        print(f"Episode {episode+1}: Score = {score}")

    print(f"\nTest Results:")
    print(f"  Mean: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"  Max: {max(scores)}")
    print(f"  Min: {min(scores)}")

    return scores


# ============================================
# 主函数
# ============================================
if __name__ == "__main__":
    # 导入GameInterface
    from GameInterface import GameInterface

    print("Initializing...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")

    # 创建环境和agent
    env = GameInterface()
    agent = CNN_DQN_Agent(action_dim=16)
    memory = ReplayMemory()

    # 测试一步确认状态形状
    env.reset()
    state, _, _ = env.next(0)
    print(f"GameInterface state shape: {state.shape}")
    print(f"After preprocessing: {agent.preprocess_state(state).shape}\n")

    # 训练
    scores, losses = train_cnn_dqn(env, agent, memory, num_episodes=2000)

    # 测试
    test_cnn_dqn(env, agent, num_episodes=10)
