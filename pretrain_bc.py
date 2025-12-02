"""
bc_pretrain.py

从 replay_buffer.pt 中提取 (state, action) 做 Behavior Cloning (BC) 预训练，
训练一个 policy 网络，输出 BC_pretrained_policy.pt。
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



REPLAY_BUFFER_PATH = "buffer/replay_buffer_final.pt"
SAVE_MODEL_PATH = "BC/BC_pretrained_policy.pt"

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3


class BCDataset(Dataset):
    def __init__(self, buffer):
        self.states = []
        self.actions = []

        print("[INFO] Loading transitions for BC...")
        for (s, a, r, ns, alive) in buffer:

            # ---- 统一 action 为 int ----
            if hasattr(a, "shape") and a.shape == (1,):
                a = a.item()
            elif hasattr(a, "item"):
                a = a.item()
            a = int(a)

            self.states.append(torch.tensor(s, dtype=torch.float32))
            self.actions.append(torch.tensor(a, dtype=torch.long))

        self.states = torch.stack(self.states)
        self.actions = torch.stack(self.actions)

        print(f"[INFO] Total BC samples: {len(self.states)}")
        print(f"[INFO] State dim: {self.states[0].shape}, Action dim: {self.actions.max().item() + 1}")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


# Policy Model (MLP)
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)



def main():
    print(f"[INFO] Loading replay buffer from {REPLAY_BUFFER_PATH} ...")
    buffer = torch.load(REPLAY_BUFFER_PATH,weights_only=False)
    print(f"[INFO] Loaded transitions: {len(buffer)}")

    # --- 构建 BC 数据集 ---
    dataset = BCDataset(buffer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.max().item() + 1  # 动作是 0~N−1

    print(f"[INFO] state_dim={state_dim}, action_dim={action_dim}")

    # --- 初始化模型 ---
    model = PolicyNetwork(state_dim, action_dim)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()


    print("[INFO] Start BC pretraining...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss={avg_loss:.4f}")

    # save model
    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_MODEL_PATH)
    print(f"[INFO] BC pretrained policy saved → {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
