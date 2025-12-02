# BC pretrain pipline

## 1. DQN — Save Offline Replay Buffer
Simply run `DQN.py`.  
The offline replay buffer will be automatically saved to:
`buffer/replay_buffer_final.pt`

## 2. Pretrain BC
Run `pretrain_bc.py`.  
The pretrained BC policy weights will be saved to:
`BC/BC_pretrained_policy.pt`