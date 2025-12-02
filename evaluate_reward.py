import os
import numpy as np

import paddle
from DQN import Agent, RandomAgent, build_model, evaluate
from GameInterface import GameInterface
from PRNG import PRNG

evaluate_random = PRNG()
evaluate_random.seed("RedContritio")

# Checkpoint paths for the three reward modes
CHECKPOINTS = {
    1: "reward1.pdparams",
    2: "reward2.pdparams",
    3: "reward3.pdparams",
}

if __name__ == "__main__":
    EVALUATE_TIMES = 200

    feature_map_height = GameInterface.FEATURE_MAP_HEIGHT
    feature_map_width = GameInterface.FEATURE_MAP_WIDTH

    action_dim = GameInterface.ACTION_NUM
    feature_dim = feature_map_height * feature_map_width * 2
    e_greed = 0.0          # pure greedy during evaluation
    e_greed_decrement = 0  # no decay needed

    # One env per reward mode, so each agent is evaluated with the
    # reward function it was trained on (even though we only print scores).
    envs = {
        mode: GameInterface(reward_mode=mode)
        for mode in CHECKPOINTS.keys()
    }

    # Create agents and load their corresponding checkpoints
    agents = {}
    for mode, ckpt_path in CHECKPOINTS.items():
        agent = Agent(build_model, feature_dim, action_dim, e_greed, e_greed_decrement)
        if os.path.exists(ckpt_path):
            agent.policy_net.set_state_dict(paddle.load(ckpt_path))
            print(f"[INFO] Loaded checkpoint for reward_mode {mode} from '{ckpt_path}'.")
            agents[mode] = agent
        else:
            print(f"[WARN] Checkpoint '{ckpt_path}' not found. Skipping reward_mode {mode}.")

    # Random agent & its env (reward_mode can be anything; we only care about score)
    random_env = GameInterface(reward_mode=1)
    random_agent = RandomAgent(GameInterface.ACTION_NUM)

    # Containers for scores
    scores_by_mode = {mode: [] for mode in agents.keys()}
    random_scores = []

    for _ in range(EVALUATE_TIMES):
        seed = evaluate_random.random()

        # Evaluate each trained agent (one episode per model per seed)
        for mode, agent in agents.items():
            env = envs[mode]
            score, _ = evaluate(env, agent, seed)
            scores_by_mode[mode].append(score)

        # Evaluate random agent once per seed
        score_rnd, _ = evaluate(random_env, random_agent, seed)
        random_scores.append(score_rnd)

    # Print summary: ONLY scores (no rewards)
    for mode, scores in scores_by_mode.items():
        if len(scores) == 0:
            print(f"[DQN Agent - reward_mode {mode}]\t: no valid evaluations (no scores).")
            continue

        print(
            f"[DQN Agent - reward_mode {mode}]\t:"
            f"\tmean_score: {np.mean(scores)},"
            f"\tmax_score: {np.max(scores)},"
            f"\tmin_score: {np.min(scores)}"
        )

    if len(random_scores) > 0:
        print(
            f"[Random Agent]\t\t\t:"
            f"\tmean_score: {np.mean(random_scores)},"
            f"\tmax_score: {np.max(random_scores)},"
            f"\tmin_score: {np.min(random_scores)}"
        )
    else:
        print("[Random Agent]\t\t\t: no valid evaluations (no scores).")
