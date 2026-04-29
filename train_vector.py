#!/usr/bin/env python3
"""
train_vector.py — Option A vector-only policy, using OLD RLlib API stack.
The old stack handles advantage computation automatically.
"""
from pathlib import Path
import pettingzoo
import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from utils import create_environment
from submission_vector import CustomWrapper


def training(env, checkpoint_path, max_iterations=300):
    rllib_env = ParallelPettingZooEnv(
        pettingzoo.utils.conversions.aec_to_parallel(env)
    )
    id_env = "kaz_vector_v0"
    register_env(id_env, lambda config: rllib_env)

    np.random.seed(42)
    torch.manual_seed(42)

    policies = list(env.agents)

    config = (
        PPOConfig()
        # ── Use OLD API stack — handles advantages automatically ──────────────
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=1)
        .multi_agent(
            policies=set(policies),
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies,
        )
        .training(
            model={
                "fcnet_hiddens": [128, 128, 64],
                "fcnet_activation": "relu",
            },
            train_batch_size=512,
            lr=3e-4,
            gamma=0.99,
        )
        .debugging(log_level="ERROR")
    )

    algo = config.build()
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        # Extract reward means
        env_runners = result.get("env_runners", {})
        means = env_runners.get("policy_reward_mean", {})
        ep_mean = env_runners.get("episode_reward_mean", 0)
        ep_max  = env_runners.get("episode_reward_max", 0)
        vf0 = result.get("info", {}).get("learner", {}).get("archer_0", {}).get("learner_stats", {}).get("vf_explained_var", 0)

        a0 = means.get("archer_0", 0)
        a1 = means.get("archer_1", 0)
        print(f"[{i:03d}] reward: a0={a0:.2f}  a1={a1:.2f}  total_mean={ep_mean:.2f}  best={ep_max:.1f}  vf_var={vf0:.2f}")

        if i % 5 == 0:
            algo.save(checkpoint_path)
            print(f"  → checkpoint saved")

        if a0 > 5 or ep_mean > 8:
            print("Early stopping criterion met.")
            break

    algo.save(checkpoint_path)
    print("Training complete.")


if __name__ == "__main__":
    env = create_environment()
    env = CustomWrapper(env)
    training(env, str(Path("results_vector").resolve()), max_iterations=300)