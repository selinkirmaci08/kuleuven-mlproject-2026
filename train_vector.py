#!/usr/bin/env python3
# encoding: utf-8
"""
train_vector.py — Training script for Option A: Vector-only policy.

Observation vector (20 floats):
  [archer_0_x, archer_0_y, archer_0_dir_x, archer_0_dir_y,
   archer_1_x, archer_1_y, archer_1_dir_x, archer_1_dir_y,
   z0_cx, z0_cy, z0_conf,
   z1_cx, z1_cy, z1_conf,
   z2_cx, z2_cy, z2_conf,
   z3_cx, z3_cy, z3_conf]

Uses the OLD RLlib API stack — avoids KeyError: 'advantages' with custom
RLModules on this version of Ray. Built-in PPO MLP is equivalent to
VectorPPORLModule and is configured via fcnet_hiddens.
"""

from pathlib import Path

import ray
import pettingzoo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
import numpy as np
import torch

from utils import create_environment
from submission_vector import CustomWrapper


def make_env():
    """Factory function — each Ray worker creates its own env instance."""
    env = create_environment()
    env = CustomWrapper(env)
    return ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))


def algo_config(id_env, policies, policies_to_train):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=4)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .training(
            model={
                # RLlib infers input size (20) from observation space automatically
                "fcnet_hiddens": [128, 128, 64],
                "fcnet_activation": "relu",
            },
            train_batch_size=4000,
            minibatch_size=512,
            num_sgd_iter=5,
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.001,
        )
        .debugging(log_level="ERROR")
    )
    return config


def get_returns(result):
    """Extract per-agent mean rewards — old API stack uses policy_reward_mean."""
    return result.get("env_runners", {}).get("policy_reward_mean", {})


def training(env, checkpoint_path, max_iterations=200):

    id_env = "knights_archers_zombies_v10_vector"
    register_env(id_env, lambda config: make_env())

    np.random.seed(42)
    torch.manual_seed(42)

    policies = list(env.agents)
    config = algo_config(id_env, policies, policies)
    algo = config.build()

    # ── Sanity check ─────────────────────────────────────────────────────────
    from submission_vector import unwrap_to_raw
    check_env = create_environment()
    check_env = CustomWrapper(check_env)
    check_env.reset()
    for agent in check_env.agent_iter(max_iter=50):
        obs, rew, term, trunc, info = check_env.last()
        check_env.step(check_env.action_space(agent).sample())
    test_obs = check_env.observe(check_env.agents[0])
    raw = unwrap_to_raw(check_env)
    print("\n[SANITY CHECK] Observation after 50 steps:")
    print(f"  archer_0  x={test_obs[0]:.3f}  y={test_obs[1]:.3f}  dir_x={test_obs[2]:.3f}  dir_y={test_obs[3]:.3f}")
    print(f"  archer_1  x={test_obs[4]:.3f}  y={test_obs[5]:.3f}  dir_x={test_obs[6]:.3f}  dir_y={test_obs[7]:.3f}")
    print(f"  zombie_0  cx={test_obs[8]:.3f}   cy={test_obs[9]:.3f}   conf={test_obs[10]:.3f}")
    print(f"  zombie_1  cx={test_obs[11]:.3f}   cy={test_obs[12]:.3f}   conf={test_obs[13]:.3f}")
    print(f"  zombie_2  cx={test_obs[14]:.3f}   cy={test_obs[15]:.3f}   conf={test_obs[16]:.3f}")
    print(f"  zombie_3  cx={test_obs[17]:.3f}   cy={test_obs[18]:.3f}   conf={test_obs[19]:.3f}")
    if hasattr(raw, "archer_list"):
        for i, archer in enumerate(raw.archer_list):
            print(f"  [raw] archer_{i} angle={archer.angle}  direction=({archer.direction.x:.2f}, {archer.direction.y:.2f})")
    if hasattr(raw, "zombie_list"):
        print(f"  [raw] zombie count: {len(list(raw.zombie_list))}")
    check_env.close()
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 55)
    print("  Vector policy training — KAZ (old API stack)")
    print("  Observation: 20 floats (pos + dir + zombies)")
    print("  Agents:", policies)
    print("=" * 55)

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        returns = get_returns(result)

        if returns:
            r0 = returns.get("archer_0", float("nan"))
            r1 = returns.get("archer_1", float("nan"))
            print(f"iter {i:03d} | archer_0: {r0:7.3f}  |  archer_1: {r1:7.3f}")
        else:
            print(f"iter {i:03d} | (no reward data yet)")

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            print(f"         └─ checkpoint saved → {save_result.checkpoint.path}")


    print("=" * 55)
    save_result = algo.save(checkpoint_path)
    print(f"Final checkpoint → {save_result.checkpoint.path}")
    algo.stop()


if __name__ == "__main__":
    ray.init()

    env = create_environment()
    env = CustomWrapper(env)
    env.reset()

    checkpoint_path = str(Path("results_vector").resolve())
    training(env, checkpoint_path, max_iterations=200)