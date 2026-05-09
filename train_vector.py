#!/usr/bin/env python3
# encoding: utf-8
"""
train_vector.py — Training script for vector policy with relative observations.

Observation vector (24 floats):
  [archer_0_x, archer_0_y, archer_0_dir_x, archer_0_dir_y,
   z0_align_0, z0_dist_0, z1_align_0, z1_dist_0,
   z2_align_0, z2_dist_0, z3_align_0, z3_dist_0,
   archer_1_x, archer_1_y, archer_1_dir_x, archer_1_dir_y,
   z0_align_1, z0_dist_1, z1_align_1, z1_dist_1,
   z2_align_1, z2_dist_1, z3_align_1, z3_dist_1]

align: 1.0 = zombie directly in front of archer (shoot now!)
dist:  0.0 = zombie very close, 1.0 = far away

Uses OLD RLlib API stack. Sprite rects during training (fast, perfect).
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
from submission_vector import CustomWrapper, unwrap_to_raw


def make_env():
    """Factory — each Ray worker gets its own env. Uses sprite rects (fast)."""
    env = create_environment()
    env = CustomWrapper(env, inference_mode=False)
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
                # RLlib infers input size (24) from observation space automatically
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
            entropy_coeff=0.01,   # higher entropy to encourage shooting
        )
        .debugging(log_level="ERROR")
    )
    return config


def get_returns(result):
    return result.get("env_runners", {}).get("policy_reward_mean", {})


def training(env, checkpoint_path, max_iterations=500):

    id_env = "knights_archers_zombies_v10_vector"
    register_env(id_env, lambda config: make_env())

    policies = list(env.agents)
    config   = algo_config(id_env, policies, policies)
    algo     = config.build()

    # ── Sanity check ─────────────────────────────────────────────────────────
    check_env = create_environment()
    check_env = CustomWrapper(check_env, inference_mode=False)
    check_env.reset()
    for agent in check_env.agent_iter(max_iter=50):
        obs, rew, term, trunc, info = check_env.last()
        check_env.step(check_env.action_space(agent).sample())

    test_obs = check_env.observe(check_env.agents[0])
    raw      = unwrap_to_raw(check_env)

    print("\n[SANITY CHECK] Observation after 50 steps (24 floats):")
    print(f"  archer_0  x={test_obs[0]:.3f}  y={test_obs[1]:.3f}  "
          f"dir_x={test_obs[2]:.3f}  dir_y={test_obs[3]:.3f}")
    for slot in range(4):
        align = test_obs[4 + slot*2]
        dist  = test_obs[4 + slot*2 + 1]
        print(f"  zombie_{slot} (rel to archer_0)  align={align:.3f}  dist={dist:.3f}")
    print(f"  archer_1  x={test_obs[12]:.3f}  y={test_obs[13]:.3f}  "
          f"dir_x={test_obs[14]:.3f}  dir_y={test_obs[15]:.3f}")
    if hasattr(raw, "zombie_list"):
        print(f"  [raw] zombie count: {len(list(raw.zombie_list))}")
    check_env.close()
    # ─────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  Vector policy training — KAZ")
    print("  Observation: 24 floats (abs pos + relative zombie alignment)")
    print("  Agents:", policies)
    print("=" * 60)

    best_avg_reward = -float("inf")

    for i in range(max_iterations):
        result = algo.train()
        result.pop("config", None)

        returns = get_returns(result)
        if returns:
            r0 = returns.get("archer_0", float("nan"))
            r1 = returns.get("archer_1", float("nan"))
            avg = (r0 + r1) / 2
            print(f"iter {i:03d} | archer_0: {r0:7.3f}  |  archer_1: {r1:7.3f}")
        else:
            print(f"iter {i:03d} | (no reward data yet)")

        if i < 50:
            if i % 5 == 0:
                save_result = algo.save(checkpoint_path)
                print(f"         └─  {avg:.3f} — checkpoint saved → {save_result.checkpoint.path}")
        else:
            if avg > best_avg_reward:
                best_avg_reward = avg
                save_result = algo.save(checkpoint_path)
                print(f"         └─ NEW BEST {avg:.3f} — checkpoint saved → {save_result.checkpoint.path}")

    print("=" * 60)
    save_result = algo.save(checkpoint_path)
    print(f"Final checkpoint → {save_result.checkpoint.path}")
    algo.stop()


if __name__ == "__main__":
    ray.init()

    env = create_environment()
    env = CustomWrapper(env, inference_mode=False)
    env.reset()

    checkpoint_path = str(Path("results_vector").resolve())
    training(env, checkpoint_path, max_iterations=500)