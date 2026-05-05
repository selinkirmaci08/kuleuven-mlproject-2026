#!/usr/bin/env python3
# encoding: utf-8
"""
This file contains agents training using Ray RLLib.
The PPO algorithm is adapted here from example_training_rllib.py.

"""

from pathlib import Path

import pettingzoo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env
import numpy as np
import torch
import supersuit as ss

from utils import create_environment
from submission_m import CustomWrapper

def algo_config(id_env):
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=3)
        .multi_agent(
            #policies={x for x in policies},
            policies = ["archer"],  # one shared policy
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "archer",
            #policies_to_train=policies_to_train,
            policies_to_train = ["archer"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    #x: RLModuleSpec
                    "archer": RLModuleSpec(module_class=PPOTorchRLModule, model_config={
                                                                    "conv_filters": [
                                                                        [16, 8, 4],
                                                                        [32, 4, 2],
                                                                        [32, 3, 1],
                                                                    ],
                                                                    "fcnet_hiddens": [256],
                                                                    "conv_activation": "relu",
                                                                    "fcnet_hiddens": [512],
                                                                    "fcnet_activation": "relu",
                                                                })
                    #if x in policies_to_train
                    #else
                    #RLModuleSpec(module_class=RandomRLModule)
                    #for x in policies
                    },
            ))
        .training(
            train_batch_size=2048,
            minibatch_size=256,
            num_epochs=10,         # PPO default, make sure it's not 1
            lr=1e-4,               # lower LR for pixel-based policies
            entropy_coeff=0.02,    # slightly higher to maintain exploration
        )
        .debugging(log_level="ERROR")

    )

    return config


""" def training(env, checkpoint_path, max_iterations = 500):

    # Translating the PettingZoo environment to an RLLib environment.
    # Note: RLLib use a parallelized version of the environment.
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)

    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm

    # TODO: check each agent has their own policy, should they?
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    # Train the model
    algo = config.build()
    print("Algo built OK, starting training...")
    for i in range(max_iterations):
        print(f"Starting iteration {i}...")
        result = algo.train()
        result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"])
            #if result["env_runners"]["agent_episode_returns_mean"]["archer_0"] > 5: # Or any early stopping criterion
            #    break
        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            ) """

def training(_, checkpoint_path, max_iterations=500):
    id_env = "knights_archers_zombies_v10"

    def make_env(_config):
        e = create_environment(max_cycles=200)
        e = ss.reward_lambda_v0(e, lambda r: r * 10 + 0.001)
        e = CustomWrapper(e)
        return ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(e))

    register_env(id_env, make_env)

    np.random.seed(42)
    torch.manual_seed(42)

    config = algo_config(id_env)
    algo = config.build()

    reward_history = []

    print("Algo built OK, starting training...")
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"])
            agent_means = result["env_runners"]["agent_episode_returns_mean"]
            mean_reward = np.mean(list(agent_means.values()))
            reward_history.append(mean_reward)
            window = 10
            if len(reward_history) >= window:
                moving_avg = np.mean(reward_history[-window:])
            else:
                moving_avg = np.mean(reward_history)
            print(f"{i} | mean: {mean_reward:.2f} | trend({window}): {moving_avg:.2f}")
            
        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            print(f"Checkpoint saved: {save_result.checkpoint.path}")

if __name__ == "__main__":
    checkpoint_path = str(Path("results").resolve())
    training(None, checkpoint_path, max_iterations=500)

    # Create the PettingZoo environment for training
    #env = create_environment(max_cycles=500)

    # reward shaping, not allowed in evaluation only in training
    #env = ss.reward_lambda_v0(env, lambda r: r * 10 + 0.001)

    #env = CustomWrapper(env)

    # Running training routine
    #checkpoint_path = str(Path("results").resolve())
    #training(env, checkpoint_path, max_iterations = 500)
