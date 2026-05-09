#!/usr/bin/env python3
# encoding: utf-8
"""
This file contains an example of training your agents using Ray RLLib.
Notice that this code will not learn properlu, as no preprocessing of the data nor adaptation of the default PPO algorithm is done here.

"""

from pathlib import Path

import numpy as np
import pettingzoo
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.tune.registry import register_env

from submission_vector import CustomWrapper, VectorPPORLModule
from utils import create_environment


def algo_config(id_env, policies, policies_to_train):


    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=id_env, disable_env_checking=True)
        .env_runners(num_env_runners=1)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(module_class=VectorPPORLModule)
                    if x in policies_to_train
                    else RLModuleSpec(module_class=RandomRLModule)
                    for x in policies
                },
            )
        )
        .training(
            model={
                "fcnet_hiddens": [128, 128, 64],
                "fcnet_activation": "relu",
            },
            train_batch_size=4000,
            minibatch_size=512,
            num_epochs=10,
            lr=1e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.001,
        )
        .debugging(log_level="ERROR")

    )

    return config


def training(env, checkpoint_path, max_iterations = 500):

    # Translating the PettingZoo environment to an RLLib environment.
    # Note: RLLib use a parallelized version of the environment.
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    id_env = "knights_archers_zombies_v10"
    register_env(id_env, lambda config: rllib_env)

    # Fix seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(id_env, policies, policies_to_train)

    # Train the model
    algo = config.build()
    for i in range(max_iterations):
        result = algo.train()
        result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            print(i, result["env_runners"]["agent_episode_returns_mean"])

        if i % 5 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )



if __name__ == "__main__":


    # Create the PettingZoo environment for training
    env = create_environment()
    env = CustomWrapper(env)

    # Running training routine
    checkpoint_path = str(Path("results").resolve())
    training(env, checkpoint_path, max_iterations = 200)
