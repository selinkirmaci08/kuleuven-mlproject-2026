#!/usr/bin/env python3
"""
evaluate_baseline_diagonal.py — Evaluate a shoot-diagonally baseline in KAZ.

The agent rotates clockwise for a fixed number of steps to reach a diagonal
angle, then always shoots. This simulates a simple heuristic agent that fires
in a fixed diagonal direction without any learning.

Usage:
    python evaluate_baseline_diagonal.py --episodes 10 --distortion 0
    python evaluate_baseline_diagonal.py --episodes 1  --distortion 0 --screen
"""

import argparse
import random
import time
import json
import os
import numpy as np
import pygame

from utils import create_environment
from submission_vector import CustomWrapper

_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_ROTATE_CW = 3  # rotate
ACTION_SHOOT     = 4  # shoot


# ── Baseline ──────────────────────────────────────────────────────────────────

class ShootDiagonalPredict:
    """
    Rotates clockwise for ROTATE_STEPS steps to reach a diagonal angle,
    then always shoots. Resets per episode.
    """
    ROTATE_STEPS = 4  # adjust to change the diagonal angle

    def __init__(self):
        self._step_count = {}

    def reset(self):
        self._step_count = {}

    def __call__(self, obs, agent, env=None):
        count = self._step_count.get(agent, 0)
        self._step_count[agent] = count + 1
        if count < self.ROTATE_STEPS:
            return ACTION_ROTATE_CW
        return ACTION_SHOOT


# ── Pygame event handler ──────────────────────────────────────────────────────

def handle_pygame_events():
    """Return True if user pressed Q or closed the window."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            return True
    return False


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(predict_fn, num_episodes, distortion, seed, render_mode=None, max_steps=None):
    random.seed(seed)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_episodes)]

    env = create_environment(distortion_level=distortion, render_mode=render_mode)
    env = CustomWrapper(env, inference_mode=False)

    rewards    = {agent: 0 for agent in env.possible_agents}
    ep_lengths = []
    start_time = time.time()
    do_terminate = False

    for ep, ep_seed in enumerate(seeds):
        env.reset(seed=ep_seed)
        predict_fn.reset()
        step_count = 0

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            step_count += 1

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                ep_lengths.append(step_count)
                break

            if max_steps and step_count > max_steps:
                ep_lengths.append(step_count)
                break

            if render_mode == "human" and handle_pygame_events():
                do_terminate = True
                break

            if do_terminate:
                break

            env.step(predict_fn(obs, agent, env))

        print(f"  Episode {ep + 1}/{num_episodes} — steps: {step_count}")

        if do_terminate:
            break

    env.close()
    total_time = time.time() - start_time

    avg_reward    = sum(rewards.values()) / num_episodes
    avg_per_agent = {a: rewards[a] / num_episodes for a in env.possible_agents}
    avg_ep_length = float(np.mean(ep_lengths)) if ep_lengths else 0

    print(f"\nEvaluation Results (Shoot Diagonal Baseline):")
    print(f"  Distortion level:   {distortion}")
    print(f"  Total episodes:     {num_episodes}")
    print(f"  Avg reward:         {avg_reward:.2f}")
    for a, r in avg_per_agent.items():
        print(f"  {a}: {r:.2f}")
    print(f"  Avg episode length: {avg_ep_length:.1f}")
    print(f"  Total time:         {total_time:.1f}s")

    return {
        "baseline": "Shoot Diagonal",
        "distortion": distortion,
        "avg_reward": avg_reward,
        "avg_reward_per_agent": avg_per_agent,
        "avg_episode_length": avg_ep_length,
        "total_episodes": num_episodes,
        "evaluation_time": total_time,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",   type=int, default=10)
    parser.add_argument("--distortion", type=int, default=0, choices=range(6))
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max-steps",  type=int, default=None)
    parser.add_argument("--screen", "-s", action="store_true",
                        help="Enable visual rendering (press Q to quit)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_DIR, "baseline_diagonal_results.json"))
    args = parser.parse_args()

    render_mode = "human" if args.screen else None
    if render_mode == "human":
        print("Press Q or close window to stop early.")

    result = evaluate(
        ShootDiagonalPredict(),
        num_episodes=args.episodes,
        distortion=args.distortion,
        seed=args.seed,
        render_mode=render_mode,
        max_steps=args.max_steps,
    )

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()