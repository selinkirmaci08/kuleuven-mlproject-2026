#!/usr/bin/env python3
"""
evaluate_all_distortions.py — Run evaluation for each distortion level and save results.

Usage:
    python evaluate_all_distortions.py --load submission_vector.py --episodes 10
"""

import subprocess
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_DIR     = os.path.dirname(os.path.abspath(__file__))

# ── Config ────────────────────────────────────────────────────────────────────
DISTORTION_LEVELS = [0, 1, 2, 3, 4, 5]
EPISODES          = 1    # episodes per distortion level
SEED              = 42
AGENT             = "submission_vector.py"  # change if needed
OUTPUT_FILE       = os.path.join(_DIR, "evaluation_results.json")


def run_evaluation(distortion: int, episodes: int, seed: int, agent: str) -> dict:
    """Run evaluation.py for a single distortion level and parse results."""
    tmp_output = os.path.join(_DIR, f"tmp_dist{distortion}.jsonl")

    cmd = [
        sys.executable, "evaluation.py",
        "--load", agent,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--distortion", str(distortion),
        "--output", tmp_output,
    ]

    print(f"\n{'='*60}")
    print(f"  Evaluating distortion level {distortion}...")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=_DIR)

    # Read the jsonl output
    if os.path.exists(tmp_output):
        with open(tmp_output, "r") as f:
            data = json.loads(f.readline())
        os.remove(tmp_output)
        return data
    else:
        print(f"[WARNING] No output file for distortion {distortion}")
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",     type=str, default=AGENT)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed",     type=int, default=SEED)
    parser.add_argument("--output",   type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    all_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "agent": args.load,
        "episodes_per_level": args.episodes,
        "seed": args.seed,
        "results_per_distortion": {}
    }

    for dist in DISTORTION_LEVELS:
        result = run_evaluation(dist, args.episodes, args.seed, args.load)
        all_results["results_per_distortion"][str(dist)] = result

        # Print summary for this level
        avg = result.get("avg_reward", "N/A")
        per_agent = result.get("avg_reward_per_agent", {})
        print(f"\n  Distortion {dist} summary:")
        print(f"    avg_reward: {avg}")
        for agent, reward in per_agent.items():
            print(f"    {agent}: {reward}")

    # Save combined results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  All results saved to {args.output}")
    print(f"{'='*60}")

    # Print final summary table
    print("\nSummary:")
    print(f"{'Distortion':<15} {'Avg Reward':<15} {'archer_0':<15} {'archer_1':<15}")
    print("-" * 60)
    for dist in DISTORTION_LEVELS:
        r = all_results["results_per_distortion"].get(str(dist), {})
        avg = r.get("avg_reward", "N/A")
        per_agent = r.get("avg_reward_per_agent", {})
        a0 = per_agent.get("archer_0", "N/A")
        a1 = per_agent.get("archer_1", "N/A")
        print(f"{dist:<15} {avg:<15} {a0:<15} {a1:<15}")


if __name__ == "__main__":
    main()