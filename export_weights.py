"""
export_weights.py — Convert RLlib old API stack checkpoint to plain .pt files.

Reads policy_state.pkl from results_vector/policies/<policy_id>/
and saves weights as results_vector/<policy_id>_weights.pt

These .pt files are Python-version independent and work on any machine.

Usage:
    python export_weights.py
    python export_weights.py --checkpoint results_vector
"""

import argparse
import os
import pickle
import torch


def export(checkpoint_path: str, policies: list = None):
    if policies is None:
        policies = ["archer_0", "archer_1"]

    for policy_id in policies:
        pkl_path = os.path.join(
            checkpoint_path, "policies", policy_id, "policy_state.pkl"
        )

        if not os.path.exists(pkl_path):
            print(f"[SKIP] {pkl_path} not found")
            continue

        print(f"[Load] {pkl_path}")
        with open(pkl_path, "rb") as f:
            policy_state = pickle.load(f)

        weights = policy_state.get("weights", {})

        if not weights:
            print(f"[WARN] No weights found in {pkl_path}")
            continue

        print(f"  Keys: {list(weights.keys())}")

        # Convert numpy arrays to torch tensors
        pt_weights = {}
        for k, v in weights.items():
            pt_weights[k] = torch.tensor(v) if not isinstance(v, torch.Tensor) else v

        out_path = os.path.join(checkpoint_path, f"{policy_id}_weights.pt")
        torch.save(pt_weights, out_path)
        print(f"[Save] {out_path}")

    print("\nDone. Upload these .pt files to the department computer along with submission.py:")
    for policy_id in policies:
        print(f"  {os.path.join(checkpoint_path, policy_id + '_weights.pt')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results_vector",
                        help="Path to RLlib checkpoint directory")
    args = parser.parse_args()
    export(args.checkpoint)