# extract_weights.py  — run this LOCALLY, not on department machines
import pickle, os, torch

CHECKPOINT_PATH = "results_vector"  # your local path

def extract_and_save(checkpoint_path, policy_id):
    pkl_path = os.path.join(checkpoint_path, "policies", policy_id, "policy_state.pkl")
    with open(pkl_path, "rb") as f:
        policy_state = pickle.load(f)

    weights = policy_state.get("weights", {})
    print(f"[{policy_id}] Keys: {list(weights.keys())}")

    out_path = os.path.join(checkpoint_path, f"{policy_id}_weights.pt")
    torch.save(weights, out_path)
    print(f"[{policy_id}] Saved to {out_path}")

extract_and_save(CHECKPOINT_PATH, "archer_0")
extract_and_save(CHECKPOINT_PATH, "archer_1")