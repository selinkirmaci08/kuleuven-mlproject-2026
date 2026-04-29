"""
submission_vector.py  —  Option A: Vector-only policy (ablation experiment).

NO pixel observations. The policy receives only:
  - Archer positions + facing direction vector (exact, from sprite rects)
  - Zombie positions + confidence (from sprite rects during training, YOLO at inference)

Vector layout (20 floats total):
  [archer_0_x, archer_0_y, archer_0_dir_x, archer_0_dir_y,   ← 4
   archer_1_x, archer_1_y, archer_1_dir_x, archer_1_dir_y,   ← 4
   z0_cx, z0_cy, z0_conf,                                     ← 3
   z1_cx, z1_cy, z1_conf,
   z2_cx, z2_cy, z2_conf,
   z3_cx, z3_cy, z3_conf]                                     → 20 total
"""

from pathlib import Path
from typing import Callable, Dict, Any
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from gymnasium import spaces


ENV_SETTINGS = {
    "frame_stack": None,
    "resize_dim": None,
}

# ── Dimensions ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 720
MAX_ZOMBIES        = 4
ARCHER_DIM         = 8           # 2 archers × (x, y, dir_x, dir_y)
ZOMBIE_DIM         = MAX_ZOMBIES * 3
VEC_DIM            = ARCHER_DIM + ZOMBIE_DIM   # 20
NUM_ACTIONS        = 6           # KAZ action space size

YOLO_WEIGHTS       = "runs/detect/train/weights/best.pt"
CHECKPOINT_PATH    = "results_vector"


# ── YOLO singleton (inference only) ──────────────────────────────────────────

_yolo_model = None

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[VectorPolicy] YOLO loaded from {YOLO_WEIGHTS}")
        except Exception as e:
            print(f"[VectorPolicy] YOLO not available ({e}), zombie slots will be zeros")
    return _yolo_model


# ── Feature extraction ────────────────────────────────────────────────────────

def unwrap_to_raw(env):
    """Follow wrapper chain to find the raw KAZ env with archer_list."""
    cur, seen = env, set()
    while cur is not None:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "archer_list"):
            return cur
        nxt = getattr(cur, "aec_env", None) or getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def get_archer_positions(env) -> np.ndarray:
    """
    Archer positions + facing direction, normalised to [0, 1].
    Layout: [x0, y0, dir_x0, dir_y0, x1, y1, dir_x1, dir_y1]
    direction components normalised from [-1, 1] → [0, 1].
    Dead archers get zeros.
    """
    pos = np.zeros(ARCHER_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "archer_list"):
        return pos
    alive = env.agents
    for i, archer in enumerate(raw.archer_list):
        if f"archer_{i}" not in alive:
            continue
        r = archer.rect
        pos[4*i]   = np.clip((r.x + r.width  / 2) / SCREEN_W, 0.0, 1.0)
        pos[4*i+1] = np.clip((r.y + r.height / 2) / SCREEN_H, 0.0, 1.0)
        pos[4*i+2] = (float(archer.direction.x) + 1.0) / 2.0
        pos[4*i+3] = (float(archer.direction.y) + 1.0) / 2.0
    return pos


def get_zombie_positions_from_sprites(env) -> np.ndarray:
    """
    Zombie positions from sprite rects — used during TRAINING.
    Confidence is always 1.0 (ground truth). Missing slots are zeros.
    """
    vec = np.zeros(ZOMBIE_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "zombie_list"):
        return vec
    for slot, zombie in enumerate(list(raw.zombie_list)[:MAX_ZOMBIES]):
        r = zombie.rect
        vec[slot*3]   = np.clip((r.x + r.width  / 2) / SCREEN_W, 0.0, 1.0)
        vec[slot*3+1] = np.clip((r.y + r.height / 2) / SCREEN_H, 0.0, 1.0)
        vec[slot*3+2] = 1.0
    return vec


def get_zombie_positions_yolo(raw_obs: np.ndarray) -> np.ndarray:
    """
    YOLO zombie detections — used at INFERENCE time only.
    Returns flat (12,) vector [cx, cy, conf] per slot. Missing slots are zeros.
    """
    vec   = np.zeros(ZOMBIE_DIM, dtype=np.float32)
    model = get_yolo_model()
    if model is None:
        return vec
    try:
        results = model(raw_obs, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return vec
        xyxy  = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for slot, idx in enumerate(np.argsort(-confs)[:MAX_ZOMBIES]):
            x1, y1, x2, y2 = xyxy[idx]
            vec[slot*3]   = np.clip(((x1+x2)/2) / SCREEN_W, 0.0, 1.0)
            vec[slot*3+1] = np.clip(((y1+y2)/2) / SCREEN_H, 0.0, 1.0)
            vec[slot*3+2] = np.clip(float(confs[idx]),       0.0, 1.0)
    except Exception as e:
        print(f"[VectorPolicy] YOLO error: {e}")
    return vec


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Vector-only wrapper. observe() returns a flat (20,) float32 vector — NO image.

    Layout:
      [archer_0_x, archer_0_y, archer_0_dir_x, archer_0_dir_y,
       archer_1_x, archer_1_y, archer_1_dir_x, archer_1_dir_y,
       z0_cx, z0_cy, z0_conf,
       z1_cx, z1_cy, z1_conf,
       z2_cx, z2_cy, z2_conf,
       z3_cx, z3_cy, z3_conf]
    """

    def __init__(self, env):
        self._unwrapped = env
        self._raw_env   = env
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(VEC_DIM,), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType:
        archer_pos = get_archer_positions(self._unwrapped)               # (8,)
        zombie_pos = get_zombie_positions_from_sprites(self._unwrapped)  # (12,)
        return np.concatenate([archer_pos, zombie_pos])                  # (20,)

    def observe_inference(self, agent: AgentID) -> ObsType:
        """Used at inference — runs YOLO on pixels for zombie positions."""
        raw_obs    = self._raw_env.observe(agent)
        archer_pos = get_archer_positions(self._unwrapped)
        zombie_pos = get_zombie_positions_yolo(raw_obs)
        return np.concatenate([archer_pos, zombie_pos])                  # (20,)


# ── Pure PyTorch policy network ───────────────────────────────────────────────

class PolicyNet(nn.Module):
    """
    Mirrors the MLP architecture used during training (fcnet_hiddens=[128,128,64]).
    Used for inference only — no RLlib dependency.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VEC_DIM, 128), nn.ReLU(),
            nn.Linear(128,     128), nn.ReLU(),
            nn.Linear(128,      64), nn.ReLU(),
            nn.Linear(64, NUM_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_policy_weights(checkpoint_path: str, policy_id: str) -> PolicyNet:
    """
    Extract policy weights from an RLlib old-API-stack checkpoint and load
    them into a plain PyTorch PolicyNet — no RLlib or Ray needed at runtime.
    """
    import pickle, os

    # RLlib old API stack saves policies under:
    # <checkpoint>/policies/<policy_id>/policy_state.pkl
    policy_state_path = os.path.join(
        checkpoint_path, "policies", policy_id, "policy_state.pkl"
    )

    if not os.path.exists(policy_state_path):
        raise FileNotFoundError(
            f"Policy state not found at {policy_state_path}. "
            f"Make sure the checkpoint path is correct and training completed."
        )

    with open(policy_state_path, "rb") as f:
        policy_state = pickle.load(f)

    # Weights are stored under weights -> default_policy or similar
    weights = policy_state.get("weights", {})

    net = PolicyNet()
    state = net.state_dict()

    # RLlib fcnet weight keys follow the pattern:
    # _hidden_layers.0.weight, _hidden_layers.0.bias, etc.
    # Map them to our sequential net keys
    key_map = {
        "_hidden_layers.0._model.0.weight": "net.0.weight",
        "_hidden_layers.0._model.0.bias":   "net.0.bias",
        "_hidden_layers.1._model.0.weight": "net.2.weight",
        "_hidden_layers.1._model.0.bias":   "net.2.bias",
        "_hidden_layers.2._model.0.weight": "net.4.weight",
        "_hidden_layers.2._model.0.bias":   "net.4.bias",
        "_logits._model.0.weight":          "net.6.weight",
        "_logits._model.0.bias":            "net.6.bias",
    }

    loaded = 0
    for rllib_key, our_key in key_map.items():
        if rllib_key in weights and our_key in state:
            state[our_key] = torch.tensor(weights[rllib_key])
            loaded += 1

    if loaded == 0:
        # Print available keys to help debug
        print(f"[WARNING] No weights matched. Available keys: {list(weights.keys())[:10]}")
        raise RuntimeError(
            "Could not map any weights from checkpoint to PolicyNet. "
            "The weight key format may differ — check the printed keys above."
        )

    print(f"[PolicyNet] Loaded {loaded}/8 weight tensors for {policy_id}")
    net.load_state_dict(state)
    net.eval()
    return net


# ── Prediction ────────────────────────────────────────────────────────────────

class CustomPredictFunction(Callable):
    """
    Pure PyTorch inference — no RLlib, no Ray workers, no env registration.
    Loads policy weights directly from the checkpoint pickle files.
    """

    def __init__(self, env: gymnasium.Env):
        checkpoint_path = str(Path(CHECKPOINT_PATH).resolve())
        print(f"[CustomPredictFunction] Loading weights from {checkpoint_path}")
        self._policies = {
            "archer_0": _load_policy_weights(checkpoint_path, "archer_0"),
            "archer_1": _load_policy_weights(checkpoint_path, "archer_1"),
        }
        print("[CustomPredictFunction] Weights loaded successfully.")

    def __call__(self, observation: np.ndarray, agent: str, *args, **kwargs) -> int:
        net = self._policies.get(agent)
        if net is None:
            return 0  # fallback: no-op
        with torch.no_grad():
            obs_t  = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = net(obs_t)
            action = torch.argmax(logits, dim=-1).item()  # deterministic
        return action


# ── Zombie detector ───────────────────────────────────────────────────────────

class CustomZombieDetectorFunction(Callable):
    """YOLO-based zombie detector for evaluation."""

    def __init__(self, env: gymnasium.Env):
        self._use_yolo = get_yolo_model() is not None

    def __call__(self, observation: np.ndarray, *args, **kwargs) -> np.ndarray:
        if not self._use_yolo:
            return np.zeros((0, 4), dtype=np.float32)
        img     = observation.reshape(SCREEN_H, SCREEN_W, 3).astype(np.uint8)
        model   = get_yolo_model()
        results = model(img, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        xyxy  = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        order = np.argsort(-confs)
        xyxy  = xyxy[order]
        boxes = np.zeros((len(xyxy), 4), dtype=np.float32)
        boxes[:, 0] = xyxy[:, 0]
        boxes[:, 1] = xyxy[:, 1]
        boxes[:, 2] = xyxy[:, 2] - xyxy[:, 0]
        boxes[:, 3] = xyxy[:, 3] - xyxy[:, 1]
        return boxes