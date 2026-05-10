"""
submission_vector.py  —  Vector policy with alignment + rotation direction.

Observation per archer (12 floats):
  [x, y, dir_x, dir_y,                          ← 4 absolute
   z0_align, z0_cross, z1_align, z1_cross,       ← 2 per zombie (8 total)
   z2_align, z2_cross, z3_align, z3_cross]

  align: 1.0 = zombie directly in front → shoot now
         0.5 = zombie to the side
         0.0 = zombie directly behind

  cross: 1.0 = zombie is to the LEFT  → rotate left to face it
         0.5 = zombie is directly ahead/behind (no rotation needed)
         0.0 = zombie is to the RIGHT → rotate right to face it

Total observation: 24 floats (12 per archer, own perspective first)
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

import os

ENV_SETTINGS = {
    "frame_stack": None,
    "resize_dim": None,
}

# ── Dimensions ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 720
MAX_ZOMBIES        = 4
ARCHER_ABS_DIM     = 4                    # x, y, dir_x, dir_y
ZOMBIE_REL_DIM     = MAX_ZOMBIES * 2      # align, cross per zombie
PER_ARCHER_DIM     = ARCHER_ABS_DIM + ZOMBIE_REL_DIM   # 12
VEC_DIM            = PER_ARCHER_DIM * 2  # 24 total
NUM_ACTIONS        = 6
NUM_ARCHERS        = 2

#YOLO_WEIGHTS    = "runs/detect/train/weights/best.pt"
#CHECKPOINT_PATH = "results_vector"

_DIR            = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS    = os.path.join(_DIR, "runs", "detect", "train", "weights", "best.pt")
CHECKPOINT_PATH = os.path.join(_DIR, "results_vector")

# ── YOLO singleton ────────────────────────────────────────────────────────────

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
    """Follow wrapper chain to find the raw KAZ env."""
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


def get_observation_for_archer(env, archer_idx: int,
                                zombie_positions: list) -> np.ndarray:
    """
    Build 12-float observation for one archer.

    For each zombie:
      align = dot(facing, zombie_dir)  normalised to [0,1]
              1.0 = zombie directly in front → shoot
      cross = cross(facing, zombie_dir) normalised to [0,1]
              1.0 = rotate LEFT to face zombie
              0.0 = rotate RIGHT to face zombie
              0.5 = zombie is ahead or behind (no rotation needed)

    Together align+cross give the policy both WHEN to shoot and
    WHICH WAY to rotate to face the zombie.
    """
    vec = np.zeros(PER_ARCHER_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)

    if not hasattr(raw, "archer_list"):
        return vec

    archer_list = list(raw.archer_list)

    # Archer might have died — return zeros
    if archer_idx >= len(archer_list):
        return vec

    alive = env.agents
    if f"archer_{archer_idx}" not in alive:
        return vec

    archer = archer_list[archer_idx]
    r  = archer.rect
    ax = (r.x + r.width  / 2) / SCREEN_W
    ay = (r.y + r.height / 2) / SCREEN_H
    dx = float(archer.direction.x)   # unit vector, already normalised
    dy = float(archer.direction.y)

    # Absolute archer position + direction
    vec[0] = np.clip(ax, 0.0, 1.0)
    vec[1] = np.clip(ay, 0.0, 1.0)
    vec[2] = (dx + 1.0) / 2.0   # normalise [-1,1] → [0,1]
    vec[3] = (dy + 1.0) / 2.0

    # Relative zombie signals
    for slot, (zx, zy) in enumerate(zombie_positions[:MAX_ZOMBIES]):
        rel_x = zx - ax
        rel_y = zy - ay

        # Distance — normalise by screen diagonal
        dist = (rel_x**2 + rel_y**2) ** 0.5

        # Normalise relative direction to unit vector
        if dist > 1e-6:
            norm_x = rel_x / dist
            norm_y = rel_y / dist
        else:
            norm_x, norm_y = 0.0, 0.0

        # Dot product — alignment (how directly in front is the zombie)
        # dx,dy is already a unit vector so dot is in [-1, 1]
        dot   = norm_x * dx + norm_y * dy
        align = np.clip((dot + 1.0) / 2.0, 0.0, 1.0)

        # Cross product — rotation direction needed
        # cross > 0: zombie is to the LEFT  → rotate left (action 1)
        # cross < 0: zombie is to the RIGHT → rotate right (action 2)
        # cross = 0: zombie is ahead or behind
        cross      = norm_x * dy - norm_y * dx
        cross_norm = np.clip((cross + 1.0) / 2.0, 0.0, 1.0)

        vec[ARCHER_ABS_DIM + slot*2]   = align
        vec[ARCHER_ABS_DIM + slot*2+1] = cross_norm

    return vec


def get_zombie_positions_from_sprites(env) -> list:
    """Sprite rect zombie positions — TRAINING only."""
    positions = []
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "zombie_list"):
        return positions
    for zombie in list(raw.zombie_list)[:MAX_ZOMBIES]:
        r = zombie.rect
        cx = np.clip((r.x + r.width  / 2) / SCREEN_W, 0.0, 1.0)
        cy = np.clip((r.y + r.height / 2) / SCREEN_H, 0.0, 1.0)
        positions.append((cx, cy))
    return positions


def get_zombie_positions_yolo(raw_obs: np.ndarray) -> list:
    """YOLO zombie positions — INFERENCE only."""
    positions = []
    model = get_yolo_model()
    if model is None:
        return positions
    try:
        results = model(raw_obs, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return positions
        xyxy  = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        for idx in np.argsort(-confs)[:MAX_ZOMBIES]:
            x1, y1, x2, y2 = xyxy[idx]
            cx = np.clip(((x1+x2)/2) / SCREEN_W, 0.0, 1.0)
            cy = np.clip(((y1+y2)/2) / SCREEN_H, 0.0, 1.0)
            positions.append((cx, cy))
    except Exception as e:
        print(f"[VectorPolicy] YOLO error: {e}")
    return positions


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Returns 24-float observation — each archer sees from its own perspective first.

    inference_mode=False (training):  sprite rects, fast and exact
    inference_mode=True  (evaluation): YOLO on pixels, real pipeline
    """

    def __init__(self, env, inference_mode: bool = True):
        self._unwrapped         = env
        self._raw_env           = env
        self._inference_mode    = inference_mode
        self._cached_zombie_pos = []
        self._cache_step        = -1
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(VEC_DIM,), dtype=np.float32)

    def _get_zombie_positions(self, agent: AgentID) -> list:
        if self._inference_mode:
            raw = unwrap_to_raw(self._unwrapped)
            current_step = getattr(raw, "frames", self._cache_step + 1)
            if current_step != self._cache_step:
                raw_obs = self._raw_env.observe(agent)
                self._cached_zombie_pos = get_zombie_positions_yolo(raw_obs)
                self._cache_step = current_step
            return self._cached_zombie_pos
        else:
            return get_zombie_positions_from_sprites(self._unwrapped)

    def observe(self, agent: AgentID) -> ObsType:
        zombie_pos  = self._get_zombie_positions(agent)
        obs_archer0 = get_observation_for_archer(self._unwrapped, 0, zombie_pos)
        obs_archer1 = get_observation_for_archer(self._unwrapped, 1, zombie_pos)

        # Each archer sees its own perspective first
        if agent == "archer_0":
            return np.concatenate([obs_archer0, obs_archer1])
        else:
            return np.concatenate([obs_archer1, obs_archer0])


# ── Pure PyTorch policy network ───────────────────────────────────────────────

class PolicyNet(nn.Module):
    """MLP matching fcnet_hiddens=[128,128,64]. Input: 24. Output: 6."""
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
    """Load weights from RLlib old-API checkpoint into PolicyNet."""
    import pickle, os

    policy_state_path = os.path.join(
        checkpoint_path, "policies", policy_id, "policy_state.pkl"
    )
    if not os.path.exists(policy_state_path):
        raise FileNotFoundError(f"Policy state not found at {policy_state_path}")

    with open(policy_state_path, "rb") as f:
        policy_state = pickle.load(f)

    weights = policy_state.get("weights", {})
    net     = PolicyNet()
    state   = net.state_dict()

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
        print(f"[WARNING] No weights matched. Keys: {list(weights.keys())[:5]}")
        raise RuntimeError("Could not map weights from checkpoint to PolicyNet.")

    print(f"[PolicyNet] Loaded {loaded}/8 weight tensors for {policy_id}")
    net.load_state_dict(state)
    net.eval()
    return net


# ── Prediction ────────────────────────────────────────────────────────────────

class CustomPredictFunction(Callable):
    """Pure PyTorch inference — no RLlib, no Ray."""

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
            return 0
        with torch.no_grad():
            obs_t  = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits = net(obs_t)
            action = torch.argmax(logits, dim=-1).item()
        return action


# ── Zombie detector ───────────────────────────────────────────────────────────

class CustomZombieDetectorFunction(Callable):
    """YOLO-based zombie detector for evaluation."""

    def __init__(self, env: gymnasium.Env):
        self._use_yolo = None  # lazy load

    def __call__(self, observation: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self._use_yolo is None:
            self._use_yolo = get_yolo_model() is not None
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