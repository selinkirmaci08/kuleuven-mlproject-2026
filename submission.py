"""
submission_vector.py  —  Vector policy with alignment + rotation direction + distance + dist_to_bottom.

Observation per archer (20 floats):
  [x, y, dir_x, dir_y,                                                ← 4 absolute
   z0_align, z0_cross, z0_dist, z0_dist_to_bottom,                   ← 4 per zombie (16 total)
   z1_align, z1_cross, z1_dist, z1_dist_to_bottom,
   z2_align, z2_cross, z2_dist, z2_dist_to_bottom,
   z3_align, z3_cross, z3_dist, z3_dist_to_bottom]

  align:          1.0 = zombie directly in front → shoot now
                  0.5 = zombie to the side
                  0.0 = zombie directly behind

  cross:          1.0 = zombie is to the LEFT  → rotate left to face it
                  0.5 = zombie is directly ahead/behind (no rotation needed)
                  0.0 = zombie is to the RIGHT → rotate right to face it

  dist:           0.0 = very close to archer, 1.0 = far from archer

  dist_to_bottom: 0.0 = zombie at top (safe), 1.0 = zombie at bottom (danger!)

  Zombies sorted by dist_to_bottom descending — slot 0 = most dangerous.

Total observation: 40 floats (20 per archer, own perspective first)
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
ZOMBIE_REL_DIM     = MAX_ZOMBIES * 4      # align, cross, dist, dist_to_bottom per zombie
PER_ARCHER_DIM     = ARCHER_ABS_DIM + ZOMBIE_REL_DIM   # 20
VEC_DIM            = PER_ARCHER_DIM * 2  # 40 total
NUM_ACTIONS        = 6
NUM_ARCHERS        = 2
SCREEN_DIAG        = (1.0**2 + 1.0**2) ** 0.5  # diagonal of normalised unit square

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
    Build 20-float observation for one archer.

    For each zombie:
      align          = dot(facing, zombie_dir)  normalised to [0,1]
      cross          = cross(facing, zombie_dir) normalised to [0,1]
      dist           = distance to archer normalised by screen diagonal
      dist_to_bottom = zombie y position (0.0=top/safe, 1.0=bottom/danger)
    """
    vec = np.zeros(PER_ARCHER_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)

    if not hasattr(raw, "archer_list"):
        return vec

    archer_list = list(raw.archer_list)

    if archer_idx >= len(archer_list):
        return vec

    alive = env.agents
    if f"archer_{archer_idx}" not in alive:
        return vec

    archer = archer_list[archer_idx]
    r  = archer.rect
    ax = (r.x + r.width  / 2) / SCREEN_W
    ay = (r.y + r.height / 2) / SCREEN_H
    dx = float(archer.direction.x)
    dy = float(archer.direction.y)

    vec[0] = np.clip(ax, 0.0, 1.0)
    vec[1] = np.clip(ay, 0.0, 1.0)
    vec[2] = (dx + 1.0) / 2.0
    vec[3] = (dy + 1.0) / 2.0

    for slot, (zx, zy) in enumerate(zombie_positions[:MAX_ZOMBIES]):
        rel_x = zx - ax
        rel_y = zy - ay

        dist = (rel_x**2 + rel_y**2) ** 0.5

        if dist > 1e-6:
            norm_x = rel_x / dist
            norm_y = rel_y / dist
        else:
            norm_x, norm_y = 0.0, 0.0

        dot   = norm_x * dx + norm_y * dy
        align = np.clip((dot + 1.0) / 2.0, 0.0, 1.0)

        cross      = norm_x * dy - norm_y * dx
        cross_norm = np.clip((cross + 1.0) / 2.0, 0.0, 1.0)

        dist_norm      = np.clip(dist / SCREEN_DIAG, 0.0, 1.0)
        dist_to_bottom = np.clip(zy, 0.0, 1.0)  # 0=top (safe), 1=bottom (danger)

        base = ARCHER_ABS_DIM + slot * 4
        vec[base]   = align
        vec[base+1] = cross_norm
        vec[base+2] = dist_norm
        vec[base+3] = dist_to_bottom

    return vec


def get_zombie_positions_from_sprites(env) -> list:
    positions = []
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "zombie_list"):
        return positions
    for zombie in list(raw.zombie_list)[:MAX_ZOMBIES]:
        r = zombie.rect
        cx = np.clip((r.x + r.width  / 2) / SCREEN_W, 0.0, 1.0)
        cy = np.clip((r.y + r.height / 2) / SCREEN_H, 0.0, 1.0)
        positions.append((cx, cy))
    positions.sort(key=lambda p: p[1], reverse=True)
    # randomly drop all zombies 5% of steps to simulate YOLO failures
    if np.random.random() < 0.05:
        return []
    return positions


def get_zombie_positions_yolo(raw_obs: np.ndarray) -> list:
    """YOLO zombie positions — INFERENCE only. Sorted by dist_to_bottom descending."""
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
    positions.sort(key=lambda p: p[1], reverse=True)  # slot 0 = closest to bottom
    return positions


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Returns 40-float observation — each archer sees from its own perspective first.

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

        if agent == "archer_0":
            return np.concatenate([obs_archer0, obs_archer1])
        else:
            return np.concatenate([obs_archer1, obs_archer0])


# ── Reward wrapper (training only) ───────────────────────────────────────────

class RewardWrapper(BaseWrapper):
    """
    Shaped reward on top of KAZ defaults.
    Wrap AFTER CustomWrapper — only used during training, never during eval.
    """

    W_KILL    = 1.0   # base kill reward passthrough
    W_URGENCY = 0.02  # bonus for zombie close to bottom AND close to archer

    def __init__(self, env):
        super().__init__(env)

    def _parse_obs(self, obs):
        """Read slot 0 (most dangerous zombie) from own observation."""
        dist_to_archer = obs[6]  # z0_dist
        dist_to_bottom = obs[7]  # z0_dist_to_bottom
        return dist_to_archer, dist_to_bottom

    def reward(self, agent):
        base_reward = self.env.reward(agent)

        if base_reward > 0:
            print(f"[KILL] {agent} killed a zombie! base={base_reward:.3f}")

        obs = self.observe(agent)
        dist_to_archer, dist_to_bottom = self._parse_obs(obs)

        shaped = base_reward * self.W_KILL

        # urgency bonus — only meaningful when zombie is both near bottom AND near archer
        urgency = dist_to_bottom * (1.0 - dist_to_archer)
        shaped += self.W_URGENCY * urgency

        return shaped


# ── Pure PyTorch policy network ───────────────────────────────────────────────

class PolicyNet(nn.Module):
    """MLP matching fcnet_hiddens=[128,128,64]. Input: 40. Output: 6."""
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
    """Load weights from a plain .pt file saved by extract_weights.py."""
    import os

    pt_path = os.path.join(checkpoint_path, f"{policy_id}_weights.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(
            f"Weight file not found: {pt_path}\n"
            f"Run extract_weights.py locally first to generate it."
        )

    weights = torch.load(pt_path, map_location="cpu", weights_only=False)
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
        print(f"[WARNING] No weights matched. Available keys: {list(weights.keys())[:8]}")
        raise RuntimeError(f"Could not map any weights for {policy_id}.")

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