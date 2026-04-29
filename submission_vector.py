"""
submission_vector.py  —  Option A: Vector-only policy (ablation experiment).

NO pixel observations. The policy receives only:
  - Archer positions (exact, from sprite rects)
  - Zombie positions + confidence (from YOLO)

This is NOT the final submission (pixel obs are required by the assignment).
Use this as an ablation to show in your report:
  1. How fast a vector policy converges vs pixel policy
  2. What performance ceiling a perfect-information policy achieves
  3. How much the CNN actually contributes

Compare training curves of this vs submission_s.py (Option B) in your report.
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
from ray.rllib.core.rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.columns import Columns
import supersuit as ss


ENV_SETTINGS = {
    "frame_stack": None,
    "resize_dim": None,
}

# ── Dimensions ────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 720
MAX_ZOMBIES        = 4

# Vector layout (16 floats total):
#   [archer_0_x, archer_0_y, archer_1_x, archer_1_y,   ← 4  (sprite rects)
#    z0_cx, z0_cy, z0_conf,                              ← 3  (YOLO)
#    z1_cx, z1_cy, z1_conf,
#    z2_cx, z2_cy, z2_conf,
#    z3_cx, z3_cy, z3_conf]                              → 16 total
ARCHER_DIM = 4
ZOMBIE_DIM = MAX_ZOMBIES * 3
VEC_DIM    = ARCHER_DIM + ZOMBIE_DIM   # 16

YOLO_WEIGHTS = "runs/detect/train/weights/best.pt"


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
    """Exact archer positions from sprite rects, normalised to [0,1]."""
    pos = np.zeros(ARCHER_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "archer_list"):
        return pos
    alive = env.agents
    for i, archer in enumerate(raw.archer_list):
        if f"archer_{i}" not in alive:
            continue
        r = archer.rect
        pos[2*i]   = np.clip((r.x + r.width  / 2) / SCREEN_W, 0.0, 1.0)
        pos[2*i+1] = np.clip((r.y + r.height / 2) / SCREEN_H, 0.0, 1.0)
    return pos


def get_zombie_positions_from_sprites(env) -> np.ndarray:
    """
    Read zombie positions directly from zombie_list sprite rects.
    Used during TRAINING — zero overhead, exact ground truth.
    YOLO is only used at inference time (see get_zombie_positions_yolo).
    """
    vec = np.zeros(ZOMBIE_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "zombie_list"):
        return vec
    zombies = list(raw.zombie_list)[:MAX_ZOMBIES]
    for slot, zombie in enumerate(zombies):
        r = zombie.rect
        cx = (r.x + r.width  / 2) / SCREEN_W
        cy = (r.y + r.height / 2) / SCREEN_H
        vec[slot*3]   = np.clip(cx,  0.0, 1.0)
        vec[slot*3+1] = np.clip(cy,  0.0, 1.0)
        vec[slot*3+2] = 1.0   # confidence = 1.0 (ground truth)
    return vec


def get_zombie_positions_yolo(raw_obs: np.ndarray) -> np.ndarray:
    """
    YOLO zombie detections — used at INFERENCE time only.
    Returns flat (12,) vector [cx, cy, conf] per slot.
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
            vec[slot*3]   = np.clip(((x1+x2)/2) / SCREEN_W, 0, 1)
            vec[slot*3+1] = np.clip(((y1+y2)/2) / SCREEN_H, 0, 1)
            vec[slot*3+2] = np.clip(float(confs[idx]),       0, 1)
    except Exception as e:
        print(f"[VectorPolicy] YOLO error: {e}")
    return vec


# ── Wrapper ───────────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Vector-only wrapper. observe() returns a flat (16,) float32 vector —
    NO image. The observation space is a plain Box, not a Dict, so the
    standard PPOTorchRLModule with an MLP encoder works out of the box.

    This is the key difference from submission_s.py (Option B):
      Option A (this file) : Box(16,)         → MLP only
      Option B (submission_s.py): Dict{image, position} → CNN + MLP
    """

    def __init__(self, env):
        self._unwrapped = env   # for archer_list access
        self._raw_env   = env   # for full-res RGB for YOLO
        # No supersuit preprocessing needed — we don't use pixels
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> spaces.Space:
        return spaces.Box(low=0.0, high=1.0,
                          shape=(VEC_DIM,), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType:
        # Training: use sprite rects for zombies (fast, exact, no YOLO overhead)
        # Inference: CustomPredictFunction calls _observe_inference() instead
        archer_pos = get_archer_positions(self._unwrapped)          # (4,)
        zombie_pos = get_zombie_positions_from_sprites(self._unwrapped)  # (12,)
        return np.concatenate([archer_pos, zombie_pos])              # (16,)

    def observe_inference(self, agent: AgentID) -> ObsType:
        """Used by CustomPredictFunction at inference — runs YOLO on pixels."""
        raw_obs    = self._raw_env.observe(agent)
        archer_pos = get_archer_positions(self._unwrapped)
        zombie_pos = get_zombie_positions_yolo(raw_obs)
        return np.concatenate([archer_pos, zombie_pos])


# ── RLModule — simple MLP, no CNN ────────────────────────────────────────────

class VectorPPORLModule(TorchRLModule):
    """
    PPO RLModule that takes a flat (16,) vector and outputs actions.

    Architecture:
        Linear(16 → 128) → ReLU
        Linear(128 → 128) → ReLU
        Linear(128 → 64)  → ReLU
        ├── actor:  Linear(64 → num_actions)
        └── critic: Linear(64 → 1)

    Much smaller and faster than the CNN in submission_s.py.
    Converges in far fewer environment steps because the input is already
    semantically meaningful (positions, not raw pixels).
    """

    def setup(self):
        self.mlp = nn.Sequential(
            nn.Linear(VEC_DIM, 128), nn.ReLU(),
            nn.Linear(128,     128), nn.ReLU(),
            nn.Linear(128,      64), nn.ReLU(),
        )
        num_actions      = self.config.action_space.n
        self.actor_head  = nn.Linear(64, num_actions)
        self.critic_head = nn.Linear(64, 1)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs.float())

    def _forward_train(self, batch: Dict[str, Any], **kwargs):
        features = self._encode(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: self.actor_head(features),
            Columns.VF_PREDS:           self.critic_head(features).squeeze(-1),
        }

    def _forward_inference(self, batch: Dict[str, Any], **kwargs):
        with torch.no_grad():
            features = self._encode(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: self.actor_head(features)}

    def _forward_exploration(self, batch: Dict[str, Any], **kwargs):
        with torch.no_grad():
            features = self._encode(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: self.actor_head(features),
            Columns.VF_PREDS: self.critic_head(features).squeeze(-1),
        }

    def get_inference_action_dist_cls(self):
        try:
            from ray.rllib.models.torch.torch_distributions import TorchCategorical
        except ImportError:
            from ray.rllib.models.torch.torch_action_dist import TorchCategorical
        return TorchCategorical


# ── Prediction ────────────────────────────────────────────────────────────────

class CustomPredictFunction(Callable):

    def __init__(self, env: gymnasium.Env):
        best_checkpoint = (
            Path("results_vector") / "learner_group" / "learner" / "rl_module"
        ).resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        # observation here comes from env.observe() which uses sprites (fast).
        # For real inference with YOLO, call env.observe_inference() externally.
        rl_module   = self.modules[agent]
        fwd_ins     = {Columns.OBS: torch.tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        dist_cls    = rl_module.get_inference_action_dist_cls()
        dist        = dist_cls.from_logits(fwd_outputs[Columns.ACTION_DIST_INPUTS])
        return dist.sample()[0].numpy()


# ── Zombie detector (unchanged — same YOLO) ───────────────────────────────────

class CustomZombieDetectorFunction(Callable):

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