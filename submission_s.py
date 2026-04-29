"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from pathlib import Path
from typing import Callable, Dict, Any, Optional
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
    "frame_stack": 4,
    "resize_dim": None,
}

# ── Dimensions ───────────────────────────────────────────────────────────────
IMG_H, IMG_W, N_FRAMES = 84, 84, 4
SCREEN_W, SCREEN_H     = 1280, 720
MAX_ZOMBIES            = 4    # KAZ default

# Position vector layout:
#   [archer_0_x, archer_0_y,          ← 2 floats  (HSV detection)
#    archer_1_x, archer_1_y,          ← 2 floats
#    z0_cx, z0_cy, z0_conf,           ← 3 floats per zombie  (YOLO)
#    z1_cx, z1_cy, z1_conf,
#    z2_cx, z2_cy, z2_conf,
#    z3_cx, z3_cy, z3_conf]
ARCHER_DIM  = 4                        # 2 archers × (x, y)
ZOMBIE_DIM  = MAX_ZOMBIES * 3          # 4 zombies × (cx, cy, conf)
POS_DIM     = ARCHER_DIM + ZOMBIE_DIM  # = 16

# Path to trained YOLO weights — update if you retrain
YOLO_WEIGHTS = "runs/detect/train/weights/best.pt"


# ── YOLO zombie detector (singleton — loaded once, reused every step) ────────

_yolo_model = None

def get_yolo_model():
    """Lazy-load YOLO model once and cache it."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(YOLO_WEIGHTS)
            print(f"[ZombieDetector] YOLO loaded from {YOLO_WEIGHTS}")
        except Exception as e:
            print(f"[ZombieDetector] Could not load YOLO ({e}), zombie positions will be zeros")
    return _yolo_model


def detect_zombies_yolo(raw_obs: np.ndarray) -> np.ndarray:
    """
    Run YOLO on raw 720×1280×3 uint8 image.
    Returns flat vector of shape (ZOMBIE_DIM,) = (12,):
        [z0_cx, z0_cy, z0_conf,  z1_cx, z1_cy, z1_conf, ...]
    All positions normalised to [0,1]. Slots for missing zombies are zeros.
    Zombies are sorted by confidence (highest first) so the most certain
    detections always occupy the first slots.
    """
    vec = np.zeros(ZOMBIE_DIM, dtype=np.float32)
    model = get_yolo_model()
    if model is None:
        return vec

    try:
        results = model(raw_obs, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            return vec

        xyxy  = results.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs = results.boxes.conf.cpu().numpy()   # (N,)

        # Sort by confidence descending, keep at most MAX_ZOMBIES
        order = np.argsort(-confs)[:MAX_ZOMBIES]

        for slot, idx in enumerate(order):
            x1, y1, x2, y2 = xyxy[idx]
            cx   = ((x1 + x2) / 2) / SCREEN_W
            cy   = ((y1 + y2) / 2) / SCREEN_H
            conf = float(confs[idx])
            vec[slot * 3]     = np.clip(cx,   0.0, 1.0)
            vec[slot * 3 + 1] = np.clip(cy,   0.0, 1.0)
            vec[slot * 3 + 2] = np.clip(conf, 0.0, 1.0)
    except Exception as e:
        print(f"[ZombieDetector] inference error: {e}")

    return vec


def unwrap_to_raw(env):
    """Follow wrapper chain (aec_env first, then env) to find raw KAZ env."""
    cur = env
    seen = set()
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
    Read archer positions directly from archer_list sprite rects in the raw env.
    Allowed by the assignment: uses env.agents and env.agent_list only for
    knowing which agents exist — actual positions come from the sprite rects.

    Returns flat vector of shape (ARCHER_DIM,) = (4,):
        [archer_0_x, archer_0_y, archer_1_x, archer_1_y]  normalised to [0,1].
    Dead archers get position (0, 0).
    """
    pos = np.zeros(ARCHER_DIM, dtype=np.float32)
    raw = unwrap_to_raw(env)
    if not hasattr(raw, "archer_list"):
        return pos

    # archer_list is ordered archer_0, archer_1 by KAZ convention
    alive = env.agents  # allowed call — just for liveness check
    for i, archer in enumerate(raw.archer_list):
        agent_id = f"archer_{i}"
        if agent_id not in alive:
            continue  # dead — leave as zeros
        r = archer.rect
        cx = r.x + r.width  / 2.0
        cy = r.y + r.height / 2.0
        pos[2*i]   = np.clip(cx / SCREEN_W, 0.0, 1.0)
        pos[2*i+1] = np.clip(cy / SCREEN_H, 0.0, 1.0)

    return pos


# ── Wrapper ──────────────────────────────────────────────────────────────────

class CustomWrapper(BaseWrapper):
    """
    Preprocessing wrapper that returns Dict observations:
        {
          "image":    (84, 84, 4) float32  — grayscale 4-frame stack
          "position": (16,)       float32  — [archer×4] + [zombie×12]
        }

    Archer positions: read directly from archer_list sprite rects (exact, allowed).
    Zombie positions: YOLO on full-res frame (cx, cy, confidence per zombie).
    env.agents used only for liveness check (explicitly allowed by assignment).
    """

    def __init__(self, env):
        self._raw_env = env
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=IMG_W, y_size=IMG_H)
        env = ss.dtype_v0(env, np.float32)
        env = ss.normalize_obs_v0(env, env_min=0.0, env_max=1.0)
        env = ss.frame_stack_v1(env, N_FRAMES)
        super().__init__(env)

    def observation_space(self, agent: AgentID) -> spaces.Space:
        img_space = self.env.observation_space(agent)
        return spaces.Dict({
            "image":    img_space,
            "position": spaces.Box(low=0.0, high=1.0,
                                   shape=(POS_DIM,), dtype=np.float32),
        })

    def observe(self, agent: AgentID) -> ObsType:
        img     = self.env.observe(agent)            # (84, 84, 4) preprocessed
        raw_obs = self._raw_env.observe(agent)       # (720, 1280, 3) uint8 for YOLO

        archer_pos = get_archer_positions(self._unwrapped)  # (4,)  exact sprite coords
        zombie_pos = detect_zombies_yolo(raw_obs)            # (12,) YOLO detections
        position   = np.concatenate([archer_pos, zombie_pos])  # (16,)

        return {"image": img, "position": position}


# ── Custom RLModule ───────────────────────────────────────────────────────────

class CustomPPORLModule(TorchRLModule):
    """
    PPO-compatible RLModule with CNN + MLP fusion for Dict observations.

    CNN branch  : processes 4-frame grayscale image
    MLP branch  : processes 16-float position vector
                  (4 archer + 12 zombie from YOLO)
    Fusion      : concat → Linear(512) → ReLU → actor / critic heads
    """

    def setup(self):
        # ── CNN ──────────────────────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(N_FRAMES, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32,       64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64,       64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, N_FRAMES, IMG_H, IMG_W)).shape[1]

        # ── MLP for position vector (now 16-d) ───────────────────────────────
        self.pos_mlp = nn.Sequential(
            nn.Linear(POS_DIM, 64), nn.ReLU(),   # wider: more info to process
            nn.Linear(64,      64), nn.ReLU(),
        )

        # ── Fusion + heads ────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 512), nn.ReLU(),
        )
        num_actions      = self.config.action_space.n
        self.actor_head  = nn.Linear(512, num_actions)
        self.critic_head = nn.Linear(512, 1)

    def _encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        img      = obs["image"].permute(0, 3, 1, 2).float()  # HWC → CHW
        cnn_feat = self.cnn(img)
        pos_feat = self.pos_mlp(obs["position"].float())
        return self.fusion(torch.cat([cnn_feat, pos_feat], dim=-1))

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
        # Must return VF_PREDS during rollout so PPO can compute advantages
        with torch.no_grad():
            features = self._encode(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: self.actor_head(features),
            Columns.VF_PREDS:           self.critic_head(features).squeeze(-1),
        }

    def compute_values(self, batch: Dict[str, Any], **kwargs):
        """Called by RLlib to compute value estimates for GAE/advantage computation."""
        features = self._encode(batch[Columns.OBS])
        return self.critic_head(features).squeeze(-1)

    def get_inference_action_dist_cls(self):
        try:
            from ray.rllib.models.torch.torch_distributions import TorchCategorical
        except ImportError:
            from ray.rllib.models.torch.torch_action_dist import TorchCategorical
        return TorchCategorical


# ── Prediction helper ─────────────────────────────────────────────────────────

class CustomPredictFunction(Callable):
    """Load trained RLlib checkpoint and predict actions."""

    def __init__(self, env: gymnasium.Env):
        best_checkpoint = (
            Path("results") / "learner_group" / "learner" / "rl_module"
        ).resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {
            Columns.OBS: {
                "image":    torch.tensor(observation["image"]).unsqueeze(0),
                "position": torch.tensor(observation["position"]).unsqueeze(0),
            }
        }
        fwd_outputs     = rl_module.forward_inference(fwd_ins)
        action_dist_cls = rl_module.get_inference_action_dist_cls()
        action_dist     = action_dist_cls.from_logits(
            fwd_outputs[Columns.ACTION_DIST_INPUTS]
        )
        return action_dist.sample()[0].numpy()


# ── Zombie detector (for evaluation.py --zombies) ────────────────────────────

class CustomZombieDetectorFunction(Callable):
    """
    YOLO-based zombie detector.
    Falls back to HSV thresholding if YOLO weights are not found.

    Returns (N, 4) array: [x_min, y_min, width, height] in pixels,
    sorted most-confident first.
    """

    def __init__(self, env: gymnasium.Env):
        self._use_yolo = get_yolo_model() is not None

    def __call__(self, observation: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self._use_yolo:
            return self._yolo_detect(observation)
        return self._hsv_detect(observation)

    def _yolo_detect(self, observation: np.ndarray) -> np.ndarray:
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

    def _hsv_detect(self, observation: np.ndarray) -> np.ndarray:
        """HSV fallback — same as original implementation."""
        import matplotlib.colors as mcolors
        from scipy import ndimage

        img = (observation / 255.0).reshape(SCREEN_H, SCREEN_W, 3)
        hsv = mcolors.rgb_to_hsv(img)
        hue, sat, val = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        green_mask = (hue > 0.25) & (hue < 0.42) & (sat > 0.25) & (val > 0.10)
        labeled, num_features = ndimage.label(green_mask)

        boxes = []
        for label_id in range(1, num_features + 1):
            component = labeled == label_id
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            w = int(x_max - x_min + 1)
            h = int(y_max - y_min + 1)
            if 10 <= w <= 80 and 10 <= h <= 80:
                boxes.append([x_min, y_min, w, h])

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(boxes, dtype=np.float32)