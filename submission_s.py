"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from pathlib import Path
from typing import Callable
import gymnasium
import numpy as np
import torch
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from gymnasium import spaces
from PIL import Image
from ray.rllib.core.rl_module import MultiRLModule
import supersuit as ss


ENV_SETTINGS = {
    "visual_observation": True,  # Set to True if using visual observations
    # the following parameters are only relevant if visual_observation is True
    "frame_stack": 4,  # Number of frames to stack (None for no stacking)
    "resize_dim": None,  # Tuple (width, height) to resize visual observations
}

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def __init__(self, env):
        super().__init__(env)
        env = ss.frame_skip_v0(env, 4)
        self.buffer: [AgentID, list] = []

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(84, 84, 4), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        img = Image.fromarray(obs).convert("L").resize((84, 84))  # grayscale
        arr = np.array(img, dtype=np.uint8)[..., None]  # (84,84,1)

        buf = self.buffers.setdefault(agent, [])
        buf.append(arr)
        if len(buf) > 4:
            buf.pop(0)
        while len(buf) < 4:
            buf.append(arr)

        stacked = np.concatenate(buf, axis=2)  # (84, 84, 4)
        return (stacked / 255.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.buffers = {}
        return super().reset(seed=seed, options=options)


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env: gymnasium.Env):
        best_checkpoint = (Path("results") / "learner_group" / "learner" / "rl_module").resolve()
        self.modules = MultiRLModule.from_checkpoint(best_checkpoint)

    def __call__(self, observation, agent, *args, **kwargs):
        rl_module = self.modules[agent]
        fwd_ins = {"obs": torch.Tensor(observation).unsqueeze(0)}
        fwd_outputs = rl_module.forward_inference(fwd_ins)
        action_dist_class = rl_module.get_inference_action_dist_cls()
        action_dist = action_dist_class.from_logits(
            fwd_outputs["action_dist_inputs"]
        )
        action = action_dist.sample()[0].numpy()
        return action


class CustomZombieDetectorFunction(Callable):
    """Function to use to load the trained model and predict where
    the zombies are.
    """

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, *args, **kwargs):
        """Returns a matrix of shape (nb_zombies, nb_attributes), where
        the attributes are defining a rectangle with (x,y,width,heigh) and
        indicate where the zombies are. The zombies are ordered from most
        likely to least likely positions. The evaluation uses the first k
        items if there are k zombies on the screen.
        """

        import matplotlib.colors as mcolors
        from scipy import ndimage

        # Step 1: convert to HSV
        img_normalized = observation / 255.0
        img_normalized = img_normalized.reshape(720, 1280, 3)
        hsv = mcolors.rgb_to_hsv(img_normalized)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Step 2: green mask
        green_mask = (hue > 0.25) & (hue < 0.42) & (sat > 0.25) & (val > 0.10)

        # Step 3: find blobs
        labeled, num_features = ndimage.label(green_mask)

        # Step 4: bounding boxes
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

