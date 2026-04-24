"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing import Callable
import gymnasium
import numpy as np
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from gymnasium import spaces
from PIL import Image

TARGET_SIZE = (84, 84) # TODO: if no GPU better 64

ENV_SETTINGS = {
    "visual_observation": True, # Forced to True by the project and utils.py
    "frame_stack": 3,  # Number of frames to stack (None for no stacking)
    "resize_dim": TARGET_SIZE,  # Tuple (width, height) to resize visual observations
}

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        h, w = TARGET_SIZE
        c = 3 #RGB

        return spaces.Box(low=0.0, high=1.0, shape=(h * w * c,), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        img = Image.fromarray(obs)
        resizedImg = img.resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
        arrayOfResized = np.array(resizedImg)
        return (arrayOfResized / 255.0).astype(np.float32) # normalize and flatten


class CustomPredictFunction(Callable):
    """Function to use to load the trained model and predict the action"""

    def __init__(self, env: gymnasium.Env):
        pass

    def __call__(self, observation, agent, *args, **kwargs):
        pass


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
        pass

