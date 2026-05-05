"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing                 import Callable
import gymnasium
import numpy                as np
from pettingzoo.utils       import BaseWrapper
from pettingzoo.utils.env   import AgentID, ObsType
from gymnasium              import spaces
from PIL                    import Image
import supersuit            as ss

TARGET_SIZE = (84, 84)

ENV_SETTINGS = {
    #"visual_observation": True, # Forced to True by the project and utils.py
    "frame_stack": 3,  # Number of frames to stack (None for no stacking)
    "resize_dim": TARGET_SIZE,  # Tuple (width, height) to resize visual observations
}

class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """
    def __init__(self, env):
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=TARGET_SIZE[1], y_size=TARGET_SIZE[0])
        env = ss.dtype_v0(env, dtype=np.float32)
        env = ss.normalize_obs_v0(env, env_min=0.0, env_max=1.0)
        env = ss.frame_stack_v1(env, ENV_SETTINGS["frame_stack"])
        super().__init__(env)

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

