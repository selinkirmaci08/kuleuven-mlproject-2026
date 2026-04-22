"""Template of your submission file for Task 3 (multi agent KAZ).
"""
from typing import Callable
import gymnasium
import numpy as np
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from gymnasium import spaces
from PIL import Image


class CustomWrapper(BaseWrapper):
    """
    Wrapper to use to add state pre-processing (feature engineering)
    """

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return spaces.Box(low=0.0, high=1.0, shape=(84, 84, 3), dtype=np.float32)

    def observe(self, agent: AgentID) -> ObsType | None:
        obs = super().observe(agent)
        img = Image.fromarray(obs)
        resizedImg = img.resize((84, 84))
        arrayOfResized = np.array(resizedImg)
        return (arrayOfResized / 255.0).astype(np.float32)


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

