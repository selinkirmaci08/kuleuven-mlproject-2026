from pettingzoo.utils.env import ActionType
from ray.rllib.examples.envs.classes.cliff_walking_wall_env import ACTION_LEFT

from utils import create_environment
import supersuit as ss
import numpy as np

env = create_environment(distortion_level=0)
env.reset(seed=36)
#env = ss.color_reduction_v0(env, mode="B")
#env = ss.resize_v1(env, x_size=84, y_size=84)
#env = ss.dtype_v0(env, np.float32)        # convert to float32 FIRST
#env = ss.normalize_obs_v0(env, env_min=0.0, env_max=1.0)  # then normalize
#env = ss.frame_stack_v1(env, 3)           # stack last

# Grab one observation from the first agent
print(len(env.agents))
agent = env.agents[0]
obs = env.observe(agent)

agent2 = env.agents[1]
obs2 = env.observe(agent2)

print(dir(env.unwrapped))
print(env.agent_list)

print(np.array_equal(obs, obs2))


os = env.observation_space(agent)

print("Shape:", obs.shape)
print("Dtype:", obs.dtype)
print("Min value:", obs.min())
print("Max value:", obs.max())

env.close()