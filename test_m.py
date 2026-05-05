# testing state vs observation vs all that

from utils import create_environment
from submission_m import CustomWrapper
import numpy                as np

env = create_environment()
env = CustomWrapper(env)

env.reset()

for agent in env.agents:
    obs = env.observe(agent)
    print(agent, obs.shape, obs.dtype, obs.min(), obs.max())
    obs_space = env.observation_space(agent)
    print (obs_space.shape) # (84, 84, 9)

env.reset()

obs0 = env.observe("archer_0")
obs1 = env.observe("archer_1")

print("archer_0:", obs0.shape)
print("archer_1:", obs1.shape)
print("same image:", np.array_equal(obs0, obs1))