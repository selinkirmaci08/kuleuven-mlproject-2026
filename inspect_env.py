from utils import create_environment

env = create_environment(distortion_level=0)
env.reset(seed=42)

# Grab one observation from the first agent
agent = env.agents[0]
obs = env.observe(agent)

print("Shape:", obs.shape)
print("Dtype:", obs.dtype)
print("Min value:", obs.min())
print("Max value:", obs.max())

env.close()