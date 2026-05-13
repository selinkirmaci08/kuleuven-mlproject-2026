import numpy as np
from PIL import Image
import supersuit as ss
import matplotlib.pyplot as plt
from utils import create_environment

START_FRAME = 50
END_FRAME = 55

env = create_environment(render_mode="rgb_array", distortion_level=5)
env.reset(seed=42)

obs_rgb_list = []
frame_ids = []

frame_count = 0

# get from frame 55 to see the zombies
while frame_count <= END_FRAME:
    obs, reward, termination, truncation, info = env.last()

    if obs is not None and START_FRAME <= frame_count <= END_FRAME:
        obs_rgb_list.append(obs.copy())
        frame_ids.append(frame_count)

    if termination or truncation:
        env.reset(seed=42)

    agent = env.agent_selection
    action = env.action_space(agent).sample()
    env.step(action)

    frame_count += 1

env.close()

# apply canal blue
obs_b_list = [img[:, :, 2] for img in obs_rgb_list]  # change to 0 for red, 1 for green

n = len(obs_rgb_list)
fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))

if n == 1:
    axes = np.array([axes])

for i in range(n):
    axes[i, 0].imshow(obs_rgb_list[i])
    axes[i, 0].set_title(f"Frame {frame_ids[i]} - RGB")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(obs_b_list[i], cmap="Blues")
    axes[i, 1].set_title(f"Frame {frame_ids[i]} - Canal B")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()
