import numpy as np
import matplotlib.pyplot as plt



# Increase tau so stag basin is reachable, lower alpha for smoother curves
starts = [
    (0.1, 0.1),    # hare basin - goes to (0,0)... or interior
    (0.15, 0.28),
    (0.3, 0.15),
    (0.5, 0.75),   # above separatrix
    (0.75, 0.5),
    (0.85, 0.85),  # stag basin
    (0.35, 0.38),  # near separatrix
    (0.4, 0.8),
]

colors = plt.cm.tab10.colors

fig, ax = plt.subplots(figsize=(7, 7))

# Quiver vector field
X, Y, U, V = replicator_vector_field(R1, R2, grid=20)
M = np.hypot(U, V)
M[M == 0] = 1
ax.quiver(X, Y, U/M, V/M, color='gray', alpha=0.4, scale=30, width=0.003)

for seed, (cx, cy) in enumerate(starts):
    traj = boltzmann_trajectory(
        R1, R2,
        T=50000,
        alpha=0.003,   # slower, smoother
        tau=0.003,       # higher tau → broader stag basin
        seed=seed,
        center=(cx, cy),
        noise_scale=0.0
    )
    x = traj[:, 0]
    y = traj[:, 1]

    ax.plot(x, y, linewidth=1.5, color=colors[seed % len(colors)], alpha=0.85)
    ax.scatter(x[0],  y[0],  color='green', s=120, zorder=5)           # start ●
    ax.scatter(x[-1], y[-1], color='red', marker='x', s=100,
               linewidths=2, zorder=5)                                  # end ×

ax.set_xlabel("P(Stag) Agent 1", fontsize=12)
ax.set_ylabel("P(Stag) – Agent 2", fontsize=12)
ax.set_title("StagHunt - boltzmann", fontsize=13)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.2)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='green',
           markersize=10, label='Start'),
    Line2D([0],[0], marker='x', color='red', markersize=10,
           linewidth=2, label='End'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()