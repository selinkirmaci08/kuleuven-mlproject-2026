import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# VECTOR FIELD
# ============================================================

def add_vector_field(
        ax,
        X,
        Y,
        U,
        V,
        normalize=True,
        color='gray',
        alpha=0.4,
        scale=30,
        width=0.003
):
    if normalize:
        M = np.hypot(U, V)
        M[M == 0] = 1

        Uplot = U / M
        Vplot = V / M

    else:
        Uplot = U
        Vplot = V

    ax.quiver(
        X,
        Y,
        Uplot,
        Vplot,
        color=color,
        alpha=alpha,
        scale=scale,
        width=width
    )


def add_trajectory(
        ax,
        traj,
        color=None,
        linewidth=1.5,
        alpha=0.85,
        start_marker=True,
        end_marker=True,
        label=None,
        stride=50
):
    traj = traj[::stride]

    x = traj[:, 0]
    y = traj[:, 1]

    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label
    )

    if start_marker:
        ax.scatter(
            x[0],
            y[0],
            color='green',
            s=20,
            zorder=5
        )

    if end_marker:
        ax.scatter(
            x[-1],
            y[-1],
            color='blue',
            s=130,
            linewidths=2,
            zorder=5
        )


# ============================================================
# AXIS STYLING
# ============================================================

def finalize_phase_plot(
        ax,
        title="",
        xlabel="P(Action 0) Agent 1",
        ylabel="P(Action 0) Agent 2"
):
    ax.set_title(title, fontsize=13)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.grid(alpha=0.2)


def barycentric_to_cartesian(coords):
    p0, p1, p2 = coords[:, 0], coords[:, 1], coords[:, 2]
    V0 = np.array([0.0, 0.0])  # Rock
    V1 = np.array([1.0, 0.0])  # Paper
    V2 = np.array([0.5, np.sqrt(3) / 2])  # Scissors

    x = p0 * V0[0] + p1 * V1[0] + p2 * V2[0]
    y = p0 * V0[1] + p1 * V1[1] + p2 * V2[1]
    return x, y


def add_trajectory_simplex(ax, traj, agent=0, stride=50, color='blue',
                           linewidth=1.5, alpha=0.85, **kwargs):
    # Stride the trajectory
    traj_strided = traj[::stride]

    # Handle both (T, n_agents, 3) and (T, 3) shaped trajectories
    if traj_strided.ndim == 3:
        coords = traj_strided[:, agent, :]  # (T_strided, 3)
    elif traj_strided.ndim == 2:
        coords = traj_strided  # already (T_strided, 3)
    else:
        raise ValueError(f"Unexpected trajectory shape: {traj_strided.shape}")

    xs, ys = barycentric_to_cartesian(coords)

    ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha, **kwargs)
    ax.plot(xs[0], ys[0], 'o', color='green', markersize=6, zorder=5)  # start

    # Use the actual last point of the full trajectory for the end marker
    if traj.ndim == 3:
        last = traj[-1, agent, :]
    else:
        last = traj[-1, :]
    lx, ly = barycentric_to_cartesian(last[None, :])
    ax.plot(lx[0], ly[0], 'o', color='blue', markersize=8, zorder=10)  # end


def finalize_simplex_plot(ax, title="", labels=("Rock", "Paper", "Scissors")):
    """
    Dibuja el triángulo del simplex con etiquetas en los vértices.
    """
    V = np.array([
        [0.0, 0.0],  # Rock
        [1.0, 0.0],  # Paper
        [0.5, np.sqrt(3) / 2],  # Scissors
        [0.0, 0.0]  # cierra el triángulo
    ])

    ax.plot(V[:, 0], V[:, 1], 'k-', linewidth=1.5)

    offset = 0.05
    ax.text(V[0, 0] - offset, V[0, 1] - offset, labels[0], fontsize=11, ha='center')
    ax.text(V[1, 0] + offset, V[1, 1] - offset, labels[1], fontsize=11, ha='center')
    ax.text(V[2, 0], V[2, 1] + offset, labels[2], fontsize=11, ha='center')

    ax.set_title(title, fontsize=13, pad=20)
    ax.set_aspect('equal')
    ax.axis('off')

    """ nash = np.array([nash_r, nash_p, nash_s])
    nx, ny = barycentric_to_cartesian(nash[None, :])
    ax.scatter(nx, ny, color='gray', s=150, marker='*', zorder=10, label='Nash NE') """