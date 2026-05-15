#experimetns.py

from plotting import add_vector_field, add_trajectory, finalize_phase_plot, add_trajectory_simplex, finalize_simplex_plot, barycentric_to_cartesian
from learners import *
from dynamics import replicator_vector_field_boltzmann, replicator_vector_field_lenient_boltzmann, replicator_vector_field_lenient_boltzmann_3, replicator_vector_field_boltzmann_3, add_vector_field_simplex

from games import GAMES
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

mpl.rcParams['agg.path.chunksize'] = 10000

def experiment_staghunt_epsilongreedy():
    game = GAMES["stag_hunt"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.1, 0.1),
        (0.2, 0.3),
        (0.3, 0.15),
        (0.7, 0.8),
        (0.85, 0.65),
        (0.9, 0.9),
        (0.1, 0.9),
        (0.25, 0.75),
        (0.9, 0.1),
        (0.75, 0.25),
    ]
    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = epsilon_greedy_trajectory(
            R1, R2,
            T=100000,
            alpha=0.05,
            epsilon=0.15,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0

        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], stride =20, linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Stag Hunt - Epsilon-Greedy",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/stag_hunt_epsilongreedy.png")


def experiment_subsidy_game_epsilongreedy():
    game = GAMES["subsidy_game"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.1, 0.1),
        (0.2, 0.3),
        (0.3, 0.15),
        (0.7, 0.8),
        (0.85, 0.65),
        (0.9, 0.9),
        (0.1, 0.9),
        (0.25, 0.75),
        (0.9, 0.1),
        (0.75, 0.25),
    ]
    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = epsilon_greedy_trajectory(
            R1, R2,
            T=100000,
            alpha=0.05,
            epsilon=0.15,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], stride =20, linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Subsidy Game- Epsilon-Greedy",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/subsidy_game_epsilongreedy.png")


def experiment_prisoners_dilemma_epsilongreedy():
    game = GAMES["prisoners_dilemma"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.1, 0.1),
        (0.2, 0.3),
        (0.3, 0.15),
        (0.7, 0.8),
        (0.85, 0.65),
        (0.9, 0.9),
        (0.1, 0.9),
        (0.25, 0.75),
        (0.9, 0.1),
        (0.75, 0.25),
    ]
    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = epsilon_greedy_trajectory(
            R1, R2,
            T=100000,
            alpha=0.05,
            epsilon=0.15,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], stride =20, linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Prisoners Dilemma- Epsilon-Greedy",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/prisoners_dilemma_epsilongreedy.png")


def experiment_rps_epsilongreedy():
    game = GAMES["biased_RPS"]
    R1, R2 = game

    starts = [
        (np.array([0.7, 0.2, 0.1]), np.array([0.7, 0.2, 0.1])),
        (np.array([0.6, 0.3, 0.1]), np.array([0.5, 0.4, 0.1])),
        (np.array([0.1, 0.7, 0.2]), np.array([0.2, 0.6, 0.2])),
        (np.array([0.1, 0.2, 0.7]), np.array([0.2, 0.1, 0.7])),
        (np.array([1/3, 1/3, 1/3]), np.array([1/3, 1/3, 1/3])),
        (np.array([0.5, 0.4, 0.1]), np.array([0.1, 0.5, 0.4])),
        (np.array([0.2, 0.5, 0.3]), np.array([0.3, 0.2, 0.5])),
    ]
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # trajectories
    for seed, (cx, cy) in enumerate(starts):
        rng = np.random.default_rng(seed)
        traj = epsilon_greedy_trajectory_three(
            R1, R2,
            center1=cx,
            center2=cy,
            T=100000,
            alpha=0.05,
            epsilon=0.15,
            noise_scale=0.0,
            rng=rng,
            seed=seed
        )

        for agent_idx, ax in enumerate(axes):
            add_trajectory_simplex(
                ax, traj,
                color=colors[seed % len(colors)],
                stride=5, linewidth=1.5, alpha=0.85,
                agent=agent_idx
            )

    for ax, title in zip(axes, ["Agent 1", "Agent 2"]):
        finalize_simplex_plot(ax, title=f"Biased RPS - Epsilon-Greedy\n{title}")

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/rps_epsilongreedy.png")


def experiment_rps_boltzmann():
    R1, R2 = GAMES["biased_RPS"]

    raw_starts = [
        (0.90, 0.05, 0.05),
        (0.05, 0.90, 0.05),
        (0.05, 0.05, 0.90),
        (0.50, 0.25, 0.25),
        (0.25, 0.50, 0.25),
        (0.25, 0.25, 0.50),
        (1/3,  1/3,  1/3 ),
        (0.60, 0.30, 0.10),
        (0.10, 0.60, 0.30),
        (0.30, 0.10, 0.60),
    ]
    starts = [(np.array(s), np.array(s)) for s in raw_starts]

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Vector field — same tau as learner so arrows match trajectories
    vf_results1, vf_results2 = replicator_vector_field_boltzmann_3(R1, R2, grid=28, tau=0.05)

    for agent_idx, ax in enumerate(axes):
        vf = vf_results1 if agent_idx == 0 else vf_results2
        add_vector_field_simplex(ax, vf, normalize=True, scale=0.035, color='gray', alpha=0.5)

    # Learning trajectories
    for seed, (cx, cy) in enumerate(starts):
        rng = np.random.default_rng(seed)
        traj = boltzmann_trajectory_three(
            R1, R2,
            center1=cx,
            center2=cy,
            T=100000,
            alpha=0.001,
            tau=0.05,
            noise_scale=0.0,
            rng=rng,
            seed=seed,
        )

        for agent_idx, ax in enumerate(axes):
            add_trajectory_simplex(
                ax, traj,
                color=colors[seed % len(colors)],
                stride=100, linewidth=1.5, alpha=0.85,
                agent=agent_idx,
            )

    nash = np.array([0.625, 0.3125, 0.0625])
    nx, ny = barycentric_to_cartesian(nash[None, :])

    for ax, title in zip(axes, ["Agent 1", "Agent 2"]):
        finalize_simplex_plot(ax, title=f"Agent {axes.tolist().index(ax) + 1} - BiasedRPS (boltzmann)")
        ax.scatter(nx, ny, color='blue', s=150, marker='*', zorder=10, label='Nash EQ')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
        Line2D([0], [0], marker='*', color='blue', markersize=12, linewidth=0, label='Nash EQ'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/rps_boltzmann.png")


def experiment_staghunt_boltzmann():
    game = GAMES["stag_hunt"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.1, 0.1),     (0.15, 0.28),   (0.3, 0.15),
        (0.5, 0.75),    (0.75, 0.5),    (0.85, 0.85),
        (0.35, 0.38),   (0.4, 0.8),
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

   # dynamics = MultiAgentReplicatorDynamics(R1, R2, grid =20)
    #X, Y, U, V = dynamics.boltzmann(T=0.05)
    X, Y, U, V = replicator_vector_field_boltzmann(R1, R2, grid=20)
    #X, Y, U, V = replicator_vector_field( R1, R2, grid =20)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)

    max_payoff = max(np.max(R1), np.max(R2))
    min_payoff = min(np.min(R1), np.min(R2))
    payoff_range = max_payoff - min_payoff if max_payoff != min_payoff else 1.0

    tau_adapted = 1.5 * payoff_range

    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = boltzmann_trajectory(
            R1, R2,
            T=50000,
            alpha=0.001,
            tau_init=0.3,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Stag Hunt - Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/stag_hunt_boltzmann.png")

def experiment_subsidy_game_boltzmann():

    game = GAMES["subsidy_game"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.02,0.02), (0.98,0.98), (0.02,0.98), (0.98,0.02),
        (0.5,0.5), (0.25,0.25), (0.75,0.75), (0.25,0.75),
        (0.75,0.25), (0.4,0.6), (0.6,0.4), (0.1,0.5),
        (0.5,0.1), (0.9,0.5), (0.5,0.9), (0.33,0.66),
        (0.94,0.94)
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    X, Y, U, V = replicator_vector_field_boltzmann( R1, R2, grid =20)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)


    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = boltzmann_trajectory(
            R1, R2,
            T=50000,
            alpha=0.0001,
            tau_init=0.3,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Subsidy Game - Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend

    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/subsidy_game_boltzmann.png")


def experiment_prisoners_dilemma_boltzmann():

    game = GAMES["prisoners_dilemma"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.02,0.02), (0.98,0.98), (0.02,0.98), (0.98,0.02),
        (0.5,0.5), (0.25,0.25), (0.75,0.75), (0.25,0.75),
        (0.75,0.25), (0.4,0.6), (0.6,0.4), (0.1,0.5),
        (0.5,0.1), (0.9,0.5), (0.5,0.9), (0.33,0.66),
        (0.94,0.94)
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    X, Y, U, V = replicator_vector_field_boltzmann( R1, R2, grid =20)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)

    # trajectories
    for seed, (cx, cy) in enumerate(starts):

        traj = boltzmann_trajectory(
            R1, R2,
            T=50000,
            alpha=0.0001,
            tau_init=0.3,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Prisioners dilemma - Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/prisoners_dilemma_boltzmann.png")

def experiment_stag_hunt_lenient_boltzmann():
    game = GAMES["stag_hunt"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.10, 0.10),
        (0.20, 0.15),
        (0.25, 0.30),
        (0.30, 0.20),

        (0.33, 0.33),
        (0.28, 0.40),
        (0.40, 0.28),

        (0.50, 0.70),
        (0.70, 0.50),
        (0.60, 0.80),
        (0.80, 0.60),
        (0.85, 0.85),
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    X, Y, U, V = replicator_vector_field_lenient_boltzmann( R1, R2, grid =20, tau=0.05, k=2)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)

    # trajectories
    for seed, (cx, cy) in enumerate(starts):
        traj = lenient_boltzmann_trajectory(
            R1, R2,
            T=200000,
            alpha=0.001,
            tau_init=2.0,
            tau_min=0.005,
            k =15,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Stag Hunt - Lenient Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/stag_hunt_lenient_boltzmann.png")


def experiment_subsidy_game_lenient_boltzmann():
    game = GAMES["subsidy_game"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
        (0.02,0.02), (0.98,0.98), (0.02,0.98), (0.98,0.02),
        (0.5,0.5), (0.25,0.25), (0.75,0.75), (0.25,0.75),
        (0.75,0.25), (0.4,0.6), (0.6,0.4), (0.1,0.5),
        (0.5,0.1), (0.9,0.5), (0.5,0.9), (0.33,0.66),
        (0.94,0.94)
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    X, Y, U, V = replicator_vector_field_lenient_boltzmann( R1, R2, grid =20, tau=0.05, k=2)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)

    # trajectories
    for seed, (cx, cy) in enumerate(starts):
        traj = lenient_boltzmann_trajectory(
            R1, R2,
            T=400000,
            alpha=0.0005,
            tau_init=3.0,
            tau_min=0.005,
            k =2,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Subsidy Game - Lenient Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/subsidy_game_lenient_boltzmann.png")


def experiment_prisoners_dilemma_lenient_boltzmann():
    game = GAMES["prisoners_dilemma"]
    R1 = game["R1"]
    R2 = game["R2"]

    starts = [
            (0.02,0.02), (0.98,0.98), (0.02,0.98), (0.98,0.02),
            (0.5,0.5), (0.25,0.25), (0.75,0.75), (0.25,0.75),
            (0.75,0.25), (0.4,0.6), (0.6,0.4), (0.1,0.5),
            (0.5,0.1), (0.9,0.5), (0.5,0.9), (0.33,0.66),
            (0.94,0.94)
    ]

    colors = plt.cm.tab10.colors
    _, ax = plt.subplots(figsize=(7, 7))

    X, Y, U, V = replicator_vector_field_lenient_boltzmann( R1, R2, grid =20, tau=0.05, k=2)

    add_vector_field(ax, X, Y, U, V, normalize=True, color='gray', alpha=0.4, scale = 30, width = 0.003)

    # trajectories
    for seed, (cx, cy) in enumerate(starts):
        traj = lenient_boltzmann_trajectory(
            R1, R2,
            T=200000,
            alpha=0.001,
            tau_init=2.0,
            tau_min=0.005,
            k =5,
            seed=seed,
            center=(cx, cy),
            noise_scale=0.0
        )

        add_trajectory( ax, traj, color=colors[seed % len(colors)], linewidth=1.5, alpha=0.85)

    finalize_phase_plot(
        ax,
        title="Prisoners dilemma- Lenient Boltzmann",
        xlabel="Agent 1",
        ylabel="Agent 2"
    )

    # legend
    legend_elements = [
        Line2D([0], [0],marker='o', color='w', markerfacecolor='green', markersize=10,label='Start'),
        Line2D([0], [0], marker='o', color='blue', markersize=10, linewidth=2, label='End'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/prisoners_dilemma_lenient_boltzmann.png")

def experiment_biased_rps_lenient_boltzmann():
    R1, R2 = GAMES["biased_RPS"]

    # Starting points as barycentric coords (rock, paper, scissors)
    starts = [
        (0.90, 0.05, 0.05),  # near Rock
        (0.05, 0.90, 0.05),  # near Paper
        (0.05, 0.05, 0.90),  # near Scissors

        (0.50, 0.25, 0.25),  # Rock-leaning center
        (0.25, 0.50, 0.25),  # Paper-leaning center
        (0.25, 0.25, 0.50),  # Scissors-leaning center

        (1/3,  1/3,  1/3),   # exact center
        (0.60, 0.30, 0.10),
        (0.10, 0.60, 0.30),
        (0.30, 0.10, 0.60),
    ]

    colors = plt.cm.tab10.colors
    _, axes = plt.subplots(1, 2, figsize=(14, 7))

    nash = np.array([0.625, 0.3125, 0.0625])

    vf_results1, vf_results2 = replicator_vector_field_lenient_boltzmann_3(
        R1, R2, grid=28, tau=0.3, k=5
    )

    for agent_idx, ax in enumerate(axes):
        vf = vf_results1 if agent_idx == 0 else vf_results2
        add_vector_field_simplex(ax, vf, normalize=True, scale=0.035,
                              color='gray', alpha=0.5)

        for seed, center in enumerate(starts):
            traj = lenient_boltzmann_trajectory_3(
                R1, R2,
                T=200_000,
                alpha=0.005,
                k=5,
                seed=seed,
                tau_init=0.3,
                tau_min=0.3,
                center=center,
                noise_scale=0.0,
            )

            add_trajectory_simplex(
                ax, traj,
                color=colors[seed % len(colors)],
                linewidth=1.5,
                alpha=0.85,
                stride=50,
                agent=agent_idx,
            )

        finalize_simplex_plot(
            ax,
            title=f"Biased RPS – Lenient Boltzmann\nAgent {agent_idx + 1}",
        )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=10, label='Start'),
        Line2D([0], [0], marker='o', color='blue',
               markersize=10, linewidth=2, label='End'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig("figures/biased_rps_lenient_boltzmann.png")