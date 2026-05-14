import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# MATRIX GAMES
# ============================================================

games = {
    "stag_hunt": (
        np.array([
            [1, 0],
            [2/3, 2/3]
        ]),
        np.array([
            [1, 2/3],
            [0, 2/3]
        ])
    ),

    "subsidy_game": (
        np.array([
            [12, 0],
            [11, 10]
        ]),
        np.array([
            [12, 11],
            [0, 10]
        ])
    ),

    "prisoners_dilemma": (
        np.array([
            [-1, -4],
            [0, -3]
        ]),
        np.array([
            [-1, 0],
            [-4, -3]
        ])
    ),

    "biased_RPS": (
        np.array([
            [0.00, -0.05, 0.25],
            [0.05, 0.00, -0.50],
            [-0.25, 0.50, 0.00]
        ]),
        np.array([
            [0.00, 0.05, -0.25],
            [-0.05, 0.00, 0.50],
            [0.25, -0.50, 0.00]
        ])
    ),
}


# ============================================================
# POLICY HELPERS
# ============================================================

def epsilon_greedy_policy(Q, epsilon):
    nA = len(Q)

    pi = np.ones(nA) * (epsilon / nA)

    best_action = np.argmax(Q)
    pi[best_action] += (1.0 - epsilon)

    return pi


def softmax_policy(Q, temperature):
    logits = Q / max(temperature, 1e-8)

    # numerical stability
    logits -= np.max(logits)

    exp_q = np.exp(logits)

    return exp_q / np.sum(exp_q)


# ============================================================
# MOVING AVERAGE SMOOTHING
# ============================================================

def moving_average(history, window=100):

    smoothed = np.zeros_like(history)

    for i in range(len(history)):

        start = max(0, i - window)

        smoothed[i] = np.mean(history[start:i+1], axis=0)

    return smoothed


# ============================================================
# EPSILON GREEDY Q-LEARNING
# ============================================================

def qlearning(
    reward_matrices,
    episodes=30000,
    learning_rate=0.005,
    epsilon=0.1,
    save_every=10,
    seed=0
):

    R1, R2 = reward_matrices

    nactions = R1.shape[0]

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100)

    #Q1 = rng1.uniform(-0.1, 0.1, nactions)
   # Q2 = rng2.uniform(-0.1, 0.1, nactions)
    
    Q1 = rng1.uniform(-2.0, 2.0, nactions)
    Q2 = rng2.uniform(-2.0, 2.0, nactions)

    policy_history1 = []
    policy_history2 = []

    for episode in range(episodes):

        # current policies
        pi1 = epsilon_greedy_policy(Q1, epsilon)
        pi2 = epsilon_greedy_policy(Q2, epsilon)

        # sample actions
        a1 = rng1.choice(nactions, p=pi1)
        a2 = rng2.choice(nactions, p=pi2)

        # rewards
        #r1 = R1[a1, a2]
        #r2 = R2[a1, a2]

        # Q updates
        #Q1[a1] += learning_rate * (r1 - Q1[a1])
        #Q2[a2] += learning_rate * (r2 - Q2[a2])
        expected_r1 = R1 @ pi2
        expected_r2 = R2.T @ pi1

        Q1 += learning_rate * (expected_r1 - Q1)
        Q2 += learning_rate * (expected_r2 - Q2)

        # save instantaneous policies
        if episode % save_every == 0:
            policy_history1.append(pi1.copy())
            policy_history2.append(pi2.copy())

    return np.array(policy_history1), np.array(policy_history2)


# ============================================================
# BOLTZMANN Q-LEARNING
# ============================================================

def boltzmann_qlearning(
    reward_matrices,
    episodes=30000,
    learning_rate=0.005,
    temperature=1.5,
    save_every=10,
    seed=0
):

    R1, R2 = reward_matrices

    nactions = R1.shape[0]

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100)

    #Q1 = rng1.uniform(-0.1, 0.1, nactions)
    #Q2 = rng2.uniform(-0.1, 0.1, nactions)

    Q1 = rng1.uniform(-2.0, 2.0, nactions)
    Q2 = rng2.uniform(-2.0, 2.0, nactions)

    policy_history1 = []
    policy_history2 = []

    for episode in range(episodes):

        pi1 = softmax_policy(Q1, temperature)
        pi2 = softmax_policy(Q2, temperature)

        expected_r1 = R1 @ pi2
        expected_r2 = R2.T @ pi1

        Q1 += learning_rate * (expected_r1 - Q1)
        Q2 += learning_rate * (expected_r2 - Q2)

        policy_history1.append(pi1.copy())
        policy_history2.append(pi2.copy())

    return np.array(policy_history1), np.array(policy_history2)


# ============================================================
# LENIENT BOLTZMANN Q-LEARNING
# ============================================================

def lenient_boltzmann_qlearning(
    reward_matrices,
    episodes=30000,
    learning_rate=0.005,
    maxTemp=5.0,
    alpha=0.1,
    beta=2.0,
    delta=0.9995,
    save_every=10,
    seed=0
):

    R1, R2 = reward_matrices

    nactions = R1.shape[0]

    rng = np.random.default_rng(seed)

    # utilities
    U1 = rng.uniform(-0.01, 0.01, nactions)
    U2 = rng.uniform(-0.01, 0.01, nactions)

    # temperatures
    Temp1 = np.full(nactions, maxTemp)
    Temp2 = np.full(nactions, maxTemp)

    policy_history1 = []
    policy_history2 = []

    for episode in range(episodes):

        # policies
        pi1 = softmax_policy(U1, np.mean(Temp1))
        pi2 = softmax_policy(U2, np.mean(Temp2))

        # actions
        a1 = rng.choice(nactions, p=pi1)
        a2 = rng.choice(nactions, p=pi2)

        # rewards
        r1 = R1[a1, a2]
        r2 = R2[a1, a2]

        # =========================
        # LENIENT UPDATE PLAYER 1
        # =========================

        randval = rng.random()

        leniency1 = np.exp(-alpha * Temp1[a1])

        if (r1 >= U1[a1]) or (randval < leniency1):

            # FIXED UPDATE
            U1[a1] += learning_rate * (r1 - U1[a1])

        Temp1[a1] *= delta

        # =========================
        # LENIENT UPDATE PLAYER 2
        # =========================

        randval = rng.random()

        leniency2 = np.exp(-alpha * Temp2[a2])

        if (r2 >= U2[a2]) or (randval < leniency2):

            # FIXED UPDATE
            U2[a2] += learning_rate * (r2 - U2[a2])

        Temp2[a2] *= delta

        # save policies
        if episode % save_every == 0:
            policy_history1.append(pi1.copy())
            policy_history2.append(pi2.copy())

    return np.array(policy_history1), np.array(policy_history2)


# ============================================================
# TRAJECTORY PLOTTING
# ============================================================
def plot_trajectories(
    game_name,
    reward_matrices,
    algorithm="boltzmann",
    n_seeds=30
):

    trajectories_x = []
    trajectories_y = []

    for seed in range(n_seeds):

        if algorithm == "egreedy":

            h1, h2 = qlearning(
                reward_matrices,
                episodes=50000,
                learning_rate=0.002,
                epsilon=0.15,
                save_every=1,
                seed=seed
            )

        elif algorithm == "boltzmann":

            h1, h2 = boltzmann_qlearning(
                reward_matrices,
                episodes=50000,
                learning_rate=0.002,
                temperature=1.5,
                save_every=1,
                seed=seed
            )

        elif algorithm == "lenient":

            h1, h2 = lenient_boltzmann_qlearning(
                reward_matrices,
                episodes=50000,
                learning_rate=0.002,
                maxTemp=3.0,
                save_every=1,
                seed=seed
            )

        # HEAVY smoothing
        h1 = moving_average(h1, window=500)
        h2 = moving_average(h2, window=500)

        trajectories_x.append(h1[:,0])
        trajectories_y.append(h2[:,0])

    trajectories_x = np.array(trajectories_x)
    trajectories_y = np.array(trajectories_y)

    # ==========================================
    # EMPIRICAL MEAN TRAJECTORY
    # ==========================================

    mean_x = np.mean(trajectories_x, axis=0)
    mean_y = np.mean(trajectories_y, axis=0)

    # ==========================================
    # PLOT
    # ==========================================

    plt.figure(figsize=(6,6))

    # light individual traces
    for i in range(n_seeds):

        plt.plot(
            trajectories_x[i],
            trajectories_y[i],
            alpha=0.15,
            linewidth=1
        )

    # main empirical trajectory
    plt.plot(
        mean_x,
        mean_y,
        linewidth=4
    )

    # start/end markers
    plt.scatter(
        mean_x[0],
        mean_y[0],
        color='green',
        s=80,
        zorder=5
    )

    plt.scatter(
        mean_x[-1],
        mean_y[-1],
        color='red',
        s=80,
        zorder=5
    )

    plt.xlabel("P(Player 1 plays action 0)")
    plt.ylabel("P(Player 2 plays action 0)")

    plt.title(f"{algorithm}: {game_name}")

    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# RUN ALL 2x2 GAMES
# ============================================================

for name, reward_matrices in games.items():

    if reward_matrices[0].shape[0] == 2:

        plot_trajectories(
            name,
            reward_matrices,
            algorithm="egreedy",
            n_seeds=8
        )

        plot_trajectories(
            name,
            reward_matrices,
            algorithm="boltzmann",
            n_seeds=8
        )

        plot_trajectories(
            name,
            reward_matrices,
            algorithm="lenient",
            n_seeds=8
        )