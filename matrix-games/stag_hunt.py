import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STAG HUNT
# ============================================================

R1 = np.array([
    [1, 0],
    [2/3, 2/3]
])

R2 = np.array([
    [1, 2/3],
    [0, 2/3]
])


# ============================================================
# SOFTMAX POLICY
# ============================================================

def softmax(Q, tau):

    logits = Q / tau
    logits -= np.max(logits)

    expq = np.exp(logits)

    return expq / np.sum(expq)

# ============================================================
# EPSILON-GREEDY POLICY
# ============================================================

def epsilon_greedy_policy(Q, epsilon):
    best = np.argmax(Q)

    pi = np.ones(len(Q)) * (epsilon / len(Q))
    pi[best] += 1 - epsilon

    return pi


def epsilon_greedy_trajectory(
    R1,
    R2,
    T=20000,
    alpha=0.01,
    epsilon=0.1,
    seed=0,
    center=(0.45, 0.45),
    noise_scale=0.02
):

    rng = np.random.default_rng(seed)

    x0 = np.clip(center[0] + rng.normal(0, noise_scale), 0.01, 0.99)
    y0 = np.clip(center[1] + rng.normal(0, noise_scale), 0.01, 0.99)

    # initialize Q's from logits
    Q1 = np.array([np.log(x0 / (1 - x0)), 0.0])
    Q2 = np.array([np.log(y0 / (1 - y0)), 0.0])

    history = []

    for t in range(T):

        pi1 = epsilon_greedy_policy(Q1, epsilon)
        pi2 = epsilon_greedy_policy(Q2, epsilon)

        expected_r1 = R1 @ pi2
        expected_r2 = R2.T @ pi1

        Q1 += alpha * (expected_r1 - Q1)
        Q2 += alpha * (expected_r2 - Q2)

        history.append([pi1[0], pi2[0]])

    return np.array(history)


# ============================================================
# LENIENT BOLTZMANN Q-LEARNING
# (optimistic / asymmetric TD updates)
# ============================================================

def lenient_boltzmann_trajectory(
    R1,
    R2,
    T=20000,
    alpha=0.01,
    tau=0.3,
    leniency=0.8,
    seed=0,
    center=(0.45, 0.45),
    noise_scale=0.02
):

    rng = np.random.default_rng(seed)

    x0 = np.clip(center[0] + rng.normal(0, noise_scale), 0.01, 0.99)
    y0 = np.clip(center[1] + rng.normal(0, noise_scale), 0.01, 0.99)

    Q1 = policy_to_Q(x0, tau)
    Q2 = policy_to_Q(y0, tau)

    history = []

    for t in range(T):

        pi1 = softmax(Q1, tau)
        pi2 = softmax(Q2, tau)

        expected_r1 = R1 @ pi2
        expected_r2 = R2.T @ pi1

        # TD-style update
        td1 = expected_r1 - Q1
        td2 = expected_r2 - Q2

        # lenient update: amplify positive errors, damp negative ones
        td1 = np.where(td1 > 0, td1, leniency * td1)
        td2 = np.where(td2 > 0, td2, leniency * td2)

        Q1 += alpha * td1
        Q2 += alpha * td2

        history.append([pi1[0], pi2[0]])

    return np.array(history)
# ============================================================
# DETERMINISTIC BOLTZMANN Q-LEARNING
# ============================================================
def policy_to_Q(p, tau):
    """
    Convert Bernoulli policy probability into
    equivalent 2-action softmax logits.
    """

    p = np.clip(p, 1e-6, 1 - 1e-6)

    logit = tau * np.log(p / (1 - p))

    return np.array([logit, 0.0])


def boltzmann_trajectory(
    R1,
    R2,
    T=20000,
    alpha=0.01,
    tau=0.3,
    seed=0,
    center=(0.45, 0.45),
    noise_scale=0.02
):

    rng = np.random.default_rng(seed)

    # =====================================================
    # INITIAL POLICY NEAR SHARED CENTER
    # =====================================================

    x0 = center[0] + rng.normal(0, noise_scale)
    y0 = center[1] + rng.normal(0, noise_scale)

    x0 = np.clip(x0, 0.01, 0.99)
    y0 = np.clip(y0, 0.01, 0.99)

    # =====================================================
    # CONVERT POLICY -> Q VALUES
    # =====================================================

    Q1 = policy_to_Q(x0, tau)
    Q2 = policy_to_Q(y0, tau)

    history = []

    # =====================================================
    # DETERMINISTIC LEARNING DYNAMICS
    # =====================================================

    for t in range(T):

        pi1 = softmax(Q1, tau)
        pi2 = softmax(Q2, tau)

        expected_r1 = R1 @ pi2
        expected_r2 = R2.T @ pi1

        Q1 += alpha * (expected_r1 - Q1)
        Q2 += alpha * (expected_r2 - Q2)

        history.append([pi1[0], pi2[0]])

    return np.array(history)

def policy_to_Q(p, tau):

    p = np.clip(p, 1e-5, 1-1e-5)

    logit = tau * np.log(p / (1-p))

    return np.array([logit, 0.0])


# ============================================================
# REPLICATOR VECTOR FIELD
# ============================================================

def replicator_vector_field(R1, R2, grid=25):

    xs = np.linspace(0.01, 0.99, grid)
    ys = np.linspace(0.01, 0.99, grid)

    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(grid):

        for j in range(grid):

            x = X[i,j]
            y = Y[i,j]

            pi1 = np.array([x, 1-x])
            pi2 = np.array([y, 1-y])

            # player 1 payoffs
            u1_a0 = R1[0] @ pi2
            u1_a1 = R1[1] @ pi2
            u1_avg = pi1[0]*u1_a0 + pi1[1]*u1_a1

            # replicator equation
            dx = x * (u1_a0 - u1_avg)

            # player 2 payoffs
            u2_a0 = pi1 @ R2[:,0]
            u2_a1 = pi1 @ R2[:,1]
            u2_avg = pi2[0]*u2_a0 + pi2[1]*u2_a1

            dy = y * (u2_a0 - u2_avg)

            U[i,j] = dx
            V[i,j] = dy

    return X, Y, U, V


# ============================================================
# PLOT EVERYTHING
# ============================================================

plt.figure(figsize=(8,8))


# VECTOR FIELD
X, Y, U, V = replicator_vector_field(R1, R2)

plt.streamplot(
    X,
    Y,
    U,
    V,
    density=1.2,
    linewidth=1,
    arrowsize=1
)


# EMPIRICAL TRAJECTORIES
for seed in range(8):

    traj = boltzmann_trajectory(
        R1,
        R2,
        T=20000,
        alpha=0.01,
        tau=0.3,
        seed=seed,
        center=(0.45,0.45),
        noise_scale=0.015
    )
    x = traj[:,0]
    y = traj[:,1]
    

    plt.plot(x, y, linewidth=2)

    plt.scatter(x[0], y[0], color='green', s=30)
    plt.scatter(x[-1], y[-1], color='red', s=30)

plt.xlabel("P(Player 1 plays action 0)")
plt.ylabel("P(Player 2 plays action 0)")

plt.title("Empirical Policy Traces + Replicator Dynamics")

plt.xlim(0,1)
plt.ylim(0,1)

plt.grid(alpha=0.3)

plt.show()