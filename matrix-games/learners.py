import numpy as np
from policies import epsilon_greedy_policy, softmax, softmax3

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
    n_actions = R1.shape[0]
    rng = np.random.default_rng(seed)
    if center is None:
        center = np.ones(n_actions) / n_actions

    noise = rng.normal(0, noise_scale, size=n_actions)

    x0 = center + noise
    x0 = np.maximum(x0, 1e-8)
    x0 = x0 / np.sum(x0)
    low = epsilon / n_actions

    x0_clipped = np.clip( x0,low + 1e-8, 1.0)

    p_pure = (x0_clipped - low) / (1.0 - epsilon)

    p_pure = np.maximum(p_pure, 1e-8)
    p_pure = p_pure / np.sum(p_pure)

    Q = np.log(p_pure)
    Q -= np.mean(Q)

    x0 = np.clip(center[0] + rng.normal(0, noise_scale), 0.01, 0.99)
    y0 = np.clip(center[1] + rng.normal(0, noise_scale), 0.01, 0.99)

    # initialize Q's from logits
    low = epsilon / 2.0
    high = 1.0 - low
    
    x0_clipped = np.clip(x0, low + 1e-5, high - 1e-5)
    y0_clipped = np.clip(y0, low + 1e-5, high - 1e-5)

    # 2. Invertir la función de la política para encontrar las probabilidades "puras" (explotación)
    # Bajo epsilon-greedy: p = (1 - epsilon) * p_pura + (epsilon / 2)
    p1_pure = (x0_clipped - low) / (1.0 - epsilon)
    p2_pure = (y0_clipped - low) / (1.0 - epsilon)
    
    # Evitar divisiones por cero o logaritmos de números negativos
    p1_pure = np.clip(p1_pure, 1e-5, 1.0 - 1e-5)
    p2_pure = np.clip(p2_pure, 1e-5, 1.0 - 1e-5)

    # 3. Inicializar Q's usando las probabilidades puras invertidas
    Q1 = np.array([np.log(p1_pure / (1.0 - p1_pure)), 0.0])
    Q2 = np.array([np.log(p2_pure / (1.0 - p2_pure)), 0.0])

    history = []

    history.append([x0, y0])

    for t in range(T):
        pi1 = epsilon_greedy_policy(Q1, epsilon)
        pi2 = epsilon_greedy_policy(Q2, epsilon)

        a1 = rng.choice(2, p=pi1)
        a2 = rng.choice(2, p=pi2)

        r1 = R1[a1, a2]
        r2 = R2[a1, a2]

        alpha_t = alpha / (1.0 + 0.00005 * t)
        
        Q1[a1] += alpha_t * (r1 - Q1[a1])
        Q2[a2] += alpha_t * (r2 - Q2[a2])

        p1 = pi1[0]
        p2 = pi2[0] 
        
        history.append([p1, p2])

    return np.array(history)

def epsilon_greedy_trajectory_three(
    R1,
    R2,
    center1,
    center2,
    T=10000,
    alpha=0.1,
    epsilon=0.1,
    noise_scale=0.05,
    rng=None,
    seed=42
):

    if rng is None:
        rng = np.random.default_rng()

    n_actions_1 = len(center1)
    n_actions_2 = len(center2)

    noise1 = rng.normal(0, noise_scale, size=n_actions_1)
    noise2 = rng.normal(0, noise_scale, size=n_actions_2)

    x0 = center1 + noise1
    y0 = center2 + noise2

    x0 = np.maximum(x0, 1e-8)
    y0 = np.maximum(y0, 1e-8)

    x0 /= np.sum(x0)
    y0 /= np.sum(y0)

    low1 = epsilon / n_actions_1
    low2 = epsilon / n_actions_2

    x0_clipped = np.clip(x0, low1 + 1e-8, 1.0)
    y0_clipped = np.clip(y0, low2 + 1e-8, 1.0)

    p1_pure = (x0_clipped - low1) / (1.0 - epsilon)
    p2_pure = (y0_clipped - low2) / (1.0 - epsilon)

    p1_pure = np.maximum(p1_pure, 1e-8)
    p2_pure = np.maximum(p2_pure, 1e-8)

    p1_pure /= np.sum(p1_pure)
    p2_pure /= np.sum(p2_pure)

    Q1 = np.log(p1_pure)
    Q2 = np.log(p2_pure)

    # Gauge fixing
    Q1 -= np.mean(Q1)
    Q2 -= np.mean(Q2)

    history = []

    history.append([
        epsilon_greedy_policy(Q1, epsilon),
        epsilon_greedy_policy(Q2, epsilon)
    ])

    for t in range(T):
        pi1 = epsilon_greedy_policy(Q1, epsilon)
        pi2 = epsilon_greedy_policy(Q2, epsilon)

        a1 = rng.choice(n_actions_1, p=pi1)
        a2 = rng.choice(n_actions_2, p=pi2)

        r1 = R1[a1, a2]
        r2 = R2[a1, a2]

        alpha_t = alpha / (1.0 + 0.000005 * t)

        Q1[a1] += alpha_t * (r1 - Q1[a1])
        Q2[a2] += alpha_t * (r2 - Q2[a2])

        history.append([pi1.copy(), pi2.copy()])

    return np.array(history, dtype=np.float64)


# ============================================================
# LENIENT BOLTZMANN Q-LEARNING
# (optimistic / asymmetric TD updates)
# ============================================================
def lenient_boltzmann_trajectory(
    R1, R2,
    T=200000,
    alpha=0.005,
    tau=0.3,
    k = 10,
    seed=42,
    tau_init=1.0,     
    tau_min=0.01,    
    
    center=(0.45, 0.45),
    noise_scale=0.02
):
    
    n_actions =2
    rng = np.random.default_rng(seed)

    #tau_init = 1.5   # Inicia con alta aleatoriedad
    #tau_min = 0.01   # Obliga a terminar en estrategia pura (esquinas)

    x0 = np.clip(center[0] + rng.normal(0, noise_scale), 0.01, 0.99)
    y0 = np.clip(center[1] + rng.normal(0, noise_scale), 0.01, 0.99)
    Q1 = policy_to_Q(x0, tau_init)
    Q2 = policy_to_Q(y0, tau_init)
    
    buffers1 = [[] for _ in range(n_actions)]
    buffers2 = [[] for _ in range(n_actions)]

    tau = tau_init
    tau_decay = (tau_min / tau_init) ** (1.0 / T)

    history = []

    for t in range(T):
        # tau decreases over time

        pi1 = softmax(Q1, tau)
        pi2 = softmax(Q2, tau)
        history.append([pi1[0], pi2[0]])

        a1 = rng.choice(2, p=pi1)
        a2 = rng.choice(2, p=pi2)

        r1 = R1[a1,a2]
        r2 = R2[a1,a2]

        buffers1[a1].append(r1)
        buffers2[a2].append(r2)

        #lenient update      

        if len(buffers1[a1]) >= k:
            target1 = max(buffers1[a1])
            td1 = target1 - Q1[a1]
            Q1[a1] += alpha * td1
            buffers1[a1].clear()

        if len(buffers2[a2]) >= k:
            target2 = max(buffers2[a2])
            td2 = target2 - Q2[a2]
            Q2[a2] += alpha * td2
            buffers2[a2].clear()

        tau = max(tau_min, tau * tau_decay)

    return np.array(history)


# ============================================================
# DETERMINISTIC BOLTZMANN Q-LEARNING
# ============================================================

def policy_to_Q(p, tau):

    p = np.clip(p, 1e-5, 1-1e-5)
    q0 = tau * np.log(p)
    q1 = tau * np.log(1.0 - p)
    
    return np.array([q0, q1])

def policy_to_Q3(p, tau):
    probs = np.array(p, dtype=float)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum()
    Q = tau * np.log(probs)
    return Q - Q.mean()

def boltzmann_trajectory(
    R1,
    R2,
    T=20000,
    alpha=0.01,
    tau_init=0.05,
    seed=0,
    center=(0.45, 0.45),
    noise_scale=0.02
):

    rng = np.random.default_rng(seed)
    tau = tau_init
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
        tau =max(tau_init * (0.9999 ** t), 0.02)
        if tau < 1e-4: tau = 1e-4

        pi1 = softmax(Q1, tau)
        pi2 = softmax(Q2, tau)

        history.append([pi1[0], pi2[0]])

        a1 = rng.choice(2, p=pi1)
        a2 = rng.choice(2, p=pi2)
        
        r1 = R1[a1,a2]
        r2 = R2[a1,a2]

        r1_stag = R1[0, a2]
        r1_hare = R1[1, a2]
        
        r2_stag = R2[a1, 0]
        r2_hare = R2[a1, 1]

        Q1[0] += alpha * (r1_stag - Q1[0])
        Q1[1] += alpha * (r1_hare - Q1[1])
        
        Q2[0] += alpha * (r2_stag - Q2[0])
        Q2[1] += alpha * (r2_hare - Q2[1])

    return np.array(history)

def lenient_boltzmann_trajectory_3(
    R1, R2,
    T=200000,
    alpha=0.005,
    k=10,
    seed=42,
    tau_init=1.0,
    tau_min=0.01,
    center=(1/3, 1/3, 1/3),   # initial distribution over R, P, S
    noise_scale=0.02
):
    n_actions = 3
    rng = np.random.default_rng(seed)

    # Initialise starting policy near `center` with small noise
    probs0 = np.array(center, dtype=float)
    probs0 += rng.normal(0, noise_scale, size=n_actions)
    probs0 = np.clip(probs0, 1e-6, None)
    probs0 /= probs0.sum()

    Q1 = policy_to_Q3(probs0.copy(), tau_init)
    Q2 = policy_to_Q3(probs0.copy(), tau_init)

    buffers1 = [[] for _ in range(n_actions)]
    buffers2 = [[] for _ in range(n_actions)]

    tau = tau_init
    tau_decay = (tau_min / tau_init) ** (1.0 / T)

    history = []   # stores (pi1[0], pi1[1], pi2[0], pi2[1]) — rock & paper probs

    for t in range(T):
        pi1 = softmax3(Q1, tau)
        pi2 = softmax3(Q2, tau)

        # Store first two probs; scissors = 1 - sum is recoverable
        history.append([pi1, pi2])          # shape per step: (2, 3)

        a1 = rng.choice(n_actions, p=pi1)
        a2 = rng.choice(n_actions, p=pi2)

        r1 = R1[a1, a2]
        r2 = R2[a1, a2]

        buffers1[a1].append(r1)
        buffers2[a2].append(r2)

        # Lenient update — only when buffer is full
        if len(buffers1[a1]) >= k:
            target1 = max(buffers1[a1])
            Q1[a1] += alpha * (target1 - Q1[a1])
            buffers1[a1].clear()

        if len(buffers2[a2]) >= k:
            target2 = max(buffers2[a2])
            Q2[a2] += alpha * (target2 - Q2[a2])
            buffers2[a2].clear()

        tau = max(tau_min, tau * tau_decay)

    return np.array(history)   # shape: (T, 2, 3)
