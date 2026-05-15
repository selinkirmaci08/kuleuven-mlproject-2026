import numpy as np
import matplotlib.pyplot as plt

# matrix games
games = {
    "stag_hunt": (
        np.array([[1,  0  ],            # R1: Player 1 payoffs
                  [2/3, 2/3]]),
        np.array([[1,   2/3],           # R2: Player 2 payoffs
                  [0,   2/3]])
    ),
    "subsidy_game": (
        np.array([[12,  0],             # R1: Player 1 payoffs
                  [11, 10]]),
        np.array([[12, 11],             # R2: Player 2 payoffs
                  [ 0, 10]])
    ),
    "prisoners_dilemma": (
        np.array([[-1, -4],             # R1: Player 1 payoffs ✓
                  [ 0, -3]]),
        np.array([[-1,  0],             # R2: Player 2 payoffs ✓
                  [-4, -3]])
    ),
    "biased_RPS": (
        np.array([[ 0.00, -0.05,  0.25],  # R1: Player 1 payoffs
                  [ 0.05,  0.00, -0.50],
                  [-0.25,  0.50,  0.00]]),
        np.array([[ 0.00,  0.05, -0.25],  # R2: Player 2 payoffs (negated, zero-sum)
                  [-0.05,  0.00,  0.50],
                  [ 0.25, -0.50,  0.00]])
    ),
}
# e greedy q-learning (off policy TD)
def qlearning(reward_matrices, episodes=100_000, learning_rate=0.1, epsilon=0.1, seed=0):
    print("perfoming qlearning")

    nactions = reward_matrices[0].shape[0]
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100)
    Q1 = rng1.uniform(-0.1, 0.1, nactions)
    Q2 = rng2.uniform(-0.1, 0.1, nactions)


    avg_pi1 = np.zeros(nactions)
    avg_pi2 = np.zeros(nactions)
    hist1, hist2 = [], []

    for episode in range(episodes):
        #S = np.zeros()
        # choose either greedy or random action
        # define a tie breaking policy -> max function

        # player 1
        # choose random number between 0 and 1
        random1 = rng1.random()
        # print(random1)
        if random1 < epsilon:
            # random action
            # print("choosing a random action for p2")
            action_p1 = rng1.integers(0, nactions)
            value_p1 = Q1[action_p1]
        else:
           #  print("choosing the best action for p2")
            value_p1 = np.max(Q1)
            action_p1 = np.argmax(Q1)

        # player 2
        # choose random number between 0 and 1
        random2 = rng2.random()
        # print(random2)
        if random2 < epsilon:
            # random action
            # print("choosing a random action for p2")
            action_p2 = rng2.integers(0, nactions)
            value_p2 = Q2[action_p2]
        else:
            # print("choosing the best action for p2")
            value_p2 = np.max(Q2)
            action_p2 = np.argmax(Q2)

        R1 = reward_matrices[0]
        R2 = reward_matrices[1]

        r1 = R1[action_p1, action_p2]
        r2 = R2[action_p1, action_p2]

        # update Q1
        value_p1 = value_p1 + learning_rate * (r1 - value_p1)
        Q1[action_p1] = value_p1

        # update Q2
        value_p2 = value_p2 + learning_rate * (r2 - value_p2)
        Q2[action_p2] = value_p2

        avg_pi1 += (np.eye(nactions)[action_p1] - avg_pi1) / (episode+1)
        avg_pi2 += (np.eye(nactions)[action_p2] - avg_pi2) / (episode+1)

        if episode % 500 == 0:
            hist1.append(avg_pi1.copy())
            hist2.append(avg_pi2.copy())

    print("Q1:", Q1)
    print("Q2:", Q2)

    return np.array(hist1), np.array(hist2)

def boltzmann_qlearning(reward_matrices, episodes=100_000, learning_rate=0.1, temperature=0.1, seed=0):
    print("perfoming qlearning")

    nactions = reward_matrices[0].shape[0]
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 100)
    Q1 = rng1.uniform(-0.1, 0.1, nactions)
    Q2 = rng2.uniform(-0.1, 0.1, nactions)

    avg_pi1 = np.zeros(nactions)
    avg_pi2 = np.zeros(nactions)
    hist1, hist2 = [], []

    for episode in range(episodes):
        # S = np.zeros()
        # choose either greedy or random action
        # define a tie breaking policy -> max function

        # player 1
        # choose action according to highest probability
        exp_q1 = np.exp(Q1 / temperature)
        pi1 = exp_q1 / np.sum(exp_q1)
        action_p1 = rng1.choice(nactions, p=pi1)

        # player 2
        # choose action according to highest probability
        exp_q2 = np.exp(Q2 / temperature)
        pi2 = exp_q2 / np.sum(exp_q2)
        action_p2 = rng2.choice(nactions, p=pi2)

        R1 = reward_matrices[0]
        R2 = reward_matrices[1]

        r1 = R1[action_p1, action_p2]
        r2 = R2[action_p1, action_p2]

        # update Q1
        value_p1 = Q1[action_p1] + learning_rate * (r1 - Q1[action_p1])
        Q1[action_p1] = value_p1

        # update Q2
        value_p2 = Q2[action_p2] + learning_rate * (r2 - Q2[action_p2])
        Q2[action_p2] = value_p2

        avg_pi1 += (np.eye(nactions)[action_p1] - avg_pi1) / (episode + 1)
        avg_pi2 += (np.eye(nactions)[action_p2] - avg_pi2) / (episode + 1)

        if episode % 500 == 0:
            hist1.append(avg_pi1.copy())
            hist2.append(avg_pi2.copy())

    print("Q1:", Q1)
    print("Q2:", Q2)

    return np.array(hist1), np.array(hist2)

# alpha = temperature multiplication coefficient
# beta  = exponent coefficient
# delta = temperature decay coefficient
def lenient_boltzmann_qlearning(
        reward_matrices, episodes = 100_000, learning_rate=0.1, maxTemp = 10.0, 
        alpha = 0.1, beta = 1.0, delta = 0.999, seed =0
):
    nactions = reward_matrices[0].shape[0]

    # initial settings
    rng = np.random.default_rng(seed)
    
    # utilities (Q values)
    U1 = rng.uniform(0,0.001, nactions)
    U2 = rng.uniform(0,0.001, nactions)

    # temperatures
    Temp1 = np.full(nactions, maxTemp)
    Temp2 = np.full(nactions, maxTemp)

    # for plotting purposes
    avg_pi1 = np.zeros(nactions)
    avg_pi2 = np.zeros(nactions)
    hist1, hist2 = [], []

    R1, R2 = reward_matrices

    for episode in range(episodes):
        # action selection
        logits1 = U1 / np.maximum(Temp1, 1e-6)
        logits1 -= np.max(logits1)    
        W1 = np.exp(logits1)
        W1 /= np.sum(W1)
        a1 = rng.choice(nactions, p=W1)

        logits2 = U2 / np.maximum(Temp2, 1e-6)
        logits2 -= np.max(logits2)
        W2 = np.exp(logits2)
        W2 /= np.sum(W2)
        a2 = rng.choice(nactions, p=W2)
        
        r1 = R1[a1, a2]
        r2 = R2[a1, a2]

        #W1[episode] = (e^(U[action]/MinTemp))/(sum(e^(U[action]/MinTemp)))
        # use probability distribution w1... Wn to select action i
        
        # for player 1
        RandVal = rng.random()
        leniency_term = 1e-2 + beta ** (-alpha * Temp1[a1])
        if (U1[a1] <= r1) or (RandVal < leniency_term):
            U1[a1] = learning_rate * U1[a1] + (1 - learning_rate) * r1
        
        Temp1[a1] *= delta

        # for player 2
        RandVal = rng.random()
        leniency_term = 1e-2 + beta ** (-alpha * Temp2[a2])
        if (U2[a2] <= r2) or (RandVal < leniency_term):
            U2[a2] = learning_rate * U2[a2] + (1 - learning_rate) * r2
        
        Temp2[a2] *= delta

        # histogram tracking
        avg_pi1 += (np.eye(nactions)[a1] - avg_pi1) / (episode + 1)
        avg_pi2 += (np.eye(nactions)[a2] - avg_pi2) / (episode + 1)
        
        if episode % 500 == 0:
            hist1.append(avg_pi1.copy())
            hist2.append(avg_pi2.copy())

    print("U1:", U1)
    print("U2:", U2)

    return np.array(hist1), np.array(hist2)



def plot_trajectories(game_name, rewardMatrices, n_seeds=5):
    plt.figure(figsize=(5, 5))
    for seed in range(n_seeds):
        np.random.seed(seed)
        h1, h2 = lenient_boltzmann_qlearning(rewardMatrices,seed=seed)
        plt.plot(h1[:, 0], h2[:, 0], linewidth=1.2)
        plt.scatter(h1[0, 0],  h2[0, 0],  color='green', s=30, zorder=5)
        plt.scatter(h1[-1, 0], h2[-1, 0], color='red',   s=50, zorder=5)

    plt.xlabel("P(Player 1 plays action 0)")
    plt.ylabel("P(Player 2 plays action 0)")
    plt.title(f"ε-greedy independent: {game_name}")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


for name, reward_matrices in games.items():
    if reward_matrices[0].shape[0] == 2:
        plot_trajectories(name, reward_matrices)

#boltzmann_qlearning(games.get("prisoners_dilemma"))



