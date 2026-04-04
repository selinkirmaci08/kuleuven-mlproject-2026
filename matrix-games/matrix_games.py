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
def qlearning(reward_matrices, episodes=100_000, learning_rate=0.1, epsilon=0.1):
    print("perfoming qlearning")

    nactions = reward_matrices[0].shape[0]
    
    Q1 = np.zeros(nactions)
    Q2 = np.zeros(nactions)


    avg_pi1 = np.zeros(nactions)
    avg_pi2 = np.zeros(nactions)
    hist1, hist2 = [], []

    for episode in range(episodes):
        #S = np.zeros()
        # choose either greedy or random action
        # define a tie breaking policy -> max function

        # player 1
        # choose random number between 0 and 1
        random1 = np.random.rand()
        # print(random1)
        if random1 < epsilon:
            # random action
            # print("choosing a random action for p2")
            action_p1 = np.random.randint(0, nactions)
            value_p1 = Q1[action_p1]
        else:
           #  print("choosing the best action for p2")
            value_p1 = np.max(Q1)
            action_p1 = np.argmax(Q1)

        # player 2
        # choose random number between 0 and 1
        random2 = np.random.rand()
        # print(random2)
        if random2 < epsilon:
            # random action
            # print("choosing a random action for p2")
            action_p2 = np.random.randint(0, nactions)
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


def plot_trajectories(game_name, rewardMatrices, n_seeds=5):
    plt.figure(figsize=(5, 5))
    for seed in range(n_seeds):
        np.random.seed(seed)
        h1, h2 = qlearning(rewardMatrices)
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



