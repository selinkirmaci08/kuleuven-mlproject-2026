import numpy as np

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
def qlearning(reward_matrices, episodes=100, learning_rate=0.8, epsilon=0.1):
    print("perfoming qlearning")

    nactions = reward_matrices[0].shape[0]
    
    Q1 = np.zeros(nactions)
    Q2 = np.zeros(nactions)


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


    print("Q1:", Q1)
    print("Q2:", Q2)
        
    return Q1, Q2
    

for name, reward_matrices in games.items():
    print(name)
    qlearning(reward_matrices)



