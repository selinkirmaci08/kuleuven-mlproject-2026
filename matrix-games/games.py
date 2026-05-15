import numpy as np

GAMES = {

    "stag_hunt": {
        "R1": np.array([
            [1, 0],
            [2/3, 2/3]
        ]),
        "R2": np.array([
            [1, 2/3],
            [0, 2/3]
        ])
    },

    "subsidy_game": {
        "R1": np.array([
            [12, 0],
            [11, 10]
        ]),
        "R2": np.array([
            [12, 11],
            [0, 10]
        ])
    },

    "prisoners_dilemma": {
       "R1": np.array([
            [-1, -4],
            [0, -3]
        ]),
        "R2": np.array([
            [-1, 0],
            [-4, -3]
        ])
    },

    "biased_RPS": (
        np.array([[ 0.00, -0.05,  0.25],  # R1: Player 1 payoffs
                  [ 0.05,  0.00, -0.50],
                  [-0.25,  0.50,  0.00]]),
        np.array([[ 0.00,  0.05, -0.25],  # R2: Player 2 payoffs (negated, zero-sum)
                  [-0.05,  0.00,  0.50],
                  [ 0.25, -0.50,  0.00]])
    ),

}