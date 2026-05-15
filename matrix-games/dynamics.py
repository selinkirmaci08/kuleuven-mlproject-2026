import numpy as np
from open_spiel.python.egt import dynamics
from plotting import barycentric_to_cartesian


def replicator_vector_field_boltzmann(R1, R2, grid=20):
    x_range = np.linspace(0.01, 0.99, grid)
    y_range = np.linspace(0.01, 0.99, grid)

    X, Y = np.meshgrid(x_range, y_range)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    payoff_tensor = np.stack([R1, R2], axis=0)
    temperature = 0.05
    boltzmann_dynamics = dynamics.MultiPopulationDynamics(
        payoff_tensor,
        lambda state, fitness: dynamics.boltzmannq(state, fitness, temperature)
    )
    for i in range(grid):
        for j in range(grid):
            x_val = X[i, j]
            y_val = Y[i, j]

            state = np.array([x_val, 1.0 - x_val, y_val, 1.0 - y_val])

            dx_dt = boltzmann_dynamics(state)

            U[i, j] = dx_dt[0]
            V[i, j] = dx_dt[2]

    return X, Y, U, V


def replicator_vector_field_lenient_boltzmann(R1, R2, grid=21, tau=0.05, k=2):
    x_range = np.linspace(0.01, 0.99, grid)
    y_range = np.linspace(0.01, 0.99, grid)

    X, Y = np.meshgrid(x_range, y_range)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(grid):
        for j in range(grid):
            x_val = X[i, j]
            y_val = Y[i, j]

            # Políticas actuales de los dos agentes (probabilidades)
            pi1 = np.array([x_val, 1.0 - x_val])
            pi2 = np.array([y_val, 1.0 - y_val])

            # === AJUSTE DE LENIENCY: Cálculo de Recompensas Efectivas ===
            lenient_r1 = np.zeros(2)
            lenient_r2 = np.zeros(2)

            # --- Agente 1 (Sus pagos dependen de las acciones del Agente 2) ---
            for a1 in range(2):
                r_vs_0 = R1[a1, 0]  # Pago si el rival elige Acción 0
                r_vs_1 = R1[a1, 1]  # Pago si el rival elige Acción 1

                r_max = max(r_vs_0, r_vs_1)
                r_min = min(r_vs_0, r_vs_1)

                p_min = pi2[0] if r_vs_0 == r_min else pi2[1]

                prob_todos_min = p_min ** k
                prob_al_menos_un_max = 1.0 - prob_todos_min

                lenient_r1[a1] = (prob_al_menos_un_max * r_max) + (prob_todos_min * r_min)

            for a2 in range(2):
                r_vs_0 = R2[0, a2]  # payoff if rival chooses action 0
                r_vs_1 = R2[1, a2]  # payoff if rival chooses action 1
                r_max = max(r_vs_0, r_vs_1)
                r_min = min(r_vs_0, r_vs_1)

                p_min = pi1[0] if r_vs_0 == r_min else pi1[1]

                prob_todos_min = p_min ** k
                prob_al_menos_un_max = 1.0 - prob_todos_min

                lenient_r2[a2] = (prob_al_menos_un_max * r_max) + (prob_todos_min * r_min)

            exp_1 = np.exp((lenient_r1 - np.max(lenient_r1)) / tau)
            softmax_1 = exp_1 / np.sum(exp_1)

            exp_2 = np.exp((lenient_r2 - np.max(lenient_r2)) / tau)
            softmax_2 = exp_2 / np.sum(exp_2)


            U[i, j] = pi1[0] * (1.0 - pi1[0]) * (softmax_1[0] - pi1[0])
            V[i, j] = pi2[0] * (1.0 - pi2[0]) * (softmax_2[0] - pi2[0])

    return X, Y, U, V

def replicator_vector_field_lenient_boltzmann_3(R1, R2, grid=15, tau=0.3, k=5):
    vals = np.linspace(0.01, 0.98, grid)
    points = []
    for p0 in vals:
        for p1 in vals:
            p2 = 1.0 - p0 - p1
            if 0.01 <= p2 <= 0.99:  # valid simplex point
                points.append((p0, p1, p2))

    def lenient_payoffs(R, pi_self, pi_opp, axis):
        n = len(pi_self)
        lenient_r = np.zeros(n)
        for a in range(n):
            payoffs = R[a, :] if axis == 0 else R[:, a]

            worst_idx = np.argmin(payoffs)  # single worst opponent action
            best_idx = np.argmax(payoffs)  # single best

            r_min = payoffs[worst_idx]
            r_max = payoffs[best_idx]

            p_min = pi_opp[worst_idx]  # probability of that single worst action

            prob_all_min = p_min ** k
            prob_any_better = 1.0 - prob_all_min

            lenient_r[a] = prob_any_better * r_max + prob_all_min * r_min
        return lenient_r

    def boltzmann(r):
        e = np.exp((r - r.max()) / tau)
        return e / e.sum()

    results1, results2 = [], []

    for (p0, p1, p2) in points:
        pi = np.array([p0, p1, p2])

        # --- Agent 1: self=pi, opponent=pi (symmetric point, like x,y in 2-action) ---
        lr1 = lenient_payoffs(R1, pi, pi, axis=0)  # opponent is pi2 = pi (symmetric)
        sigma1 = boltzmann(lr1)
        d1 = pi * (sigma1 - pi)
        results1.append((pi.copy(), d1))

        # --- Agent 2: self=pi, opponent=pi ---
        lr2 = lenient_payoffs(R2, pi, pi, axis=1)
        sigma2 = boltzmann(lr2)
        d2 = pi * (sigma2 - pi)
        results2.append((pi.copy(), d2))

    return results1, results2


def replicator_vector_field_boltzmann_3(R1, R2, grid=20, tau=0.05):
    """
    Boltzmann replicator vector field on the 3-action simplex.
    For biased RPS, R2.T == R1 so both agents have the same vector field.
    Agent1 fitness for a1: R1 @ pi_opp
    Agent2 fitness for a2: R2.T @ pi_opp == R1 @ pi_opp
    """

    def boltzmann_delta(payoff_row, pi_self, pi_opp):
        fitness = payoff_row @ pi_opp
        logits = (fitness - fitness.max()) / tau
        sigma = np.exp(logits)
        sigma /= sigma.sum()
        return pi_self * (sigma - pi_self)

    vals = np.linspace(0.02, 0.96, grid)
    results1, results2 = [], []

    for p0 in vals:
        for p1 in vals:
            p2 = 1.0 - p0 - p1
            if p2 < 0.02 or p2 > 0.96:
                continue
            pi = np.array([p0, p1, p2])

            d1 = boltzmann_delta(R1, pi, pi)
            results1.append((pi.copy(), d1))

            d2 = boltzmann_delta(R2.T, pi, pi)
            results2.append((pi.copy(), d2))

    return results1, results2

    return results1, results2


def add_vector_field_simplex(ax, results, normalize=True,
                             color='gray', alpha=0.5, scale=0.035, width=0.002):
    for (pi, d_pi) in results:
        x0, y0 = barycentric_to_cartesian(pi[None, :])

        bary_tip = np.clip(pi + d_pi, 1e-6, None)
        bary_tip /= bary_tip.sum()
        x1, y1 = barycentric_to_cartesian(bary_tip[None, :])

        dx, dy = x1[0] - x0[0], y1[0] - y0[0]

        if normalize:
            norm = np.hypot(dx, dy)
            if norm > 1e-10:
                dx, dy = dx / norm * scale, dy / norm * scale

        ax.annotate(
            "", xy=(x0[0] + dx, y0[0] + dy), xytext=(x0[0], y0[0]),
            arrowprops=dict(arrowstyle='->', color=color, alpha=alpha, lw=0.8)
        )