
# ============================================================
# REPLICATOR VECTOR FIELD
# ============================================================

import numpy as np
from open_spiel.python.egt import dynamics
from plotting import barycentric_to_cartesian

def replicator_vector_field_boltzmann(R1, R2, grid =20):
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
    # Usar un grid impar (ej. 21) asegura que el centro exacto (0.5, 0.5) sea evaluado
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

            # --- Agente 2 (Sus pagos dependen de las acciones del Agente 1) ---
            for a2 in range(2):
                r_vs_0 = R2[0, a2]  # Pago si el rival elige Acción 0
                r_vs_1 = R2[1, a2]  # Pago si el rival elige Acción 1
                
                r_max = max(r_vs_0, r_vs_1)
                r_min = min(r_vs_0, r_vs_1)
                
                p_min = pi1[0] if r_vs_0 == r_min else pi1[1]
                
                prob_todos_min = p_min ** k
                prob_al_menos_un_max = 1.0 - prob_todos_min
                
                lenient_r2[a2] = (prob_al_menos_un_max * r_max) + (prob_todos_min * r_min)

            # === DINÁMICA DE ACTUALIZACIÓN CONTINUA ===
            # Evitar desbordamientos (overflow) restando el máximo antes de la exponencial
            exp_1 = np.exp((lenient_r1 - np.max(lenient_r1)) / tau)
            softmax_1 = exp_1 / np.sum(exp_1)
            
            exp_2 = np.exp((lenient_r2 - np.max(lenient_r2)) / tau)
            softmax_2 = exp_2 / np.sum(exp_2)
            
            # CORRECCIÓN: Multiplicar por la probabilidad actual (Dinámica de Replicador Boltzmann)
            # Esto estabiliza las fronteras del simplex y centra los cuatro cuadrantes
            U[i, j] = pi1[0] * (1.0 - pi1[0]) * (softmax_1[0] - pi1[0])
            V[i, j] = pi2[0] * (1.0 - pi2[0]) * (softmax_2[0] - pi2[0])
            
    return X, Y, U, V
def replicator_vector_field_lenient_boltzmann_3(R1, R2, grid=15, tau=0.3, k=5,
                                                  opponent_policy=None):
    """
    Vector field on the simplex for agent i, with opponent fixed at
    `opponent_policy` (defaults to uniform / Nash for RPS).
    """
    if opponent_policy is None:
        opponent_policy = np.array([1/3, 1/3, 1/3])  # Nash for standard RPS

    vals = np.linspace(0.05, 0.95, grid)
    points = []
    for p0 in vals:
        for p1 in vals:
            p2 = 1.0 - p0 - p1
            if p2 > 0.05:
                points.append((p0, p1, p2))

    def lenient_payoffs(R, pi_self, pi_opp, axis):
        """
        axis=0: agent is row player (R indexed as R[a_self, a_opp])
        axis=1: agent is col player (R indexed as R[a_opp, a_self])
        """
        n = len(pi_self)
        lenient_r = np.zeros(n)
        for a in range(n):
            if axis == 0:
                payoffs = R[a, :]          # vs each opponent action
            else:
                payoffs = R[:, a]
            r_max = payoffs.max()
            r_min = payoffs.min()
            a_min = payoffs.argmin()
            p_min = pi_opp[a_min]
            prob_all_min = p_min ** k
            lenient_r[a] = (1 - prob_all_min) * r_max + prob_all_min * r_min
        return lenient_r

    def boltzmann(r):
        e = np.exp((r - r.max()) / tau)
        return e / e.sum()

    results1, results2 = [], []

    for (p0, p1, p2) in points:
        pi = np.array([p0, p1, p2])

        # Agent 1: own policy = pi, opponent fixed at opponent_policy
        lr1 = lenient_payoffs(R1, pi, opponent_policy, axis=0)
        sigma1 = boltzmann(lr1)
        d1 = pi * (sigma1 - pi)
        results1.append((pi.copy(), d1))

        # Agent 2: own policy = pi, opponent fixed at opponent_policy
        lr2 = lenient_payoffs(R2, pi, opponent_policy, axis=1)
        sigma2 = boltzmann(lr2)
        d2 = pi * (sigma2 - pi)
        results2.append((pi.copy(), d2))

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