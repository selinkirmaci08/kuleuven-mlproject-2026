# POLICY HELPERS
import numpy as np

def softmax(Q, tau):
   # expQ = np.exp(Q / tau)
   # return expQ / np.sum(expQ)
    logits = Q / tau
    logits -= np.max(logits)
    expq = np.exp(logits)
    return expq / np.sum(expq) 

def softmax3(Q, tau):
   # expQ = np.exp(Q / tau)
   # return expQ / np.sum(expQ)
    Q_shifted = Q - np.max(Q)
    exp_Q = np.exp(Q_shifted / tau)
    return exp_Q / exp_Q.sum()

def epsilon_greedy_policy(Q, epsilon):
    n = len(Q)
    # 1. Inicializar con la probabilidad base de exploración uniforme
    pi = np.ones(n) * (epsilon / n)  
    
    # 2. Encontrar cuál o cuáles acciones tienen el Q-value máximo (maneja empates)
    max_val = np.max(Q)
    best_actions = np.where(Q == max_val)[0]
    
    # 3. Asignar el (1 - epsilon) de explotación puramente a los ganadores
    pi[best_actions] += (1.0 - epsilon) / len(best_actions)
    
    return pi