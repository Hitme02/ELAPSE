"""
m2_epidemic.py
--------------
M2: Epidemic-Entropy model.
SIR compartmental dynamics replace pure diffusion.

Each node i has three compartments:
  S_i : susceptible (doesn't have the data yet)
  I_i : infected    (has the data, actively spreading it)
  R_i : recovered   (deleted the data)

Data "concentration" at node i = I_i (infected fraction).

Spreading:  dI_i/dt += beta_sir * S_i * sum_j A_ij * I_j  (neighbour infection)
Recovery:   dI_i/dt -= gamma * I_i                          (spontaneous deletion)
Entropy:    mortality fires when H(I / sum(I)) > H_c        (same entropy trigger)

The SIR layer replaces the diffusion term from M1.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_utils import entropy, max_entropy, sigma_egdm, ou_noise


def derivatives(state, A, params):
    """
    state : (3n,) vector -- first n are S, next n are I, last n are R
    A     : adjacency matrix (n, n)
    params: beta_sir, gamma, mu, H_c, beta, s
    """
    n         = A.shape[0]
    S         = state[:n]
    I         = state[n:2*n]
    R         = state[2*n:]

    beta_sir  = params.get('beta_sir', 0.3)
    gamma     = params.get('gamma', 0.1)
    mu        = params['mu']
    H_c       = params['H_c']
    beta      = params.get('beta', 2.0)
    s         = params.get('s', np.zeros(n))

    # Force of infection at each node from infected neighbours
    lambda_i  = beta_sir * (A @ I)

    # SIR transitions
    new_infections = S * lambda_i
    new_recoveries = gamma * I

    # Entropy-triggered extra mortality on infected nodes
    H     = entropy(I)
    H_max = max_entropy(n)
    sig   = sigma_egdm(H, H_c, H_max, beta)
    entropy_mortality = mu * sig * I

    dS = -new_infections + s                 # injection refills susceptible pool
    dI =  new_infections - new_recoveries - entropy_mortality
    dR =  new_recoveries + entropy_mortality

    return np.concatenate([dS, dI, dR])


def vote(state, A, params, t, lambda2):
    """
    Epidemic vote: fraction of nodes currently infected.
    High when data has spread widely. Value in [0, 1].
    """
    n = A.shape[0]
    I = state[n:2*n]
    return float(I.sum() / max(n, 1))


def simulate(x0, A, L, params, T=20.0, dt=0.01, stochastic=True):
    """
    x0 : initial infected concentration (n,) -- treated as I_0
    A  : adjacency matrix
    L  : Laplacian (not used directly but passed for consistency)
    """
    n     = len(x0)
    steps = int(T / dt)

    # Initialise compartments
    I0 = x0.copy()
    S0 = np.ones(n) - I0
    S0 = np.maximum(S0, 0)
    R0 = np.zeros(n)
    state = np.concatenate([S0, I0, R0])

    t_arr = np.zeros(steps + 1)
    x_arr = np.zeros((steps + 1, n))   # store I(t) as "data concentration"
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)

    x_arr[0] = I0
    H_arr[0] = entropy(I0)
    M_arr[0] = I0.sum()

    for i in range(steps):
        dstate = derivatives(state, A, params)
        state  = state + dstate * dt

        if stochastic:
            sigma_n = params.get('sigma_noise', 0.02)
            I_curr  = state[n:2*n]
            noise   = sigma_n * np.sqrt(np.maximum(I_curr, 0)) * np.random.normal(0, np.sqrt(dt), n)
            state[n:2*n] += noise

        state = np.maximum(state, 0)

        # Renormalise so S+I+R = 1 per node
        total_per_node = state[:n] + state[n:2*n] + state[2*n:]
        total_per_node = np.maximum(total_per_node, 1e-12)
        state[:n]    /= total_per_node
        state[n:2*n] /= total_per_node
        state[2*n:]  /= total_per_node

        I = state[n:2*n]
        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = I
        H_arr[i + 1] = entropy(I)
        M_arr[i + 1] = I.sum()

    return t_arr, x_arr, H_arr, M_arr
