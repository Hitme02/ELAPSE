"""
m5_social.py
------------
M5: Cascade-Entropy model.
Local neighbourhood cascade deletion + global entropy threshold.

Borrowed from social network cascade theory (Watts 2002, extended).
A node deletes its data when EITHER:
  (a) Global entropy H > H_c  (global signal -- same as EGDM)
  (b) Local cascade: fraction of neighbours who have already deleted > kappa

This captures a realistic social dynamic: even if globally data seems
acceptably contained, a node will delete if it sees most of its neighbours
have already deleted (social conformity / local pressure).

The local cascade vote at node i:
  cascade_i(t) = (number of neighbours with x_j < threshold) / degree(i)

When cascade_i > kappa, node i gets an additional deletion push.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_utils import entropy, max_entropy, sigma_egdm


def local_cascade_pressure(x, A, params):
    """
    For each node i, compute the fraction of its neighbours that have
    already 'deleted' (concentration below deletion_threshold).

    Returns array of shape (n,) with values in [0, 1].
    """
    deletion_threshold = params.get('deletion_threshold', 0.05)
    degrees = A.sum(axis=1)                         # degree of each node
    deleted = (x < deletion_threshold).astype(float)  # 1 if deleted, 0 if not
    neighbour_deleted = A @ deleted                  # count of deleted neighbours
    cascade = np.where(degrees > 0, neighbour_deleted / degrees, 0.0)
    return cascade


def derivatives(x, L, A, params):
    """
    x      : data concentration (n,)
    L      : Laplacian
    A      : adjacency matrix
    params : alpha, mu, H_c, beta, kappa, s
    """
    n     = len(x)
    alpha = params['alpha']
    mu    = params['mu']
    H_c   = params['H_c']
    beta  = params.get('beta', 2.0)
    kappa = params.get('kappa', 0.3)   # cascade threshold: 30% of neighbours deleted
    s     = params.get('s', np.zeros(n))

    H     = entropy(x)
    H_max = max_entropy(n)

    # Global entropy signal (same as EGDM)
    global_sig = sigma_egdm(H, H_c, H_max, beta)

    # Local cascade pressure at each node
    cascade = local_cascade_pressure(x, A, params)
    # Cascade deletion fires when cascade_i > kappa
    local_sig = np.where(cascade > kappa, cascade, 0.0)

    # Combined: take max of global and local signal per node
    combined_sig = np.maximum(global_sig, local_sig)

    diffusion = -alpha * L @ x
    mortality = -mu * combined_sig * x
    injection = s

    return diffusion + mortality + injection


def vote(x, A, params, t, lambda2):
    """
    Social cascade vote: mean of (global entropy signal, mean local cascade).
    Returns weighted combination in [0, 1].
    """
    n     = len(x)
    H     = entropy(x)
    H_max = max_entropy(n)
    H_c   = params['H_c']
    beta  = params.get('beta', 2.0)
    kappa = params.get('kappa', 0.3)

    global_v  = sigma_egdm(H, H_c, H_max, beta)
    cascade   = local_cascade_pressure(x, A, params)
    local_v   = float(np.mean(np.where(cascade > kappa, cascade, 0.0)))

    return 0.5 * global_v + 0.5 * local_v


def simulate(x0, L, A, params, T=20.0, dt=0.01, stochastic=True):
    """
    x0 : initial data concentration (n,)
    L  : Laplacian
    A  : adjacency matrix
    """
    n     = len(x0)
    steps = int(T / dt)
    x     = x0.copy()

    t_arr = np.zeros(steps + 1)
    x_arr = np.zeros((steps + 1, n))
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)

    x_arr[0] = x
    H_arr[0] = entropy(x)
    M_arr[0] = x.sum()

    for i in range(steps):
        dxdt = derivatives(x, L, A, params)
        x    = x + dxdt * dt

        if stochastic:
            sigma_n = params.get('sigma_noise', 0.02)
            noise   = sigma_n * np.sqrt(np.maximum(x, 0)) * np.random.normal(0, np.sqrt(dt), n)
            x       = x + noise

        x = np.maximum(x, 0)

        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = x
        H_arr[i + 1] = entropy(x)
        M_arr[i + 1] = x.sum()

    return t_arr, x_arr, H_arr, M_arr
