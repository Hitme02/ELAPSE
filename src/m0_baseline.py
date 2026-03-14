"""
m0_baseline.py
--------------
M0: No-deletion baseline — pure Laplacian diffusion, zero mortality.

This is the "worst case" scenario: data spreads across the network forever
with no self-erasure mechanism of any kind.

Its IEE should be the HIGHEST of all models, demonstrating why autonomous
deletion is necessary. Without it, entropy exposure is unbounded.

dx_i/dt = -alpha * sum_j L_ij * x_j   [diffusion only]
         + s_i(t)                       [injection]
         + noise                        [stochastic]
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_utils import entropy


def derivatives(x, L, params):
    """
    Pure diffusion: no mortality term.

    x      : data concentration (n,)
    L      : graph Laplacian (n,n)
    params : alpha, s
    """
    n = len(x)
    alpha = params['alpha']
    s     = params.get('s', np.zeros(n))

    diffusion = -alpha * L @ x
    return diffusion + s


def vote(x, L, params, t, lambda2):
    """
    M0 never votes to delete. Returns 0 always.
    Included so it can participate in the ensemble signature if needed.
    """
    return 0.0


def simulate(x0, L, params, T=20.0, dt=0.01, stochastic=True):
    """
    Run the no-deletion baseline forward in time.

    Returns:
        t_arr : time steps
        x_arr : (steps+1, n) concentration trajectory
        H_arr : entropy trajectory
        M_arr : total mass trajectory
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
        dxdt = derivatives(x, L, params)
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
