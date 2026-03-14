"""
m1_egdm.py
----------
M1: The original EGDM model (baseline).
Graph-Laplacian diffusion + entropy-triggered mortality.

dx_i/dt = alpha * sum_j w_ij (x_j - x_i)   [diffusion]
         - mu * sigma(H(p)) * x_i            [mortality]
         + s_i(t)                             [injection]
         + noise                              [stochastic]
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_utils import entropy, max_entropy, sigma_egdm, ou_noise


def derivatives(x, L, params):
    """
    Compute dx/dt for the EGDM system.

    x      : current data concentration vector (n,)
    L      : graph Laplacian matrix (n, n)
    params : dict with keys alpha, mu, H_c, beta, s (injection vector)
    """
    n = len(x)
    alpha  = params['alpha']
    mu     = params['mu']
    H_c    = params['H_c']
    beta   = params.get('beta', 2.0)
    s      = params.get('s', np.zeros(n))

    H     = entropy(x)
    H_max = max_entropy(n)
    sig   = sigma_egdm(H, H_c, H_max, beta)

    diffusion = -alpha * L @ x
    mortality = -mu * sig * x
    injection = s

    return diffusion + mortality + injection


def vote(x, L, params, t, lambda2):
    """
    EGDM vote for the ensemble.
    Returns sigma(H) -- how strongly EGDM thinks deletion should happen.
    Value in [0, 1].
    """
    n = len(x)
    H     = entropy(x)
    H_max = max_entropy(n)
    H_c   = params['H_c']
    beta  = params.get('beta', 2.0)
    return sigma_egdm(H, H_c, H_max, beta)


def simulate(x0, L, params, T=20.0, dt=0.01, stochastic=True):
    """
    Simulate M1 forward in time using Euler-Maruyama integration.

    x0         : initial data concentration (n,)
    L          : Laplacian
    params     : model parameters
    T          : total simulation time
    dt         : timestep
    stochastic : whether to add OU noise

    Returns:
        t_arr  : time array
        x_arr  : (timesteps, n) state trajectory
        H_arr  : entropy trajectory
        M_arr  : total mass trajectory
    """
    n      = len(x0)
    steps  = int(T / dt)
    x      = x0.copy()

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

        # Stochastic term: CIR-type sqrt(x) noise
        if stochastic:
            noise_scale = params.get('sigma_noise', 0.02)
            noise = noise_scale * np.sqrt(np.maximum(x, 0)) * np.random.normal(0, np.sqrt(dt), n)
            x = x + noise

        # Keep non-negative
        x = np.maximum(x, 0)

        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = x
        H_arr[i + 1] = entropy(x)
        M_arr[i + 1] = x.sum()

    return t_arr, x_arr, H_arr, M_arr
