"""
m4_biology.py
-------------
M4: Bio-Switch model.
Graph-Laplacian diffusion + Hill function mortality (from gene regulatory networks).

The Hill function replaces the linear σ(H) from EGDM with a sigmoidal switch:

    σ_hill(H) = H^n / (H_c^n + H^n)

This is well-studied in systems biology as a model of cooperative switching.
Key properties:
  - Near 0 when H << H_c (data safe, mortality suppressed)
  - = 0.5 exactly when H = H_c (half-activation)
  - Near 1 when H >> H_c (data exposed, full mortality)
  - Steepness controlled by Hill coefficient n_hill
    n_hill = 1 : gradual (Michaelis-Menten kinetics)
    n_hill = 4 : cooperative, switch-like
    n_hill > 10: nearly binary step function

The mathematical advantage over EGDM's σ: the Hill function is smoother,
biologically validated, and its stability properties are well-understood
from decades of gene network analysis.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_utils import entropy, max_entropy, sigma_hill


def derivatives(x, L, params):
    """
    x      : current data concentration (n,)
    L      : graph Laplacian (n, n)
    params : alpha, mu, H_c, n_hill, s
    """
    n       = len(x)
    alpha   = params['alpha']
    mu      = params['mu']
    H_c     = params['H_c']
    n_hill  = params.get('n_hill', 4.0)
    s       = params.get('s', np.zeros(n))

    H   = entropy(x)
    sig = sigma_hill(H, H_c, n_hill)

    diffusion = -alpha * L @ x
    mortality = -mu * sig * x
    injection = s

    return diffusion + mortality + injection


def vote(x, L, params, t, lambda2):
    """
    Biology vote: Hill function evaluated at current entropy.
    Switch-like: near 0 below H_c, near 1 above H_c.
    Value in [0, 1].
    """
    H      = entropy(x)
    H_c    = params['H_c']
    n_hill = params.get('n_hill', 4.0)
    return sigma_hill(H, H_c, n_hill)


def simulate(x0, L, params, T=20.0, dt=0.01, stochastic=True):
    """
    x0 : initial data concentration (n,)
    L  : Laplacian
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
