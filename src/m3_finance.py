"""
m3_finance.py
-------------
M3: Stochastic-Finance model.
Ornstein-Uhlenbeck (OU) mean-reverting diffusion + first-passage time mortality.

The OU process models data concentration drifting toward a natural equilibrium,
with stochastic fluctuations representing noisy network conditions.

dx_i = alpha*(mu_i - x_i)*dt + sigma*dW_i   [OU diffusion]
      - mu_mort * P(tau <= t) * x_i * dt     [first-passage mortality]
      + s_i * dt                             [injection]

Where P(tau <= t) is the probability that entropy has already crossed H_c
-- borrowed from barrier option theory in mathematical finance.

This replaces the sharp σ(H) threshold with a smooth probabilistic vote
that grows over time as crossing becomes increasingly likely.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_utils import entropy, max_entropy, first_passage_prob


def derivatives(x, params, t, lambda2):
    """
    x      : current data concentration (n,)
    params : theta (OU reversion speed), mu_ou (OU mean), sigma_ou,
             mu_mort (mortality rate), H_c, alpha, s
    t      : current time
    lambda2: Fiedler value of network (used in first-passage rate)
    """
    n         = len(x)
    theta     = params.get('theta_ou', 0.5)
    mu_ou     = params.get('mu_ou', np.zeros(n) if not isinstance(params.get('mu_ou'), np.ndarray) else params.get('mu_ou'))
    mu_mort   = params['mu']
    H_c       = params['H_c']
    alpha     = params['alpha']
    s         = params.get('s', np.zeros(n))

    if not isinstance(mu_ou, np.ndarray):
        mu_ou = np.full(n, mu_ou)

    H     = entropy(x)
    H_max = max_entropy(n)

    # First-passage probability (smooth probabilistic mortality)
    fp_prob = first_passage_prob(H, H_c, H_max, t, alpha, lambda2)

    # OU mean reversion (deterministic part)
    ou_drift  = theta * (mu_ou - x)

    # Mortality scaled by first-passage probability
    mortality = -mu_mort * fp_prob * x

    return ou_drift + mortality + s


def vote(x, params, t, lambda2):
    """
    Finance vote: first-passage probability P(tau <= t).
    Grows from 0 toward 1 as time passes and crossing becomes more likely.
    Value in [0, 1].
    """
    n     = len(x)
    H     = entropy(x)
    H_max = max_entropy(n)
    H_c   = params['H_c']
    alpha = params['alpha']
    return first_passage_prob(H, H_c, H_max, t, alpha, lambda2)


def simulate(x0, L, params, lambda2, T=20.0, dt=0.01, stochastic=True):
    """
    x0      : initial data concentration (n,)
    L       : Laplacian (used to compute lambda2 if not provided)
    params  : model parameters
    lambda2 : Fiedler value
    """
    n     = len(x0)
    steps = int(T / dt)
    x     = x0.copy()

    sigma_ou = params.get('sigma_ou', 0.05)

    t_arr = np.zeros(steps + 1)
    x_arr = np.zeros((steps + 1, n))
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)

    x_arr[0] = x
    H_arr[0] = entropy(x)
    M_arr[0] = x.sum()

    for i in range(steps):
        t_curr = i * dt
        dxdt   = derivatives(x, params, t_curr, lambda2)
        x      = x + dxdt * dt

        if stochastic:
            # OU stochastic term
            dW = np.random.normal(0, np.sqrt(dt), n)
            x  = x + sigma_ou * dW

        x = np.maximum(x, 0)

        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = x
        H_arr[i + 1] = entropy(x)
        M_arr[i + 1] = x.sum()

    return t_arr, x_arr, H_arr, M_arr
