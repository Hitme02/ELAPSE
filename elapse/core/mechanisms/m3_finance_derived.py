"""
m3_finance_derived.py
---------------------
M3 (derived): Time-varying effective spectral gap from the spectral entropy
growth equation. This is the PRIMARY formulation of M3 in the revised paper.

DERIVATION
----------
From Corollary 1, the normalised distribution p(t) = x(t)/M(t) satisfies:

    ||p(t) - 1/n||^2 ~ C0 * exp(-2*alpha*lambda2*t)

Differentiating the Shannon entropy H(t) = -sum_i p_i log p_i and using the
spectral decay, the instantaneous entropy growth rate obeys:

    dH/dt ~ 2*alpha*lambda2*(H_max - H(t))

This is a first-order linear ODE whose solution is:

    H(t) = H_max - (H_max - H(0)) * exp(-2*alpha*lambda2*t)

The interpretation: the effective rate at which entropy approaches H_max
at the current state H(t) is proportional to the remaining headroom
(H_max - H(t)).  Defining the effective spectral gap as the coefficient
of the remaining headroom normalised to the headroom at trigger onset:

    lambda_eff(t) = alpha * lambda2 * (H_max - H(t)) / (H_max - H_c)

This recovers a dimensionally correct, state-dependent replacement for the
static lambda2 in the original M3 first-passage formula.

Comparison with Appendix B heuristic
--------------------------------------
Appendix B proposes:
    lambda_eff_appB(t) = lambda2 * (H(t)/H_max) * (M(t)/M0)

The derived version here is MORE PRINCIPLED because:
1. It follows directly from the spectral entropy growth equation (not heuristic).
2. It scales with (H_max - H) (remaining headroom), not H (entropy already reached).
3. It has natural limits: lambda_eff -> 0 as H -> H_max (no room left to spread),
   and lambda_eff -> alpha*lambda2 near H_c (full spectral rate at trigger onset).
4. It does not depend on M(t)/M0, which conflates spreading with deletion.

Usage
-----
    from elapse.core.mechanisms.m3_finance_derived import vote, simulate

    v3 = vote(x, params, t, lambda2)          # derived M3 vote in [0,1]
    t_arr, x_arr, H_arr, M_arr = simulate(x0, L, params, lambda2, T=15.0)
"""

import numpy as np
import os
import sys

# Support both direct execution and package import
_this_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.join(_this_dir, '..', '..', '..', 'src')
sys.path.insert(0, os.path.abspath(_src_dir))

from math_utils import entropy, max_entropy


# ── Core derived functions ──────────────────────────────────────────────────

def lambda_eff_derived(H, H_c, H_max, alpha, lambda2):
    """
    Derived time-varying effective spectral gap.

    lambda_eff(t) = alpha * lambda2 * (H_max - H(t)) / (H_max - H_c)

    Derivation: the instantaneous entropy growth rate from spectral decay is
        dH/dt = 2*alpha*lambda2*(H_max - H)
    The effective first-passage rate at the current entropy H is proportional
    to (H_max - H), normalised to its value at trigger onset (H_max - H_c).

    Parameters
    ----------
    H      : float, current entropy H(t)
    H_c    : float, critical threshold entropy
    H_max  : float, maximum entropy ln(n)
    alpha  : float, Laplacian diffusion coefficient
    lambda2: float, static Fiedler value of the network

    Returns
    -------
    float, effective spectral gap >= 0
    """
    denom  = max(H_max - H_c, 1e-6)
    h_room = max(H_max - H, 0.0)
    return float(alpha * lambda2 * h_room / denom)


def first_passage_prob_derived(H, H_c, H_max, t, alpha, lambda2):
    """
    First-passage probability using the derived state-dependent lambda_eff.

    P(tau <= t) = 1 - exp(-lambda_eff(t) * t)

    where lambda_eff = alpha * lambda2 * (H_max - H) / (H_max - H_c).

    When H is close to H_max (near full spread), lambda_eff -> 0 and the
    deletion probability saturates more slowly, correctly capturing the
    diminishing marginal rate of new crossings once near-full spreading
    has already occurred.

    Returns float in [0, 1].
    """
    if H_c >= H_max:
        return 0.0
    lam_eff = lambda_eff_derived(H, H_c, H_max, alpha, lambda2)
    return float(1.0 - np.exp(-lam_eff * t))


# ── Vote function ────────────────────────────────────────────────────────────

def vote(x, params, t, lambda2):
    """
    Derived M3 vote: first-passage probability with state-dependent lambda_eff.

    This is the PRIMARY M3 formulation in the revised ELAPSE paper.

    Parameters
    ----------
    x       : array (n,), current data concentration
    params  : dict with keys 'H_c', 'alpha'
    t       : float, current time
    lambda2 : float, Fiedler value

    Returns
    -------
    float in [0, 1]
    """
    n     = len(x)
    H     = entropy(x)
    H_max = max_entropy(n)
    H_c   = params['H_c']
    alpha = params['alpha']
    return first_passage_prob_derived(H, H_c, H_max, t, alpha, lambda2)


# ── Derivatives ──────────────────────────────────────────────────────────────

def derivatives(x, params, t, lambda2):
    """
    SDE drift for derived M3.

    Combines Laplacian diffusion with derived first-passage mortality:
        drift_i = -mu * P(tau<=t; derived) * x_i + s_i

    (The Laplacian term is handled by the simulate function separately
    to keep this consistent with other mechanisms.)

    Parameters
    ----------
    x       : array (n,)
    params  : dict with 'mu', 'H_c', 'alpha', 's'
    t       : float
    lambda2 : float

    Returns
    -------
    array (n,) drift contribution (excluding Laplacian term)
    """
    n     = len(x)
    mu    = params['mu']
    s     = params.get('s', np.zeros(n))

    H     = entropy(x)
    H_max = max_entropy(n)
    H_c   = params['H_c']
    alpha = params['alpha']

    fp_prob   = first_passage_prob_derived(H, H_c, H_max, t, alpha, lambda2)
    mortality = -mu * fp_prob * x

    return mortality + s


# ── Simulation ───────────────────────────────────────────────────────────────

def simulate(x0, L, params, lambda2, T=15.0, dt=0.02, stochastic=True):
    """
    Simulate derived M3 via Euler-Maruyama.

    Full SDE:
        dx_i = [-alpha*(Lx)_i + derivatives_i(x,t)] dt
               + sigma_n * sqrt(max(x_i,0)) * dW_i

    where derivatives includes mortality from derived first-passage probability.

    Parameters
    ----------
    x0      : array (n,), initial concentration
    L       : array (n,n), graph Laplacian
    params  : dict, model parameters
    lambda2 : float, Fiedler value
    T       : float, terminal time
    dt      : float, step size
    stochastic : bool, include CIR noise

    Returns
    -------
    t_arr  : array (steps+1,)
    x_arr  : array (steps+1, n)
    H_arr  : array (steps+1,)
    M_arr  : array (steps+1,)
    """
    n       = len(x0)
    steps   = int(T / dt)
    alpha   = params['alpha']
    sigma_n = params.get('sigma_noise', 0.015)

    x = x0.copy()

    t_arr = np.zeros(steps + 1)
    x_arr = np.zeros((steps + 1, n))
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)

    x_arr[0] = x
    H_arr[0] = entropy(x)
    M_arr[0] = x.sum()

    for i in range(steps):
        t_curr = i * dt

        # Combined drift: Laplacian diffusion + derived first-passage mortality + source
        dxdt = -alpha * (L @ x) + derivatives(x, params, t_curr, lambda2)
        x    = x + dxdt * dt

        if stochastic:
            dW = np.random.normal(0, np.sqrt(dt), n)
            x  = x + sigma_n * np.sqrt(np.maximum(x, 0.0)) * dW

        x = np.maximum(x, 0.0)

        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = x
        H_arr[i + 1] = entropy(x)
        M_arr[i + 1] = x.sum()

    return t_arr, x_arr, H_arr, M_arr
