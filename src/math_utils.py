"""
math_utils.py
-------------
Core mathematical functions shared across all models.
Everything is explained in plain English alongside the formula.
"""

import numpy as np


# ── Entropy ────────────────────────────────────────────────────────────────

def entropy(x):
    """
    Shannon entropy of the data distribution.

    x : array of data concentrations at each node (must be >= 0)

    Normalises x to a probability distribution p, then computes H = -sum(p * log(p)).
    H = 0   -> all data sits on one node (maximally concentrated)
    H = ln(n) -> data perfectly spread across all nodes (maximally exposed)

    Returns 0 if total mass is zero (no data in system).
    """
    total = x.sum()
    if total <= 0:
        return 0.0
    p = x / total
    # Only compute log where p > 0 to avoid log(0)
    mask = p > 1e-12
    return -np.sum(p[mask] * np.log(p[mask]))


def max_entropy(n):
    """Maximum possible entropy for n nodes = ln(n)."""
    return np.log(n)


def normalised_entropy(x):
    """Entropy scaled to [0, 1]. 0 = concentrated, 1 = fully spread."""
    n = len(x)
    H_max = max_entropy(n)
    if H_max == 0:
        return 0.0
    return entropy(x) / H_max


# ── Mortality activation functions ─────────────────────────────────────────

def sigma_egdm(H, H_c, H_max, beta=2.0):
    """
    EGDM mortality activation (M1 baseline).

    Fires when entropy H exceeds critical threshold H_c.
    beta controls sharpness: higher beta = more switch-like.

    Returns value in [0, 1].
    """
    denom = H_max - H_c
    if denom <= 0:
        return 0.0
    val = (H - H_c) / denom
    return float(np.clip(val, 0, None) ** beta)


def sigma_hill(H, H_c, n_hill=4.0):
    """
    Hill function mortality activation (M4 - Biology).

    Borrowed from gene regulatory networks.
    H_c acts as the half-activation point (K in standard Hill notation).
    n_hill is the Hill coefficient - controls steepness of the switch.

    H^n / (H_c^n + H^n)

    Returns value in [0, 1].
    Near 0 when H << H_c, near 1 when H >> H_c.
    """
    if H <= 0:
        return 0.0
    Hn = H ** n_hill
    Kcn = H_c ** n_hill
    return Hn / (Kcn + Hn)


# ── Stochastic helpers ─────────────────────────────────────────────────────

def ou_noise(x, theta=0.1, mu=0.0, sigma=0.05, dt=0.01):
    """
    Ornstein-Uhlenbeck noise increment for one timestep.

    theta: mean reversion speed
    mu:    long-run mean
    sigma: volatility
    dt:    timestep

    Returns noise vector of same shape as x.
    """
    n = len(x)
    dW = np.random.normal(0, np.sqrt(dt), n)
    return theta * (mu - x) * dt + sigma * dW


def first_passage_prob(H, H_c, H_max, t, alpha, lambda2):
    """
    Approximate probability that entropy has crossed H_c by time t.
    Borrowed from barrier-crossing theory in stochastic finance.

    Uses an exponential CDF approximation:
    P(tau <= t) = 1 - exp(-lambda * t)
    where lambda is the expected crossing rate derived from spectral gap.

    This is a simplified closed-form -- the exact first-passage distribution
    for entropy dynamics is an open research problem.
    """
    if H_c >= H_max:
        return 0.0
    # Expected crossing rate proportional to alpha * lambda2 / (H_max - H_c)
    rate = (alpha * lambda2) / max(H_max - H_c, 1e-6)
    return 1.0 - np.exp(-rate * t)
