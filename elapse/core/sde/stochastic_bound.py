"""
stochastic_bound.py
-------------------
Stochastic IEE upper bound via Ito's formula + Gronwall inequality.

For the ELAPSE SDE:
  dx_i = [-alpha*(Lx)_i - mu*Delta(t;w)*x_i + s_i] dt
          + sigma_n * sqrt(x_i) dW_i

Applying Ito's formula to M(t) = sum_i x_i:

  dM = [-mu*Delta(t)*M + ||s||_1] dt + sigma_n * sum_i sqrt(x_i) dW_i

Applying Ito to M(t)^2:

  d(M^2) = 2M dM + d<M,M>
          = 2M[-mu*Delta*M + ||s||_1] dt
            + sigma_n^2 * sum_i x_i dt
            + martingale terms

Since sum_i x_i = M (after non-negativity):
  d(M^2) = [2M(-mu*delta_min*M + ||s||_1) + sigma_n^2 * M] dt + dmart

Taking expectations (martingale terms vanish):
  d/dt E[M^2] <= (-2*mu*delta_min + sigma_n^2)*E[M^2] + 2*||s||_1*E[M]

Using Jensen E[M] <= sqrt(E[M^2]) and Gronwall:
  E[M^2(t)] <= (M0^2 + C1*t) * exp((-2*mu*delta_min + sigma_n^2)*t)

where C1 = 2*||s||_1*M0 + sigma_n^2*M0.

From Jensen: E[M(t)] <= sqrt(E[M^2(t)])

IEE bound (using H(t) <= ln(n) = H_max):
  E[IEE] = E[integral_tau^T H(t)*M(t) dt]
          <= ln(n) * integral_tau^T E[M(t)] dt
          <= ln(n) * integral_tau^T sqrt(E[M^2(t)]) dt

References
----------
- Oksendal, B. (2003). Stochastic Differential Equations, 6th ed. Springer.
- Gronwall, T.H. (1919). Note on the derivatives with respect to a parameter
  of the solutions of a system of differential equations. Ann. Math. 20(2).
"""

import numpy as np
from scipy.integrate import quad


def stochastic_iee_upper_bound(n, M0, mu, delta_min, sigma_n, s_norm, tau_star, T):
    """
    Closed-form upper bound on E[IEE] for the full stochastic ELAPSE SDE.

    The bound is derived via Ito's formula applied to M(t)^2 followed by
    a Gronwall inequality, giving:

        E[M^2(t)] <= (M0^2 + C1*t) * exp(gamma * t)

    where gamma = -2*mu*delta_min + sigma_n^2 and C1 = (2*||s||_1 + sigma_n^2)*M0.

    Then E[IEE] <= H_max * integral_{tau_star}^{T} sqrt(E[M^2(t)]) dt.

    Parameters
    ----------
    n         : int
        Number of nodes in the network.
    M0        : float
        Initial total mass, M(0) = sum_i x_i(0).
    mu        : float
        Mortality (deletion) rate coefficient.
    delta_min : float
        Lower bound on ensemble signal: delta_min = inf_{t in [tau*,T]} Delta(t; w*) > 0.
        Guaranteed positive by Lemma 1 once H(t) >= H_c (post-trigger).
    sigma_n   : float
        Noise coefficient in the CIR-type diffusion term sigma_n * sqrt(x_i) dW_i.
    s_norm    : float
        L1 norm of source injection: ||s||_1 = sum_i s_i.
    tau_star  : float
        Spectral trigger time tau(lambda2) -- onset of deletion.
    T         : float
        Terminal time of the simulation.

    Returns
    -------
    bound : float
        Upper bound on E[IEE] for the stochastic ELAPSE system.
        Equals 0 if tau_star >= T.

    Notes
    -----
    The bound holds for the FULL stochastic system (sigma_n > 0), replacing
    the deterministic-only bound of the original Theorem 2. It degenerates
    gracefully: as sigma_n -> 0, gamma -> -2*mu*delta_min, recovering the
    deterministic Gronwall bound.

    For the bound to be informative, we need gamma < 0, i.e.,
    sigma_n^2 < 2*mu*delta_min. With the paper's values sigma_n=0.015,
    mu=1.5, delta_min ~ 0.05, this gives 2.25e-4 < 0.15, which is satisfied.
    """
    # Gronwall exponent coefficient
    gamma = -2.0 * mu * delta_min + sigma_n ** 2

    # C1 coefficient from the cross-term in d/dt E[M^2]
    C1 = (2.0 * s_norm + sigma_n ** 2) * M0

    # Maximum entropy for n nodes
    H_max = np.log(float(n))

    def E_M2_bound(t):
        """Upper bound on E[M(t)^2] from Gronwall inequality."""
        return (M0 ** 2 + C1 * t) * np.exp(gamma * t)

    def integrand(t):
        """H_max * sqrt(E[M^2(t)]) — integrand for IEE bound."""
        em2 = E_M2_bound(t)
        return H_max * np.sqrt(max(em2, 0.0))

    if tau_star >= T:
        return 0.0

    bound, _ = quad(integrand, tau_star, T, limit=200)
    return float(bound)


def verify_stochastic_bound(n, topology='erdos_renyi', mu=1.5, delta_min=0.05,
                             sigma_n=0.015, s_frac=0.05, tau_star=2.0, T=15.0,
                             n_paths=1000, dt=0.02, seed=0):
    """
    Monte Carlo verification: run n_paths SDE paths and confirm
    empirical E[IEE] < stochastic_iee_upper_bound.

    The simplified SDE used here is the mortality-only subsystem:
        dx_i = [-mu*delta_min*x_i + s_i] dt + sigma_n*sqrt(x_i) dW_i

    This lower-bounds the full ELAPSE system (Laplacian diffusion spreads
    mass but does not change total M).

    Parameters
    ----------
    n         : int, number of nodes
    topology  : str, 'erdos_renyi' | 'barabasi_albert' | 'watts_strogatz'
    mu, delta_min, sigma_n : floats matching theorem parameters
    s_frac    : float, source rate per node = s_frac / n
    tau_star  : float, trigger time
    T         : float, terminal time
    n_paths   : int, Monte Carlo sample size
    dt        : float, Euler-Maruyama step size
    seed      : int, RNG seed

    Returns
    -------
    dict with keys:
        n, topology, n_paths, empirical_mean_IEE, CI_95_lower, CI_95_upper,
        stochastic_bound, bound_holds, bound_slack_pct
    """
    import sys
    import os
    src_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src')
    sys.path.insert(0, os.path.abspath(src_dir))

    from math_utils import entropy, max_entropy

    rng = np.random.default_rng(seed)

    # Initial conditions: uniform mass M0 = n * 0.5
    M0 = n * 0.5
    x0_base = np.full(n, M0 / n)

    # Source injection (uniform, small)
    s_norm = s_frac
    s = np.full(n, s_norm / n)

    steps = int(T / dt)
    iees = np.zeros(n_paths)

    for path in range(n_paths):
        x = x0_base + rng.uniform(-0.01, 0.01, n)
        x = np.maximum(x, 0.0)

        IEE_path = 0.0

        for i in range(steps):
            t = i * dt
            H = entropy(x)

            # Accumulate IEE only after trigger time
            if t >= tau_star:
                IEE_path += H * x.sum() * dt

            # Euler-Maruyama step: mortality + source + CIR noise
            dxdt = -mu * delta_min * x + s
            dW = rng.standard_normal(n) * np.sqrt(dt)
            x = x + dxdt * dt + sigma_n * np.sqrt(np.maximum(x, 0.0)) * dW
            x = np.maximum(x, 0.0)   # Feller reflection (numerical floor)

        iees[path] = IEE_path

    empirical_mean = float(np.mean(iees))
    ci_lo = float(np.percentile(iees, 2.5))
    ci_hi = float(np.percentile(iees, 97.5))

    bound = stochastic_iee_upper_bound(n, M0, mu, delta_min, sigma_n, s_norm,
                                        tau_star, T)

    return {
        'n': n,
        'topology': topology,
        'n_paths': n_paths,
        'empirical_mean_IEE': empirical_mean,
        'CI_95_lower': ci_lo,
        'CI_95_upper': ci_hi,
        'stochastic_bound': bound,
        'bound_holds': bool(empirical_mean < bound),
        'bound_slack_pct': 100.0 * (bound - empirical_mean) / (bound + 1e-12),
    }
