"""
m7_timer.py
-----------
M7: Vanish-style chronological fixed-timer deletion baseline.

The timer fires at a wall-clock time t_threshold, at which point a global
mortality pulse is applied.  This is the architectural assumption underlying
Vanish (DHT key expiry after 8 hours) and FADE (policy-based key deletion).

Unlike ELAPSE's entropy clock, the timer is blind to the actual spatial
spread of data.  We calibrate t_threshold to match the mean deletion time
of the ELAPSE ensemble (M6) at each (topology, n) combination, so that M7
receives equal 'deletion budget' as M6 — making IEE comparisons fair.

Mechanism:
  dxi/dt = -alpha * L @ x + s(t)    [diffusion, same as M0]
  At t = t_threshold: apply x <- x * exp(-mu * pulse_rate * dt_pulse)
                       until mass drops below deletion_threshold * M0.

For a fair comparison we use a single instantaneous erasure:
  x(t_threshold+) = 0  (all data deleted at the timer)

This is equivalent to Vanish's DHT shard erasure — once the key expires,
all copies are irrecoverable.  The IEE accumulated before t_threshold is
the full exposure cost.

IEE_M7 = integral_0^{t_threshold} H(t) M(t) dt
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_utils import entropy


def simulate(x0, L, params, t_threshold=7.5, T=20.0, dt=0.01,
             stochastic=True):
    """
    Run the fixed-timer baseline.

    At time t_threshold, all data is instantaneously erased (x -> 0),
    modelling Vanish-style DHT key deletion.  The simulation continues
    to T so that the IEE integral matches the evaluation horizon.

    Parameters
    ----------
    x0          : initial data concentration (n,)
    L           : graph Laplacian (n,n)
    params      : dict with 'alpha', 's', 'sigma_noise'
    t_threshold : wall-clock time of deletion (default T/2 = 7.5)
    T           : simulation horizon
    dt          : time step
    stochastic  : whether to add CIR-type noise (same as other mechanisms)

    Returns
    -------
    t_arr, x_arr, H_arr, M_arr  (same interface as all other mechanisms)
    """
    n     = len(x0)
    steps = int(T / dt)
    alpha = params['alpha']
    s     = params.get('s', np.zeros(n))
    sigma_n = params.get('sigma_noise', 0.02)

    x = x0.copy()
    fired = False

    t_arr = np.zeros(steps + 1)
    x_arr = np.zeros((steps + 1, n))
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)

    x_arr[0] = x
    H_arr[0] = entropy(x)
    M_arr[0] = x.sum()

    for i in range(steps):
        t = i * dt

        # Timer check: erase all data at t_threshold
        if not fired and t >= t_threshold:
            x     = np.zeros(n)
            fired = True

        if not fired:
            # Laplacian diffusion + source (same as M0)
            dxdt = -alpha * L @ x + s
            x    = x + dxdt * dt

            if stochastic:
                noise = (sigma_n * np.sqrt(np.maximum(x, 0))
                         * np.random.normal(0, np.sqrt(dt), n))
                x = x + noise

            x = np.maximum(x, 0)

        t_arr[i + 1] = (i + 1) * dt
        x_arr[i + 1] = x
        H_arr[i + 1] = entropy(x)
        M_arr[i + 1] = x.sum()

    return t_arr, x_arr, H_arr, M_arr


def calibrate_threshold(t_star_m6_mean, T=15.0):
    """
    Set M7 timer to match the mean M6 trigger time, for a fair IEE comparison.

    Parameters
    ----------
    t_star_m6_mean : mean deletion time from M6 on the same (topology, n)
    T              : simulation horizon

    Returns
    -------
    t_threshold : float — timer to use for M7
    """
    # Clamp to (0, T) with a small margin
    return float(np.clip(t_star_m6_mean, 0.5, T - 0.5))
