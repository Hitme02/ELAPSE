"""
m6_ensemble.py
--------------
M6: The ELAPSE Ensemble Voting Framework.

Each of the five mechanisms casts a vote between 0 and 1:
  v1 : EGDM entropy signal
  v2 : Epidemic infected fraction
  v3 : Finance first-passage probability
  v4 : Biology Hill function
  v5 : Social cascade pressure

The ensemble fires mortality when the weighted vote exceeds threshold theta:
  Delta(t) = w1*v1 + w2*v2 + w3*v3 + w4*v4 + w5*v5 >= theta

Weights are LEARNED via Nelder-Mead (scipy.optimize.minimize) minimising
time-integrated entropy exposure (IEE) with an ENTROPY REGULARISATION term:

  loss = IEE + lambda_reg * (-sum(w_k * log(w_k)))    [regularised objective]

The negative entropy term -sum(w*log(w)) is maximised at uniform weights,
so adding it to the loss penalises weight collapse to one-hot solutions.
lambda_reg controls the trade-off (default 5.0).
"""

import numpy as np
from scipy.optimize import minimize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_utils import entropy, max_entropy
import m1_egdm    as m1
import m2_epidemic as m2
import m3_finance  as m3
import m4_biology  as m4
import m5_social   as m5


# ── Default equal weights (overridden after learning) ─────────────────────
DEFAULT_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
DEFAULT_THETA   = 0.4   # majority-ish threshold


def collect_votes(x, state_m2, L, A, params, t, lambda2):
    """
    Collect votes from all five mechanisms at current timestep.
    Returns array of 5 votes, each in [0, 1].
    """
    v1 = m1.vote(x, L, params, t, lambda2)
    v2 = m2.vote(state_m2, A, params, t, lambda2)
    v3 = m3.vote(x, params, t, lambda2)
    v4 = m4.vote(x, L, params, t, lambda2)
    v5 = m5.vote(x, A, params, t, lambda2)
    return np.array([v1, v2, v3, v4, v5])


def ensemble_signal(votes, weights, theta=DEFAULT_THETA):
    """
    Compute weighted ensemble vote and return:
      - Delta: weighted sum (scalar)
      - fires: bool, whether ensemble triggers deletion
      - effective_sig: continuous signal for smooth mortality [0,1]
    """
    Delta       = float(np.dot(weights, votes))
    fires       = Delta >= theta
    # Smooth signal: how far above threshold (clipped to [0,1])
    eff_sig     = np.clip((Delta - theta) / (1.0 - theta + 1e-6), 0, 1) if fires else 0.0
    return Delta, fires, eff_sig


def simulate(x0, L, A, params, lambda2, weights=None, theta=DEFAULT_THETA,
             T=20.0, dt=0.01, stochastic=True):
    """
    Run the full ensemble model forward in time.

    x0      : initial data concentration (n,)
    L       : Laplacian
    A       : adjacency matrix
    params  : shared model parameters
    lambda2 : Fiedler value
    weights : learned vote weights (5,); defaults to equal
    theta   : voting threshold
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    n     = len(x0)
    steps = int(T / dt)
    x     = x0.copy()

    # M2 needs its own state (S, I, R)
    I0     = x0.copy()
    S0     = np.maximum(np.ones(n) - I0, 0)
    R0     = np.zeros(n)
    state2 = np.concatenate([S0, I0, R0])

    t_arr     = np.zeros(steps + 1)
    x_arr     = np.zeros((steps + 1, n))
    H_arr     = np.zeros(steps + 1)
    M_arr     = np.zeros(steps + 1)
    Delta_arr = np.zeros(steps + 1)
    votes_arr = np.zeros((steps + 1, 5))

    x_arr[0]    = x
    H_arr[0]    = entropy(x)
    M_arr[0]    = x.sum()
    votes0      = collect_votes(x, state2, L, A, params, 0.0, lambda2)
    Delta_arr[0], _, _ = ensemble_signal(votes0, weights, theta)
    votes_arr[0]= votes0

    for i in range(steps):
        t_curr = i * dt

        # ── Collect votes ──────────────────────────────────────────
        votes = collect_votes(x, state2, L, A, params, t_curr, lambda2)
        Delta, fires, eff_sig = ensemble_signal(votes, weights, theta)

        # ── Compute per-mechanism derivatives ──────────────────────
        # Base diffusion (from M1)
        diffusion = -params['alpha'] * L @ x

        # Ensemble mortality: fires only when Delta >= theta
        mortality = -params['mu'] * eff_sig * x

        # Injection
        s = params.get('s', np.zeros(n))

        dxdt = diffusion + mortality + s

        # ── Update M2 state (SIR) ──────────────────────────────────
        dstate2 = m2.derivatives(state2, A, params)
        state2  = state2 + dstate2 * dt
        state2  = np.maximum(state2, 0)

        # ── Euler step ─────────────────────────────────────────────
        x = x + dxdt * dt

        if stochastic:
            sigma_n = params.get('sigma_noise', 0.02)
            noise   = sigma_n * np.sqrt(np.maximum(x, 0)) * np.random.normal(0, np.sqrt(dt), n)
            x       = x + noise

        x = np.maximum(x, 0)

        t_arr[i + 1]     = (i + 1) * dt
        x_arr[i + 1]     = x
        H_arr[i + 1]     = entropy(x)
        M_arr[i + 1]     = x.sum()
        Delta_arr[i + 1] = Delta
        votes_arr[i + 1] = votes

    return t_arr, x_arr, H_arr, M_arr, Delta_arr, votes_arr


# ── Weight Learning ────────────────────────────────────────────────────────

def objective(weights_raw, training_data, theta=DEFAULT_THETA, lambda_reg=5.0):
    """
    Regularised objective function for weight learning.

    Minimise: IEE + lambda_reg * weight_entropy_penalty
    where weight_entropy_penalty = -sum(w_k * log(w_k))
    (i.e. we ADD the negative entropy, which is MINIMISED at the uniform
     distribution, so the gradient pushes weights AWAY from 0/1 extremes)

    weights_raw : unconstrained weights (softmax-normalised internally)
    training_data : list of (x0, L, A, params, lambda2) tuples
    lambda_reg  : regularisation strength (higher = more uniform weights)
    """
    # Softmax to ensure weights sum to 1 and are positive
    w_exp = np.exp(weights_raw - weights_raw.max())   # stabilised softmax
    w = w_exp / w_exp.sum()

    total_exposure = 0.0

    for (x0, L, A, params, lambda2) in training_data:
        _, _, H_arr, M_arr, _, _ = simulate(
            x0, L, A, params, lambda2,
            weights=w, theta=theta,
            T=params.get('T_train', 10.0),
            dt=params.get('dt', 0.05),
            stochastic=False   # deterministic for stable optimisation
        )
        dt_val   = params.get('dt', 0.05)
        exposure = float(np.sum(H_arr * M_arr) * dt_val)
        total_exposure += exposure

    # Entropy regularisation: penalise weight collapse.
    # weight_entropy = -sum(w * log(w)):
    #   = log(5) ≈ 1.61 at uniform distribution (maximum diversity)
    #   = 0              at one-hot (total collapse)
    # We SUBTRACT lambda_reg * weight_entropy from the loss, so the optimiser
    # is penalised for LOW entropy (= collapse) and rewarded for HIGH entropy
    # (= diversity).  loss = IEE - lambda_reg * weight_entropy
    weight_entropy = -float(np.sum(w * np.log(w + 1e-12)))
    regularisation = -lambda_reg * weight_entropy   # NOTE: negative sign

    return total_exposure + regularisation


def learn_weights(training_data, theta=DEFAULT_THETA, lambda_reg=15.0,
                  n_restarts=5, verbose=True):
    """
    Learn optimal weights via Nelder-Mead with entropy regularisation.

    training_data : list of (x0, L, A, params, lambda2) tuples
    theta         : voting threshold
    lambda_reg    : regularisation strength (default 5.0 balances IEE scale)
    n_restarts    : number of random restarts (takes best result)

    Returns:
        weights : optimised weight vector (5,) summing to 1
        history : list of (final_loss, weights) per restart
    """
    best_loss    = np.inf
    best_weights = DEFAULT_WEIGHTS.copy()
    history      = []

    for restart in range(n_restarts):
        if restart == 0:
            w0 = np.zeros(5)          # equal weights initialisation
        else:
            w0 = np.random.randn(5) * 0.5   # random restarts

        result = minimize(
            objective,
            w0,
            args=(training_data, theta, lambda_reg),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3}
        )

        # Softmax to recover actual weights (stabilised)
        w_raw   = result.x
        w_exp   = np.exp(w_raw - w_raw.max())
        w_final = w_exp / w_exp.sum()
        loss    = result.fun
        history.append((loss, w_final))

        if verbose:
            print(f"  Restart {restart+1}: loss={loss:.4f} | weights={np.round(w_final, 3)}")

        if loss < best_loss:
            best_loss    = loss
            best_weights = w_final

    if verbose:
        print(f"\nBest weights: {np.round(best_weights, 3)}")
        print(f"Best loss:    {best_loss:.4f}")
        # Check for collapse: max weight > 0.8 is a warning sign
        if best_weights.max() > 0.8:
            print(f"  ⚠  Weight concentration detected (max={best_weights.max():.3f}). "
                  f"Consider increasing lambda_reg.")
        else:
            print(f"  ✓  Weights are distributed (max={best_weights.max():.3f}).")

    return best_weights, history
