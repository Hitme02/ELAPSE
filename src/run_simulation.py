"""
run_simulation.py
-----------------
Main simulation runner for the ELAPSE framework.

Runs all 7 models (M0-M6) across three network topologies and
multiple network sizes, records key metrics, and saves results.

Also runs:
  - run_sensitivity(): H_c × beta sensitivity analysis

Metrics recorded per run:
  - H_arr       : entropy trajectory H(p(t))
  - M_arr       : total mass trajectory M(t)
  - t_star      : mortality activation time (first time M drops below 50% of M0)
  - IEE         : time-integrated entropy exposure = sum(H(t)*M(t)*dt)
  - final_mass  : M(T) -- how much data remains at end
"""

import numpy as np
import pickle, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from networks   import get_all_networks, fiedler_value
from math_utils import max_entropy

import m0_baseline as m0
import m1_egdm     as m1
import m2_epidemic as m2
import m3_finance  as m3
import m4_biology  as m4
import m5_social   as m5
from m6_ensemble import simulate as sim_m6, learn_weights


# ── Simulation parameters ─────────────────────────────────────────────────

SIZES     = [50, 100, 200, 500]   # network sizes for main scaling study
T         = 15.0                  # simulation duration
DT        = 0.02                  # timestep
STOCH     = True

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

np.random.seed(42)

OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')


def make_params(n, H_c_frac=0.65, beta=2.0):
    """
    Build shared parameter dict for a network of size n.
    H_c_frac : H_c as fraction of max entropy (0.65 = 65% of max spread)
    beta     : EGDM sharpness (also used in sensitivity analysis)
    """
    H_max = max_entropy(n)
    H_c   = H_c_frac * H_max

    # Injection: small constant source at 20% of nodes
    s = np.zeros(n)
    source_nodes = np.random.choice(n, max(1, n // 5), replace=False)
    s[source_nodes] = 0.05

    return {
        'alpha':              0.3,       # diffusion rate
        'mu':                 1.5,       # mortality rate
        'H_c':                H_c,       # entropy threshold
        'beta':               beta,      # EGDM sharpness
        'n_hill':             4.0,       # Hill coefficient
        'beta_sir':           0.4,       # SIR infection rate
        'gamma':              0.15,      # SIR recovery rate
        'theta_ou':           0.3,       # OU reversion speed
        'mu_ou':              0.1,       # OU mean
        'sigma_ou':           0.04,      # OU volatility
        'kappa':              0.3,       # cascade threshold
        'deletion_threshold': 0.05,
        'sigma_noise':        0.015,     # stochastic noise level
        's':                  s,
        'T_train':            T,
        'dt':                 DT,
    }


def make_x0(n):
    """Initial condition: data concentrated at 10% of nodes."""
    x0 = np.zeros(n)
    seed_nodes = np.random.choice(n, max(1, n // 10), replace=False)
    x0[seed_nodes] = np.random.uniform(0.5, 1.0, len(seed_nodes))
    return x0


def compute_metrics(t_arr, H_arr, M_arr, dt):
    """Compute key metrics from a simulation trajectory."""
    M0 = M_arr[0] if M_arr[0] > 0 else 1.0

    # Mortality activation time: first time M drops below 50% of M0
    below_half = np.where(M_arr < 0.5 * M0)[0]
    t_star = t_arr[below_half[0]] if len(below_half) > 0 else t_arr[-1]

    # Time-integrated entropy exposure
    IEE = float(np.sum(H_arr * M_arr) * dt)

    return {
        't_star':     float(t_star),
        'IEE':        IEE,
        'final_mass': float(M_arr[-1]),
        'final_H':    float(H_arr[-1]),
        'H_arr':      H_arr,
        'M_arr':      M_arr,
        't_arr':      t_arr,
    }


def run_single_topology(n, topo_name, G, L, verbose=True):
    """
    Run all 7 models on a single (n, topology) combination.
    Returns dict: results[model_name] = metrics
    """
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    params  = make_params(n)
    x0      = make_x0(n)
    result  = {}

    if verbose:
        print(f"\n  Topology: {topo_name}  |  λ₂ = {lambda2:.3f}")

    # ── M0: No-deletion baseline ─────────────────────────────────
    if verbose: print("    Running M0 (Baseline, no deletion)...", end=' ', flush=True)
    t, _, H, M = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=STOCH)
    result['M0_Baseline'] = compute_metrics(t, H, M, DT)
    result['M0_Baseline']['lambda2'] = lambda2
    if verbose: print(f"IEE={result['M0_Baseline']['IEE']:.3f}")

    # ── M1: EGDM (baseline) ──────────────────────────────────────
    if verbose: print("    Running M1 (EGDM)...", end=' ', flush=True)
    t, _, H, M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=STOCH)
    result['M1_EGDM'] = compute_metrics(t, H, M, DT)
    result['M1_EGDM']['lambda2'] = lambda2
    if verbose: print(f"IEE={result['M1_EGDM']['IEE']:.3f}")

    # ── M2: Epidemic ─────────────────────────────────────────────
    if verbose: print("    Running M2 (Epidemic)...", end=' ', flush=True)
    t, _, H, M = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=STOCH)
    result['M2_Epidemic'] = compute_metrics(t, H, M, DT)
    if verbose: print(f"IEE={result['M2_Epidemic']['IEE']:.3f}")

    # ── M3: Finance ──────────────────────────────────────────────
    if verbose: print("    Running M3 (Finance)...", end=' ', flush=True)
    t, _, H, M = m3.simulate(x0, L, params, lambda2, T=T, dt=DT, stochastic=STOCH)
    result['M3_Finance'] = compute_metrics(t, H, M, DT)
    if verbose: print(f"IEE={result['M3_Finance']['IEE']:.3f}")

    # ── M4: Biology ──────────────────────────────────────────────
    if verbose: print("    Running M4 (Biology)...", end=' ', flush=True)
    t, _, H, M = m4.simulate(x0, L, params, T=T, dt=DT, stochastic=STOCH)
    result['M4_Biology'] = compute_metrics(t, H, M, DT)
    if verbose: print(f"IEE={result['M4_Biology']['IEE']:.3f}")

    # ── M5: Social ───────────────────────────────────────────────
    if verbose: print("    Running M5 (Social)...", end=' ', flush=True)
    t, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=STOCH)
    result['M5_Social'] = compute_metrics(t, H, M, DT)
    if verbose: print(f"IEE={result['M5_Social']['IEE']:.3f}")

    # ── M6: Ensemble (learn weights first, with regularisation) ──
    if verbose: print("    Learning ensemble weights (λ_reg=15.0)...", flush=True)
    training_data = [(x0, L, A, params, lambda2)]
    learned_w, _ = learn_weights(training_data, lambda_reg=15.0, n_restarts=5, verbose=verbose)

    if verbose: print("    Running M6 (ELAPSE Ensemble)...", end=' ', flush=True)
    t, _, H, M, Delta, votes = sim_m6(
        x0, L, A, params, lambda2,
        weights=learned_w, T=T, dt=DT, stochastic=STOCH
    )
    result['M6_Ensemble'] = compute_metrics(t, H, M, DT)
    result['M6_Ensemble']['learned_weights'] = learned_w
    result['M6_Ensemble']['Delta_arr']       = Delta
    result['M6_Ensemble']['votes_arr']       = votes
    if verbose: print(f"IEE={result['M6_Ensemble']['IEE']:.3f}")

    return result


def run_all(sizes=None, verbose=True):
    """
    Main loop: run all models on all networks and sizes.
    Returns nested dict: results[size][topology][model_name] = metrics
    """
    if sizes is None:
        sizes = SIZES

    results = {}

    for n in sizes:
        results[n] = {}
        if verbose:
            print(f"\n{'='*60}")
            print(f"Network size: n = {n}")
            print(f"{'='*60}")

        networks = get_all_networks(n)

        for topo_name, (G, L) in networks.items():
            results[n][topo_name] = run_single_topology(n, topo_name, G, L, verbose=verbose)

    return results


def run_sensitivity(n=100, topo_name='Erdos-Renyi', verbose=True):
    """
    Sensitivity analysis: vary H_c_frac and beta, measure IEE for M1 and M5.

    Returns:
        sens_results : dict with keys 'H_c_fracs', 'betas', 'M1_IEE', 'M5_IEE'
                       where M1_IEE[i, j] = IEE for H_c_frac[i], beta[j]
    """
    H_c_fracs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    betas     = [1, 2, 3, 4, 5]

    from networks import make_erdos_renyi
    G, L = make_erdos_renyi(n)
    A    = np.abs((L - np.diag(np.diag(L))) * -1)
    lam2 = fiedler_value(L)
    x0   = make_x0(n)

    M1_IEE = np.zeros((len(H_c_fracs), len(betas)))
    M5_IEE = np.zeros((len(H_c_fracs), len(betas)))

    total = len(H_c_fracs) * len(betas)
    done  = 0

    for i, hf in enumerate(H_c_fracs):
        for j, b in enumerate(betas):
            params = make_params(n, H_c_frac=hf, beta=b)
            params['s'] = np.zeros(n)   # no injection for clean sensitivity

            t, _, H, M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=False)
            M1_IEE[i, j] = float(np.sum(H * M) * DT)

            t, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=False)
            M5_IEE[i, j] = float(np.sum(H * M) * DT)

            done += 1
            if verbose:
                print(f"  Sensitivity {done}/{total}: H_c_frac={hf:.2f}, β={b}  "
                      f"→ M1_IEE={M1_IEE[i,j]:.2f}, M5_IEE={M5_IEE[i,j]:.2f}")

    return {
        'H_c_fracs': H_c_fracs,
        'betas':     betas,
        'M1_IEE':   M1_IEE,
        'M5_IEE':   M5_IEE,
        'n':         n,
        'topo':      topo_name,
    }


if __name__ == '__main__':
    print("Running ELAPSE simulations...")
    results = run_all(verbose=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {OUTPUT_DIR}/results.pkl")

    print("\nRunning sensitivity analysis...")
    sens = run_sensitivity(n=100, verbose=True)
    with open(os.path.join(OUTPUT_DIR, 'sensitivity.pkl'), 'wb') as f:
        pickle.dump(sens, f)
    print(f"Sensitivity saved to {OUTPUT_DIR}/sensitivity.pkl")
