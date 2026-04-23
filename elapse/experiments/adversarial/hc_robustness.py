"""
hc_robustness.py
----------------
Experiment: robustness to H_c misspecification.

Sweeps H_c from 0.4*H_max to 0.9*H_max.
M6 should have lower IEE_range (more robust) than single mechanisms.
"""

import numpy as np
import pickle, os, sys, multiprocessing
import concurrent.futures

N_WORKERS = min(multiprocessing.cpu_count(), 8)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy
import m0_baseline as m0, m1_egdm as m1, m2_epidemic as m2
import m3_finance as m3, m4_biology as m4, m5_social as m5
from m6_ensemble import simulate as sim_m6, learn_weights

T = 15.0
DT = 0.02
N_TRAIN = 10
N_TEST = 20
HC_FRACS = np.linspace(0.4, 0.9, 11)  # 11 values of H_c / H_max


def make_params(n, hc_frac, seed=None):
    H_max = max_entropy(n)
    H_c   = hc_frac * H_max
    rng   = np.random.default_rng(seed)
    s     = np.zeros(n)
    src   = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'sigma_noise': 0.015, 's': s, 'T_train': T, 'dt': DT,
    }


def make_x0(n, seed=None):
    rng = np.random.default_rng(seed)
    x0  = np.zeros(n)
    idx = rng.choice(n, max(1, n//10), replace=False)
    x0[idx] = rng.uniform(0.5, 1.0, len(idx))
    return x0


def compute_iee(H, M, dt):
    return float(np.sum(H * M) * dt)


def _hc_seed_worker(args):
    """Top-level worker: run all models for one (hc_frac, seed) pair."""
    n, L, A, lambda2, hc_frac, seed, w_learned = args
    params = make_params(n, hc_frac, seed=seed)
    x0     = make_x0(n, seed=seed)
    np.random.seed(seed)
    iees = {}
    _, _, H, M = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    iees['M0'] = compute_iee(H, M, DT)
    _, _, H, M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    iees['M1'] = compute_iee(H, M, DT)
    _, _, H, M = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=True)
    iees['M2'] = compute_iee(H, M, DT)
    _, _, H, M = m3.simulate(x0, L, params, lambda2, T=T, dt=DT, stochastic=True)
    iees['M3'] = compute_iee(H, M, DT)
    _, _, H, M = m4.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    iees['M4'] = compute_iee(H, M, DT)
    _, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
    iees['M5'] = compute_iee(H, M, DT)
    _, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                               weights=w_learned, T=T, dt=DT, stochastic=True)
    iees['M6'] = compute_iee(H, M, DT)
    return hc_frac, iees


def run_hc_robustness(n=100, n_seeds=N_TEST, verbose=True):
    """Sweep H_c and compare IEE ranges across mechanisms."""
    G, L = make_erdos_renyi(n, seed=42)
    A = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)

    models = ['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']

    # Results: {model: [mean_IEE per hc_frac]}
    results = {m: [] for m in models}

    # Learn M6 weights at nominal H_c = 0.65
    nominal_data = []
    for seed in range(N_TRAIN):
        params = make_params(n, hc_frac=0.65, seed=seed)
        params['dt'] = 0.05
        x0 = make_x0(n, seed=seed)
        nominal_data.append((x0, L, A, params, lambda2))

    w_learned, _ = learn_weights(nominal_data, lambda_reg=15.0, n_restarts=2, verbose=False)

    if verbose:
        print(f"Hc robustness sweep: n={n}, {len(HC_FRACS)} H_c values × "
              f"{n_seeds} seeds = {len(HC_FRACS)*n_seeds} sims (parallel)")

    test_seeds = list(range(N_TRAIN, N_TRAIN + n_seeds))
    worker_args = [
        (n, L, A, lambda2, hc_frac, seed, w_learned)
        for hc_frac in HC_FRACS
        for seed in test_seeds
    ]

    # Accumulate: {hc_frac: {model: [iees]}}
    hc_iees = {hc: {m: [] for m in models} for hc in HC_FRACS}
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for hc_frac, iees_dict in pool.map(_hc_seed_worker, worker_args):
            for m in models:
                hc_iees[hc_frac][m].append(iees_dict[m])

    for hc_frac in HC_FRACS:
        for m in models:
            results[m].append(float(np.mean(hc_iees[hc_frac][m])))
        if verbose:
            print(f"  H_c/H_max={hc_frac:.2f}: "
                  f"M5={np.mean(hc_iees[hc_frac]['M5']):.2f}, "
                  f"M6={np.mean(hc_iees[hc_frac]['M6']):.2f}")

    # Compute IEE range (max-min) for each model
    iee_ranges = {m: float(np.max(results[m]) - np.min(results[m])) for m in models}

    if verbose:
        print("\nIEE ranges (robustness measure):")
        for m in models:
            print(f"  {m}: {iee_ranges[m]:.3f}")
        most_robust = min(models, key=lambda m: iee_ranges[m])
        print(f"Most robust: {most_robust}")

    return {
        'hc_fracs': HC_FRACS.tolist(),
        'iee_by_hc': results,
        'iee_ranges': iee_ranges,
        'w_learned': w_learned,
        'lambda2': lambda2,
    }


if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running H_c robustness experiment...")
    results = run_hc_robustness(n=100, n_seeds=20, verbose=True)

    with open(os.path.join(OUTPUT_DIR, 'hc_robustness_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print("Done.")
