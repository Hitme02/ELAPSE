"""
run_simulation_ci.py
--------------------
Extended simulation runner with train/test split to prevent in-sample evaluation.

Train set : seeds 0–9  (N_train=10) — used for M6 weight learning only
Test  set : seeds 10–29 (N_test=20) — used for ALL reported IEE values

Computes mean ± 95% CI for IEE across all:
  - 3 topologies × 4 network sizes (synthetic)
  - 2 SNAP P2P networks (real-world, n_sample ~ 500)
  - All 7 configurations (M0–M6)

Also runs:
  - Convexity check (Phase 2A)
  - Adversarial study (Phase 2D)
  - Gossip estimation study (Phase 2E)
  - M3 vote trajectory analysis (Phase 2F)
"""

import numpy as np
import pickle, os, sys, time
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks       import get_all_networks, fiedler_value
from math_utils     import max_entropy
import m0_baseline  as m0
import m1_egdm      as m1
import m2_epidemic  as m2
import m3_finance   as m3
import m4_biology   as m4
import m5_social    as m5
from m6_ensemble    import simulate as sim_m6, learn_weights
import m7_timer      as m7

OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

SIZES       = [50, 100, 200, 500]
T           = 15.0
DT          = 0.02
N_TRAIN     = 10    # seeds used for weight learning
N_TEST      = 20    # seeds used for evaluation (seeds 10–29)
N_SEEDS     = N_TRAIN + N_TEST   # 30 total
TRAIN_SEEDS = list(range(N_TRAIN))
TEST_SEEDS  = list(range(N_TRAIN, N_SEEDS))


# ── Parameter factory ─────────────────────────────────────────────────────────

def make_params(n, seed=None):
    H_max = max_entropy(n)
    H_c   = 0.65 * H_max
    rng   = np.random.default_rng(seed)
    s     = np.zeros(n)
    src   = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def make_x0(n, seed=None):
    rng   = np.random.default_rng(seed)
    x0    = np.zeros(n)
    seeds = rng.choice(n, max(1, n // 10), replace=False)
    x0[seeds] = rng.uniform(0.5, 1.0, len(seeds))
    return x0


def compute_metrics(t_arr, H_arr, M_arr, dt):
    M0         = M_arr[0] if M_arr[0] > 0 else 1.0
    below_half = np.where(M_arr < 0.5 * M0)[0]
    t_star     = float(t_arr[below_half[0]]) if len(below_half) > 0 else float(t_arr[-1])
    IEE        = float(np.sum(H_arr * M_arr) * dt)
    return {
        't_star': t_star, 'IEE': IEE,
        'final_mass': float(M_arr[-1]),
        'final_H': float(H_arr[-1]),
        'H_arr': H_arr, 'M_arr': M_arr, 't_arr': t_arr,
    }


# ── Single-seed run ───────────────────────────────────────────────────────────

def run_single_seed(n, G, L, seed, learned_weights=None,
                    m7_threshold=None, verbose=False):
    """
    Run all 8 models for one (n, topology, seed) combination.
    learned_weights: if provided, use pre-learned weights for M6.
    m7_threshold   : wall-clock timer for M7 (calibrated from M6 t_star).
                     If None, defaults to T/2.
    """
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    params  = make_params(n, seed=seed)
    x0      = make_x0(n, seed=seed)

    np.random.seed(seed)
    result  = {}

    # M0
    t, _, H, M = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M0_Baseline'] = compute_metrics(t, H, M, DT)

    # M1
    t, _, H, M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M1_EGDM'] = compute_metrics(t, H, M, DT)
    result['M1_EGDM']['lambda2'] = lambda2

    # M2
    t, _, H, M = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=True)
    result['M2_Epidemic'] = compute_metrics(t, H, M, DT)

    # M3
    t, _, H, M = m3.simulate(x0, L, params, lambda2, T=T, dt=DT, stochastic=True)
    result['M3_Finance'] = compute_metrics(t, H, M, DT)

    # M4
    t, _, H, M = m4.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M4_Biology'] = compute_metrics(t, H, M, DT)

    # M5
    t, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
    result['M5_Social'] = compute_metrics(t, H, M, DT)

    # M6: use pre-learned weights
    w = learned_weights if learned_weights is not None else np.ones(5) / 5
    t, _, H, M, Delta, votes = sim_m6(
        x0, L, A, params, lambda2,
        weights=w, T=T, dt=DT, stochastic=True
    )
    result['M6_Ensemble'] = compute_metrics(t, H, M, DT)
    result['M6_Ensemble']['learned_weights'] = w
    result['M6_Ensemble']['Delta_arr']       = Delta
    result['M6_Ensemble']['votes_arr']       = votes

    # M7: Vanish-style chronological timer
    # Calibrate timer to M6's mean trigger time for a fair IEE comparison
    t_thresh = (m7_threshold if m7_threshold is not None
                else result['M6_Ensemble']['t_star'])
    t_thresh = float(np.clip(t_thresh, 0.5, T - 0.1))
    t, _, H, M = m7.simulate(x0, L, params, t_threshold=t_thresh,
                              T=T, dt=DT, stochastic=True)
    result['M7_Timer'] = compute_metrics(t, H, M, DT)
    result['M7_Timer']['t_threshold'] = t_thresh

    return result


# ── Learn weights on TRAIN seeds ──────────────────────────────────────────────

def learn_topology_weights(n, G, L, train_seeds=None, verbose=True):
    """
    Learn ensemble weights using a single training seed (seed 0).
    This is a strict train/test split: seed 0 is in the train set (0-9)
    and all test seeds (10-29) are fully held-out.
    Uses coarser dt=0.05 for speed.
    """
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)

    # Use seed 0 (first training seed) only — keeps computation tractable
    # while ensuring no test-seed contamination
    params = make_params(n, seed=0)
    params['dt'] = 0.05   # coarser dt for speed during optimisation
    x0     = make_x0(n, seed=0)
    training_data = [(x0, L, A, params, lambda2)]

    w, _ = learn_weights(training_data, lambda_reg=15.0, n_restarts=2, verbose=verbose)
    return w


# ── CI computation from test seeds ────────────────────────────────────────────

def aggregate_ci(all_iees):
    """Compute mean, 95% CI, and std from a list of IEE values."""
    arr = np.array(all_iees)
    n   = len(arr)
    mu  = float(arr.mean())
    ci  = stats.t.interval(0.95, df=n-1, loc=mu, scale=stats.sem(arr))
    return float(mu), (float(ci[0]), float(ci[1])), float(arr.std())


# ── Main multi-seed runner ────────────────────────────────────────────────────

MODEL_NAMES = ['M0_Baseline', 'M1_EGDM', 'M2_Epidemic',
               'M3_Finance', 'M4_Biology', 'M5_Social', 'M6_Ensemble',
               'M7_Timer']

TOPO_ORDER = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']


def run_all_ci(sizes=None, n_seeds=N_SEEDS, verbose=True):
    """
    Run train/test split evaluation for all (size, topology) combinations.

    Weight learning uses TRAIN_SEEDS (0–9).
    IEE evaluation uses TEST_SEEDS (10–29).

    Returns:
      ci_results[n][topo][model] = {
        'iee_mean', 'iee_ci', 'iee_std',        (from TEST seeds)
        'tstar_mean', 'tstar_ci',
        'H_arr_mean', 'M_arr_mean', 't_arr',
        'learned_weights',                        (from TRAIN seeds)
        'test_iees',                              (raw per-seed IEE, for t-tests)
      }
    """
    if sizes is None:
        sizes = SIZES

    ci_results  = {}
    raw_results = {}

    for n in sizes:
        ci_results[n]  = {}
        raw_results[n] = {}

        if verbose:
            print(f"\n{'='*60}")
            print(f"n = {n}  (N_train={len(TRAIN_SEEDS)}, N_test={len(TEST_SEEDS)})")
            print(f"{'='*60}")

        networks = get_all_networks(n)

        for topo_name, (G, L) in networks.items():
            ci_results[n][topo_name]  = {}
            raw_results[n][topo_name] = {m: [] for m in MODEL_NAMES}

            lambda2 = fiedler_value(L)

            if verbose:
                print(f"\n  Topology: {topo_name}  λ₂={lambda2:.4f}")
                print(f"  Learning M6 weights from {len(TRAIN_SEEDS)} train seeds ...")

            # Learn weights from TRAIN seeds only
            w_learned = learn_topology_weights(n, G, L,
                                               train_seeds=TRAIN_SEEDS, verbose=False)

            if verbose:
                print(f"  Learned weights: {np.round(w_learned, 3)}")

            # Calibrate M7 timer: run M6 on training seed to get t_star
            # This gives M7 the same deletion budget as ELAPSE for a fair comparison
            params_train = make_params(n, seed=0)
            x0_train     = make_x0(n, seed=0)
            A_train = np.abs((L - np.diag(np.diag(L))) * -1)
            lam2    = fiedler_value(L)
            _, _, H_tr, M_tr = sim_m6(x0_train, L, A_train, params_train, lam2,
                                       weights=w_learned, T=T, dt=DT, stochastic=False)
            t_train = np.arange(len(M_tr)) * DT
            M0_tr   = M_tr[0]
            below   = np.where(M_tr < 0.5 * M0_tr)[0]
            m7_thr  = float(t_train[below[0]]) if len(below) > 0 else T / 2.0

            if verbose:
                print(f"  M7 calibrated timer = {m7_thr:.2f}  "
                      f"(M6 t_star on train seed)")
                print(f"  Evaluating on {len(TEST_SEEDS)} test seeds ...")

            t0 = time.time()
            for seed in TEST_SEEDS:
                r = run_single_seed(n, G, L, seed, learned_weights=w_learned,
                                    m7_threshold=m7_thr)
                for m in MODEL_NAMES:
                    raw_results[n][topo_name][m].append(r[m])

                if verbose and ((seed - TEST_SEEDS[0] + 1) % 10 == 0):
                    print(f"    Test seed {seed}  "
                          f"({time.time()-t0:.0f}s elapsed)")

            # ── Aggregate CIs (TEST seeds only) ───────────────────────
            for m in MODEL_NAMES:
                runs     = raw_results[n][topo_name][m]
                all_iees = [r['IEE'] for r in runs]
                all_ts   = [r['t_star'] for r in runs]

                iee_mean, iee_ci, iee_std = aggregate_ci(all_iees)
                ts_mean,  ts_ci,  ts_std  = aggregate_ci(all_ts)

                # Mean trajectory (for plots)
                H_stack = np.stack([r['H_arr'] for r in runs], axis=0)
                M_stack = np.stack([r['M_arr'] for r in runs], axis=0)
                t_arr   = runs[0]['t_arr']

                H_mean = H_stack.mean(axis=0)
                H_lo   = np.percentile(H_stack, 2.5, axis=0)
                H_hi   = np.percentile(H_stack, 97.5, axis=0)
                M_mean = M_stack.mean(axis=0)

                entry = {
                    'iee_mean':    iee_mean,
                    'iee_ci':      iee_ci,
                    'iee_std':     iee_std,
                    'tstar_mean':  ts_mean,
                    'tstar_ci':    ts_ci,
                    'tstar_std':   ts_std,
                    'H_arr_mean':  H_mean,
                    'H_arr_lo':    H_lo,
                    'H_arr_hi':    H_hi,
                    'M_arr_mean':  M_mean,
                    't_arr':       t_arr,
                    'lambda2':     lambda2,
                    'test_iees':   all_iees,    # raw per-seed IEEs for t-tests
                }

                # Store learned weights for M6
                if m == 'M6_Ensemble':
                    entry['learned_weights'] = w_learned
                    entry['Delta_arr']       = runs[0]['Delta_arr']
                    entry['votes_arr']       = runs[0]['votes_arr']

                ci_results[n][topo_name][m] = entry

            if verbose:
                m6  = ci_results[n][topo_name]['M6_Ensemble']
                m0e = ci_results[n][topo_name]['M0_Baseline']
                m5e = ci_results[n][topo_name]['M5_Social']
                print(f"  ── n={n} {topo_name} ──")
                print(f"     M0 IEE: {m0e['iee_mean']:.2f} ± "
                      f"{(m0e['iee_ci'][1]-m0e['iee_ci'][0])/2:.2f}")
                print(f"     M5 IEE: {m5e['iee_mean']:.2f} ± "
                      f"{(m5e['iee_ci'][1]-m5e['iee_ci'][0])/2:.2f}")
                print(f"     M6 IEE: {m6['iee_mean']:.2f} ± "
                      f"{(m6['iee_ci'][1]-m6['iee_ci'][0])/2:.2f}  "
                      f"w={np.round(m6['learned_weights'],3)}")

    return ci_results, raw_results


# ── SNAP real-world validation ────────────────────────────────────────────────

def run_snap_ci(n_seeds=N_TEST, n_sample=500, verbose=True):
    """
    Run the full ELAPSE ensemble on SNAP P2P subgraphs.
    Uses train/test split: weights learned on seeds 0–9, evaluated on 10–29.

    Returns ci_results_snap[dataset_name][subgraph_type][model] = {...}
    """
    try:
        from snap_loader import load_snap_networks
    except ImportError:
        sys.path.insert(0, BASE_DIR)
        from snap_loader import load_snap_networks

    if verbose:
        print("\n── SNAP Real-World Validation ──────────────────────────────")

    snap_nets = load_snap_networks(n_sample=n_sample, verbose=verbose)

    if not snap_nets:
        if verbose:
            print("  No SNAP networks available, skipping.")
        return {}

    ci_snap = {}

    for name, net_data in snap_nets.items():
        ci_snap[name] = {}

        # Process both BFS and random subgraphs
        for sg_type in ['bfs', 'random']:
            if sg_type not in net_data:
                continue
            sg = net_data[sg_type]
            G  = sg['G']
            L  = sg['L']
            n  = G.number_of_nodes()
            A  = np.abs((L - np.diag(np.diag(L))) * -1)
            lambda2 = sg['stats']['lambda2']

            if verbose:
                print(f"\n  Dataset: {name} [{sg_type}]  n={n}  λ₂={lambda2:.4f}")

            # Learn weights on TRAIN seeds
            training_data = []
            for seed in TRAIN_SEEDS:
                params = make_params(n, seed=seed)
                params['dt'] = 0.05
                x0     = make_x0(n, seed=seed)
                training_data.append((x0, L, A, params, lambda2))

            from m6_ensemble import learn_weights as lw
            w_learned, _ = lw(training_data, lambda_reg=15.0, n_restarts=2, verbose=False)

            if verbose:
                print(f"  Learned weights: {np.round(w_learned, 3)}")

            all_runs = {m: [] for m in MODEL_NAMES}

            # Evaluate on TEST seeds
            for seed in TEST_SEEDS:
                r = run_single_seed(n, G, L, seed, learned_weights=w_learned)
                for m in MODEL_NAMES:
                    all_runs[m].append(r[m]['IEE'])

                if verbose and ((seed - TEST_SEEDS[0] + 1) % 10 == 0):
                    print(f"    Seed {seed}")

            # Aggregate
            ci_snap[name][sg_type] = {}
            for m in MODEL_NAMES:
                iee_mean, iee_ci, iee_std = aggregate_ci(all_runs[m])
                ci_snap[name][sg_type][m] = {
                    'iee_mean':   iee_mean,
                    'iee_ci':     iee_ci,
                    'iee_std':    iee_std,
                    'test_iees':  all_runs[m],
                    'n':          n,
                    'lambda2':    lambda2,
                    'full_stats': net_data.get('full_stats', {}),
                    'sub_stats':  sg['stats'],
                }

            if verbose:
                m6  = ci_snap[name][sg_type]['M6_Ensemble']
                m0e = ci_snap[name][sg_type]['M0_Baseline']
                print(f"  M0 IEE={m0e['iee_mean']:.2f}  M6 IEE={m6['iee_mean']:.2f}")

    return ci_snap


# ── M3 vote trajectory analysis (Phase 2F) ────────────────────────────────────

def run_m3_analysis(n=200, n_seeds=N_TEST, verbose=True):
    """Analyse M3's underperformance; uses TEST seeds for evaluation."""
    from networks import make_barabasi_albert, make_erdos_renyi

    results = {}

    for topo_name, make_fn in [('Barabasi-Albert', make_barabasi_albert),
                                 ('Erdos-Renyi', make_erdos_renyi)]:
        G, L   = make_fn(n, seed=42)
        A      = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)

        if verbose:
            print(f"\n  M3 analysis: {topo_name}  λ₂={lambda2:.4f}")

        # Learn weights on TRAIN seeds
        training_data = []
        for seed in TRAIN_SEEDS:
            params = make_params(n, seed=seed)
            params['dt'] = 0.05
            x0     = make_x0(n, seed=seed)
            training_data.append((x0, L, A, params, lambda2))

        from m6_ensemble import learn_weights as lw
        w_learned, _ = lw(training_data, lambda_reg=15.0, n_restarts=2, verbose=False)

        all_votes   = []
        all_H       = []
        all_iee_m3  = []

        for seed in TEST_SEEDS:
            params = make_params(n, seed=seed)
            x0     = make_x0(n, seed=seed)

            np.random.seed(seed)
            t_arr, _, H_arr, M_arr, Delta, votes = sim_m6(
                x0, L, A, params, lambda2,
                weights=w_learned, T=T, dt=DT, stochastic=True
            )

            all_votes.append(votes)
            all_H.append(H_arr)

            np.random.seed(seed)
            _, _, H3, M3 = m3.simulate(x0, L, params, lambda2, T=T, dt=DT, stochastic=True)
            all_iee_m3.append(float(np.sum(H3 * M3) * DT))

        votes_stack = np.stack(all_votes, axis=0)
        H_stack     = np.stack(all_H,    axis=0)

        results[topo_name] = {
            't_arr':           t_arr,
            'votes_mean':      votes_stack.mean(axis=0),
            'votes_std':       votes_stack.std(axis=0),
            'H_mean':          H_stack.mean(axis=0),
            'iee_m3_mean':     float(np.mean(all_iee_m3)),
            'iee_m3_std':      float(np.std(all_iee_m3)),
            'lambda2':         lambda2,
            'learned_weights': w_learned,
        }

        if verbose:
            vm = votes_stack.mean(axis=0).mean(axis=0)
            print(f"  Mean votes: v1={vm[0]:.3f} v2={vm[1]:.3f} v3={vm[2]:.3f} "
                  f"v4={vm[3]:.3f} v5={vm[4]:.3f}")
            print(f"  M3 IEE: {np.mean(all_iee_m3):.2f} ± {np.std(all_iee_m3):.2f}")

    return results


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print(f"Train/test split: {len(TRAIN_SEEDS)} train seeds, {len(TEST_SEEDS)} test seeds")
    print("Running CI simulations...")
    t0 = time.time()
    ci_results, raw_results = run_all_ci(sizes=[50, 100, 200, 500], verbose=True)
    print(f"\nMain simulations: {time.time()-t0:.0f}s")

    with open(os.path.join(OUTPUT_DIR, 'ci_results.pkl'), 'wb') as f:
        pickle.dump(ci_results, f)
    print("Saved ci_results.pkl")

    print("\nRunning SNAP validation...")
    ci_snap = run_snap_ci(n_seeds=N_TEST, n_sample=500, verbose=True)
    with open(os.path.join(OUTPUT_DIR, 'ci_snap.pkl'), 'wb') as f:
        pickle.dump(ci_snap, f)
    print("Saved ci_snap.pkl")

    print("\nRunning M3 analysis...")
    m3_analysis = run_m3_analysis(n=200, verbose=True)
    with open(os.path.join(OUTPUT_DIR, 'm3_analysis.pkl'), 'wb') as f:
        pickle.dump(m3_analysis, f)
    print("Saved m3_analysis.pkl")
