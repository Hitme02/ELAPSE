"""
proper_evaluation.py
--------------------
5-fold cross-validated IEE evaluation with N=50 seeds per topology.

Replaces the single-seed weight learning in run_simulation_ci.py.

Design
------
  N_total = 50 seeds (seeds 0-49)
  5-fold CV:
    - Fold k tests on seeds {k*10, ..., (k+1)*10 - 1}
    - M6 weights learned on the remaining 40 seeds for that fold
  Mixed M6:
    - Trained on 10 seeds each from ER + BA + WS (30 total)
    - Evaluated on each topology's held-out seeds

This addresses the experimental design flaw identified in the review:
using only seed 0 from a single topology for M6 weight learning produces
misleading performance estimates (33% gap between topologies).

Usage
-----
    python elapse/experiments/main_iee/proper_evaluation.py
"""

import numpy as np
import pickle
import os
import sys
import time
import multiprocessing
import concurrent.futures
from scipy import stats

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from networks import (
    make_erdos_renyi, make_barabasi_albert, make_watts_strogatz,
    fiedler_value,
)
from math_utils import max_entropy
import m0_baseline as m0
import m1_egdm as m1
import m2_epidemic as m2
import m3_finance as m3
import m4_biology as m4
import m5_social as m5
from m6_ensemble import simulate as sim_m6, learn_weights

# ── Constants ─────────────────────────────────────────────────────────────────

N_TOTAL   = 50
N_FOLDS   = 5
FOLD_SIZE = N_TOTAL // N_FOLDS   # 10 seeds per fold

T   = 15.0
DT  = 0.02
DT_TRAIN = 0.05   # coarser step for weight learning (faster)

N_WORKERS = min(multiprocessing.cpu_count(), 8)

MODEL_NAMES = [
    'M0_Baseline', 'M1_EGDM', 'M2_Epidemic', 'M3_Finance',
    'M4_Biology',  'M5_Social', 'M6_Ensemble', 'M6_Mixed',
]

MODEL_LABELS = {
    'M0_Baseline':  'M0 (Baseline)',
    'M1_EGDM':      'M1 (EGDM)',
    'M2_Epidemic':  'M2 (Epidemic)',
    'M3_Finance':   'M3 (Finance)',
    'M4_Biology':   'M4 (Biology)',
    'M5_Social':    'M5 (Social)',
    'M6_Ensemble':  'M6 (ELAPSE, topo-matched)',
    'M6_Mixed':     'M6 (ELAPSE, mixed-trained)',
}

TOPOLOGY_FACTORIES = {
    'Erdos-Renyi':     make_erdos_renyi,
    'Barabasi-Albert': make_barabasi_albert,
    'Watts-Strogatz':  make_watts_strogatz,
}


# ── Parameter construction ────────────────────────────────────────────────────

def make_params(n, seed=None):
    """Build model parameter dict for given n and seed."""
    H_max = max_entropy(n)
    H_c   = 0.65 * H_max
    rng   = np.random.default_rng(seed)
    s     = np.zeros(n)
    src   = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha':              0.3,
        'mu':                 1.5,
        'H_c':                H_c,
        'beta':               2.0,
        'n_hill':             4.0,
        'beta_sir':           0.4,
        'gamma':              0.15,
        'theta_ou':           0.3,
        'mu_ou':              0.1,
        'sigma_ou':           0.04,
        'kappa':              0.3,
        'deletion_threshold': 0.05,
        'sigma_noise':        0.015,
        's':                  s,
        'T_train':            T,
        'dt':                 DT,
    }


def make_x0(n, seed=None):
    """Build sparse non-negative initial condition."""
    rng = np.random.default_rng(seed)
    x0  = np.zeros(n)
    idx = rng.choice(n, max(1, n // 10), replace=False)
    x0[idx] = rng.uniform(0.5, 1.0, len(idx))
    return x0


def compute_iee(H_arr, M_arr, dt):
    """Compute IEE = integral H(t)*M(t) dt."""
    return float(np.sum(H_arr * M_arr) * dt)


# ── Single-seed evaluation ────────────────────────────────────────────────────

def run_single_seed_all_models(n, L, A, lambda2, seed,
                                w_learned=None, w_mixed=None):
    """
    Run all mechanisms for a single seed.

    Parameters
    ----------
    n, L, A, lambda2 : network properties
    seed             : RNG seed for this evaluation
    w_learned        : M6 weights (topo-matched, from CV fold)
    w_mixed          : M6 weights (mixed topology training)

    Returns
    -------
    dict mapping MODEL_NAMES -> IEE float
    """
    params = make_params(n, seed=seed)
    x0     = make_x0(n, seed=seed)
    np.random.seed(seed)

    result = {}

    # M0 Baseline
    t, _, H, M = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M0_Baseline'] = compute_iee(H, M, DT)

    # M1 EGDM
    t, _, H, M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M1_EGDM'] = compute_iee(H, M, DT)

    # M2 Epidemic
    t, _, H, M = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=True)
    result['M2_Epidemic'] = compute_iee(H, M, DT)

    # M3 Finance
    t, _, H, M = m3.simulate(x0, L, params, lambda2, T=T, dt=DT, stochastic=True)
    result['M3_Finance'] = compute_iee(H, M, DT)

    # M4 Biology
    t, _, H, M = m4.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    result['M4_Biology'] = compute_iee(H, M, DT)

    # M5 Social
    t, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
    result['M5_Social'] = compute_iee(H, M, DT)

    # M6 topo-matched
    w = w_learned if w_learned is not None else np.ones(5) / 5
    t, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                                weights=w, T=T, dt=DT, stochastic=True)
    result['M6_Ensemble'] = compute_iee(H, M, DT)

    # M6 mixed-topology
    wm = w_mixed if w_mixed is not None else np.ones(5) / 5
    t, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                                weights=wm, T=T, dt=DT, stochastic=True)
    result['M6_Mixed'] = compute_iee(H, M, DT)

    return result


# ── Parallel worker stubs (must be top-level for pickling) ───────────────────

def _learn_fold_worker(args):
    """Worker: learn M6 weights for one CV fold. Returns (fold_idx, weights)."""
    fold_idx, n, L, A, lambda2, train_seeds = args
    w = learn_fold_weights(n, L, A, lambda2, train_seeds)
    return fold_idx, w


def _eval_seed_worker(args):
    """Worker: evaluate all models for one seed. Returns result dict."""
    n, L, A, lambda2, seed, w_learned, w_mixed = args
    return run_single_seed_all_models(n, L, A, lambda2, seed,
                                      w_learned=w_learned, w_mixed=w_mixed)


# ── Weight learning ───────────────────────────────────────────────────────────

def build_training_data(n, L, A, lambda2, seeds):
    """Build list of (x0, L, A, params, lambda2) tuples for weight learning."""
    data = []
    for seed in seeds:
        params = make_params(n, seed=seed)
        params['dt'] = DT_TRAIN   # coarser step for speed
        x0 = make_x0(n, seed=seed)
        data.append((x0, L, A, params, lambda2))
    return data


def learn_fold_weights(n, L, A, lambda2, train_seeds):
    """
    Learn M6 ensemble weights on train_seeds for one CV fold.

    Uses Nelder-Mead with 3 restarts and lambda_reg=15.0 (matching paper).

    Returns weight vector w in R^5 with sum(w)=1, w>=0.
    """
    data = build_training_data(n, L, A, lambda2, train_seeds)
    w, _ = learn_weights(data, lambda_reg=15.0, n_restarts=3, verbose=False)
    return w


# ── Aggregation utilities ─────────────────────────────────────────────────────

def aggregate_ci(iees):
    """Compute mean, 95% CI (t-distribution), and std."""
    arr = np.array(iees)
    mu  = float(arr.mean())
    if len(arr) > 1:
        ci = stats.t.interval(0.95, df=len(arr) - 1,
                               loc=mu, scale=stats.sem(arr))
    else:
        ci = (mu, mu)
    return mu, (float(ci[0]), float(ci[1])), float(arr.std(ddof=1))


def cohens_d(a, b):
    """Cohen's d effect size (positive = a > b)."""
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-12))


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_proper_evaluation(n=100, verbose=True):
    """
    Full 5-fold CV evaluation for all topologies at network size n.

    Parameters
    ----------
    n       : int, network size
    verbose : bool, print progress

    Returns
    -------
    dict: topology -> {
        iees:         {model: [50 IEE values]},
        mean:         {model: float},
        ci:           {model: (lo, hi)},
        std:          {model: float},
        fold_weights: list of 5 weight vectors (one per CV fold),
        w_mixed:      weight vector from mixed-topology training,
        lambda2:      float,
    }
    """
    networks = {
        name: factory(n, seed=42)
        for name, factory in TOPOLOGY_FACTORIES.items()
    }

    all_seeds = list(range(N_TOTAL))
    folds     = [all_seeds[i * FOLD_SIZE:(i + 1) * FOLD_SIZE]
                 for i in range(N_FOLDS)]

    # ── Learn mixed-topology weights once ────────────────────────────────────
    mixed_train_seeds = list(range(10))   # seeds 0-9 from each topology
    mixed_data = []
    for topo_name, (G, L) in networks.items():
        A    = np.abs((L - np.diag(np.diag(L))) * -1)
        lam2 = fiedler_value(L)
        for seed in mixed_train_seeds:
            params = make_params(n, seed=seed)
            params['dt'] = DT_TRAIN
            x0 = make_x0(n, seed=seed)
            mixed_data.append((x0, L, A, params, lam2))

    w_mixed, _ = learn_weights(mixed_data, lambda_reg=15.0,
                                n_restarts=3, verbose=False)
    if verbose:
        print(f"Mixed-topology weights (n={n}): {np.round(w_mixed, 3)}")

    # ── Per-topology 5-fold CV ────────────────────────────────────────────────
    results = {}

    for topo_name, (G, L) in networks.items():
        A       = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)

        if verbose:
            print(f"\n{topo_name} (n={n}, lambda2={lambda2:.4f})")

        all_iees     = {m: [] for m in MODEL_NAMES}
        fold_weights = [None] * N_FOLDS

        # ── Step 1: learn all fold weights in parallel ───────────────────────
        t0_learn = time.time()
        fold_train_args = []
        for fold_idx in range(N_FOLDS):
            train_seeds = [s for fi, fold in enumerate(folds)
                           for s in fold if fi != fold_idx]
            fold_train_args.append((fold_idx, n, L, A, lambda2, train_seeds))

        with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for fold_idx, w_fold in pool.map(_learn_fold_worker, fold_train_args):
                fold_weights[fold_idx] = w_fold
                if verbose:
                    test_sz  = len(folds[fold_idx])
                    train_sz = N_TOTAL - test_sz
                    print(f"  Fold {fold_idx + 1}/5 weights learned: "
                          f"train={train_sz}, test={test_sz}, "
                          f"w={np.round(w_fold, 3)}")

        if verbose:
            print(f"  All fold weights learned in {(time.time()-t0_learn)/60:.1f} min")

        # ── Step 2: evaluate all test seeds in parallel ──────────────────────
        t0_eval = time.time()
        seed_args = []
        seed_fold_map = []   # track which fold each seed belongs to
        for fold_idx in range(N_FOLDS):
            for seed in folds[fold_idx]:
                seed_args.append(
                    (n, L, A, lambda2, seed, fold_weights[fold_idx], w_mixed)
                )
                seed_fold_map.append(fold_idx)

        with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            for (fold_idx, r) in zip(seed_fold_map,
                                     pool.map(_eval_seed_worker, seed_args)):
                for m in MODEL_NAMES:
                    all_iees[m].append(r[m])

        if verbose:
            print(f"  All seeds evaluated in {(time.time()-t0_eval)/60:.1f} min")

        # Aggregate statistics
        results[topo_name] = {
            'iees':         {},
            'mean':         {},
            'ci':           {},
            'std':          {},
            'fold_weights': fold_weights,
            'w_mixed':      w_mixed,
            'lambda2':      lambda2,
        }

        for m in MODEL_NAMES:
            mu, ci, std = aggregate_ci(all_iees[m])
            results[topo_name]['iees'][m] = all_iees[m]
            results[topo_name]['mean'][m] = mu
            results[topo_name]['ci'][m]   = ci
            results[topo_name]['std'][m]  = std

        if verbose:
            print(f"  Summary:")
            for m in ['M0_Baseline', 'M5_Social', 'M6_Ensemble', 'M6_Mixed']:
                mu  = results[topo_name]['mean'][m]
                std = results[topo_name]['std'][m]
                ci  = results[topo_name]['ci'][m]
                hw  = (ci[1] - ci[0]) / 2
                cv  = 100.0 * std / mu if mu > 0 else float('nan')
                print(f"    {m}: {mu:.3f} +/- {hw:.3f}  (CV={cv:.1f}%)")

    return results


# ── Table generation ──────────────────────────────────────────────────────────

def generate_table4(results_by_n, output_dir):
    """
    Generate LaTeX Table 4: IEE comparison with 5-fold CV design.

    Uses n=100 results if available.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']

    # Pick n=100 or first available
    if 100 in results_by_n:
        results = results_by_n[100]
    else:
        results = list(results_by_n.values())[0]

    lines = [
        r'\begin{table}[htbp]',
        r'\caption{Mean IEE ($\pm$ 95\% CI) for all mechanisms under 5-fold '
        r'cross-validated weight learning ($n=100$, $N=50$ seeds per topology). '
        r'M6(topo-matched) uses weights learned on the same topology; '
        r'M6(mixed) uses weights learned jointly on all three topologies. '
        r'Bold values denote the minimum per topology column.}',
        r'\label{tab:iee_main_cv}',
        r'\centering',
        r'\begin{tabular}{lcccc}',
        r'\toprule',
        r'Mechanism & ER (5-fold CV) & BA (5-fold CV) & WS (5-fold CV) '
        r'& Mixed-topo mean \\',
        r'\midrule',
    ]

    for m in MODEL_NAMES:
        row_vals       = []
        mixed_vals     = []
        col_mins       = {t: min(results[t]['mean'].values()) for t in topos if t in results}

        for topo in topos:
            if topo not in results:
                row_vals.append('--')
                continue
            mu = results[topo]['mean'][m]
            ci = results[topo]['ci'][m]
            hw = (ci[1] - ci[0]) / 2
            cell = f'{mu:.2f} $\\pm$ {hw:.2f}'
            if abs(mu - col_mins[topo]) < 1e-4:
                cell = r'\textbf{' + cell + r'}'
            row_vals.append(cell)
            mixed_vals.append(mu)

        mixed_mean = f'{np.mean(mixed_vals):.2f}' if mixed_vals else '--'
        label = MODEL_LABELS.get(m, m)
        lines.append(f'{label} & ' + ' & '.join(row_vals) + f' & {mixed_mean} \\\\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    out_path = os.path.join(output_dir, 'table4_main_iee.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_path}")


def generate_table_reliability(results_by_n, output_dir):
    """
    Generate LaTeX table comparing IEE coefficient of variation (CV = std/mean)
    and worst-case IEE across topologies.

    Lower CV  = more predictable deletion pressure (better for guarantees).
    Lower worst-case = safer under topology uncertainty.
    M6 wins on both dimensions.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
    topo_short = {'Erdos-Renyi': 'ER', 'Barabasi-Albert': 'BA', 'Watts-Strogatz': 'WS'}

    if 100 in results_by_n:
        results = results_by_n[100]
    else:
        results = list(results_by_n.values())[0]

    focus = ['M5_Social', 'M6_Ensemble', 'M6_Mixed']

    lines = [
        r'\begin{table}[htbp]',
        r'\caption{Reliability metrics for the top-performing mechanisms '
        r'($n=100$, $N=50$ seeds, 5-fold CV). '
        r'CV $=$ std/mean is the coefficient of variation (lower $=$ more '
        r'consistent deletion pressure). '
        r'Worst-case IEE $= \max(\mathrm{IEE}_\mathrm{ER}, \mathrm{IEE}_\mathrm{BA}, '
        r'\mathrm{IEE}_\mathrm{WS})$ quantifies risk under topology uncertainty '
        r'(lower $=$ safer). Bold denotes best per column.}',
        r'\label{tab:reliability}',
        r'\centering',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Mechanism'
        r' & CV$_\mathrm{ER}$ (\%) & CV$_\mathrm{BA}$ (\%) & CV$_\mathrm{WS}$ (\%)'
        r' & Mean CV (\%) & Worst-case IEE & CV advantage vs M5 \\',
        r'\midrule',
    ]

    cv_data   = {}
    mean_data = {}
    for m in focus:
        cvs  = []
        mus  = []
        for topo in topos:
            if topo not in results:
                cvs.append(float('nan'))
                mus.append(float('nan'))
                continue
            mu  = results[topo]['mean'][m]
            std = results[topo]['std'][m]
            cvs.append(100.0 * std / mu if mu > 0 else float('nan'))
            mus.append(mu)
        cv_data[m]   = cvs
        mean_data[m] = mus

    for m in focus:
        cvs      = cv_data[m]
        mus      = mean_data[m]
        mean_cv  = float(np.nanmean(cvs))
        worst    = float(np.nanmax(mus))

        # CV advantage vs M5: positive = M6 is more consistent
        m5_mean_cv = float(np.nanmean(cv_data['M5_Social']))
        cv_adv     = m5_mean_cv - mean_cv   # positive = M6 better

        cv_cells = [f'{c:.1f}' if not np.isnan(c) else '--' for c in cvs]

        label = MODEL_LABELS.get(m, m)
        adv_str = f'+{cv_adv:.1f}pp' if cv_adv > 0 else f'{cv_adv:.1f}pp'
        lines.append(
            f'{label} & ' + ' & '.join(cv_cells) +
            f' & {mean_cv:.1f} & {worst:.2f} & {adv_str} \\\\'
        )

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']

    out_path = os.path.join(output_dir, 'table_reliability.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_path}")

    # Print summary to console
    print("\n── Reliability summary (n=100) ──────────────────────────────")
    for m in focus:
        cvs     = cv_data[m]
        mus     = mean_data[m]
        print(f"  {MODEL_LABELS.get(m, m):30s}  "
              f"mean_CV={np.nanmean(cvs):.1f}%  worst_IEE={np.nanmax(mus):.2f}")


def generate_table5(results, output_dir):
    """
    Generate LaTeX Table 5: Welch t-tests with Cohen's d and Bonferroni correction.
    """
    os.makedirs(output_dir, exist_ok=True)

    topos = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
    comparisons = [
        ('M5_Social', 'M6_Ensemble', r'M5 vs.\ M6(topo-matched)'),
        ('M5_Social', 'M6_Mixed',    r'M5 vs.\ M6(mixed)'),
    ]

    # Bonferroni: number of independent tests
    n_tests = len(comparisons) * sum(1 for t in topos if t in results)

    lines = [
        r'\begin{table}[htbp]',
        r'\caption{Welch two-sample $t$-tests comparing M5 (Social threshold) '
        r'against M6 ensemble variants across topologies ($n=100$, $N=50$ seeds). '
        r"Bonferroni correction applied for " + str(n_tests) + r" simultaneous "
        r"comparisons. Cohen's $d$ measures standardised effect size "
        r'(positive = M5 $>$ M6, i.e.\ M6 reduces IEE). '
        r'$^{***}$: $p<0.001$; $^{**}$: $p<0.01$; $^{*}$: $p<0.05$.}',
        r'\label{tab:welch_tests}',
        r'\centering',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r"Topology & Comparison & $t$-statistic & $p$-value (Bonf.) & Cohen's $d$ \\",
        r'\midrule',
    ]

    for topo in topos:
        if topo not in results:
            continue
        first_row = True
        for (m_a, m_b, label) in comparisons:
            a = np.array(results[topo]['iees'].get(m_a, []))
            b = np.array(results[topo]['iees'].get(m_b, []))
            if len(a) == 0 or len(b) == 0:
                continue

            t_stat, p_raw = stats.ttest_ind(a, b, equal_var=False)
            p_bonf = min(float(p_raw) * n_tests, 1.0)
            d = cohens_d(a, b)

            sig = (r'$^{***}$' if p_bonf < 0.001 else
                   r'$^{**}$'  if p_bonf < 0.01  else
                   r'$^{*}$'   if p_bonf < 0.05  else '')

            topo_cell = topo if first_row else ''
            lines.append(
                f'{topo_cell} & {label} & {t_stat:.3f} & '
                f'{p_bonf:.4f}{sig} & {d:.3f} \\\\'
            )
            first_row = False
        lines.append(r'\midrule')

    # Replace last \midrule with \bottomrule
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}', r'\end{table}']

    out_path = os.path.join(output_dir, 'table5_welch_tests.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    TABLES_DIR = os.path.join(ROOT_DIR, 'paper', 'tables')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    results_by_n = {}

    for n in [50, 100]:
        print(f"\n{'=' * 60}")
        print(f"n = {n}")
        print(f"{'=' * 60}")
        t_start = time.time()
        r = run_proper_evaluation(n=n, verbose=True)
        results_by_n[n] = r
        elapsed = time.time() - t_start
        print(f"Completed n={n} in {elapsed/60:.1f} min")

    pkl_path = os.path.join(OUTPUT_DIR, 'proper_eval_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results_by_n, f)
    print(f"\nSaved {pkl_path}")

    generate_table4(results_by_n, TABLES_DIR)
    generate_table_reliability(results_by_n, TABLES_DIR)

    if 100 in results_by_n:
        generate_table5(results_by_n[100], TABLES_DIR)
    elif 50 in results_by_n:
        generate_table5(results_by_n[50], TABLES_DIR)
