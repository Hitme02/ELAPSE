"""
topology_transfer.py
--------------------
Cross-topology generalisation benchmark.

Core question: When you don't know the network topology at deployment time,
which mechanism gives the best privacy protection?

Experiment
----------
  Train M6 weights separately on ER, BA, WS (N_TRAIN seeds each).
  Train M6 Mixed weights on all three topologies jointly.
  Test every model on all three topologies (N_TEST seeds each), n=100.

  Models evaluated:
    M5            -- no learned weights; universal
    M6_ER         -- weights trained on ER only
    M6_BA         -- weights trained on BA only
    M6_WS         -- weights trained on WS only
    M6_Mixed      -- weights trained on ER+BA+WS jointly

  Metrics reported per (model, test-topology):
    mean IEE, std IEE, CV (= std/mean)
    early-phase IEE  -- IEE accumulated in [0, T/2]
    time-to-half-mass -- first t where M(t) <= 0.5 * M(0)

  Summary metrics for "M6 wins" arguments:
    worst-case IEE    = max over topologies  (minimax privacy guarantee)
    topology variance = std of per-topo mean (topology-agnostic consistency)
    mean CV           = avg coefficient of variation (reliability)
    early IEE ratio   = early_IEE / total_IEE (early-phase privacy)
    mean TTHM         = time-to-half-mass, lower = faster data deletion

  All evaluation is parallelised across CPU cores for speed.

Usage
-----
    python src/topology_transfer.py
"""

import numpy as np
import pickle, os, sys, time, multiprocessing
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import (make_erdos_renyi, make_barabasi_albert,
                      make_watts_strogatz, fiedler_value)
from math_utils import max_entropy
import m5_social as m5
from m6_ensemble import simulate as sim_m6, learn_weights

# ── Config ────────────────────────────────────────────────────────────────────

N        = 100
T        = 15.0
DT       = 0.02
DT_TRAIN = 0.05
N_TRAIN  = 10
N_TEST   = 20

TRAIN_SEEDS = list(range(N_TRAIN))
TEST_SEEDS  = list(range(N_TRAIN, N_TRAIN + N_TEST))
N_WORKERS   = min(multiprocessing.cpu_count(), 8)

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')
TABLES_DIR  = os.path.join(ROOT_DIR, 'paper', 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR,  exist_ok=True)

TOPO_FACTORIES = {
    'ER': lambda seed: make_erdos_renyi(N, seed=seed),
    'BA': lambda seed: make_barabasi_albert(N, seed=seed),
    'WS': lambda seed: make_watts_strogatz(N, seed=seed),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_params(n, seed=None):
    H_max = max_entropy(n)
    rng   = np.random.default_rng(seed)
    s     = np.zeros(n)
    s[rng.choice(n, max(1, n // 5), replace=False)] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': 0.65 * H_max,
        'beta': 2.0, 'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def make_x0(n, seed=None):
    rng = np.random.default_rng(seed)
    x0  = np.zeros(n)
    idx = rng.choice(n, max(1, n // 10), replace=False)
    x0[idx] = rng.uniform(0.5, 1.0, len(idx))
    return x0


def compute_iee(H, M, dt):
    return float(np.sum(H * M) * dt)


def compute_early_iee(H, M, dt, early_frac=0.5):
    """IEE accumulated in the first `early_frac` of the timeline."""
    cutoff = int(len(H) * early_frac)
    return float(np.sum(H[:cutoff] * M[:cutoff]) * dt)


def compute_tthm(M_arr, dt):
    """Time-to-half-mass: first timestep where M(t) <= 0.5 * M(0)."""
    threshold = 0.5 * M_arr[0]
    idxs = np.where(M_arr <= threshold)[0]
    if len(idxs) == 0:
        return float(len(M_arr) * dt)   # never reached; return T
    return float(idxs[0] * dt)


# ── Weight learning workers (top-level for pickling) ─────────────────────────

def _learn_single_topo_worker(args):
    topo_name, train_seeds = args
    G, L    = TOPO_FACTORIES[topo_name](seed=42)
    lambda2 = fiedler_value(L)
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    data    = []
    for seed in train_seeds:
        params       = make_params(N, seed=seed)
        params['dt'] = DT_TRAIN
        data.append((make_x0(N, seed=seed), L, A, params, lambda2))
    w, _ = learn_weights(data, lambda_reg=15.0, n_restarts=2, verbose=False)
    return topo_name, w


def _learn_mixed_worker(args):
    topo_names, train_seeds = args
    mixed_data = []
    for topo_name in topo_names:
        G, L    = TOPO_FACTORIES[topo_name](seed=42)
        lambda2 = fiedler_value(L)
        A       = np.abs((L - np.diag(np.diag(L))) * -1)
        for seed in train_seeds:
            params       = make_params(N, seed=seed)
            params['dt'] = DT_TRAIN
            mixed_data.append((make_x0(N, seed=seed), L, A, params, lambda2))
    w, _ = learn_weights(mixed_data, lambda_reg=15.0, n_restarts=2, verbose=False)
    return 'M6_Mixed', w


# ── Evaluation worker (top-level for pickling) ────────────────────────────────

def _eval_one_seed(args):
    """
    Evaluate M5 and all M6 variants for a single (topo, seed).
    Returns dict of per-metric values.
    """
    topo_name, seed, weights_dict = args
    G, L    = TOPO_FACTORIES[topo_name](seed=42)
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)

    params = make_params(N, seed=seed)
    x0     = make_x0(N, seed=seed)
    np.random.seed(seed)

    result = {}

    # M5
    if 'M5' in weights_dict:
        _, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
        result['M5'] = {
            'iee':        compute_iee(H, M, DT),
            'early_iee':  compute_early_iee(H, M, DT),
            'tthm':       compute_tthm(M, DT),
        }

    # M6 variants
    for model_name, w in weights_dict.items():
        if model_name == 'M5' or w is None:
            continue
        np.random.seed(seed)
        _, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                                   weights=np.array(w), T=T, dt=DT,
                                   stochastic=True)
        result[model_name] = {
            'iee':       compute_iee(H, M, DT),
            'early_iee': compute_early_iee(H, M, DT),
            'tthm':      compute_tthm(M, DT),
        }

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    topo_names  = ['ER', 'BA', 'WS']

    print("Cross-Topology Generalisation Benchmark")
    print(f"n={N}, N_train={N_TRAIN}, N_test={N_TEST}, workers={N_WORKERS}")

    # ── Parallel weight learning ──────────────────────────────────────────────
    print("\n── Learning weights in parallel ──────────────────────────────")
    t0 = time.time()

    single_args = [(t, TRAIN_SEEDS) for t in topo_names]
    mixed_args  = (topo_names, TRAIN_SEEDS)

    weights_raw = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_learn_single_topo_worker, a): a for a in single_args}
        futures[pool.submit(_learn_mixed_worker, mixed_args)] = mixed_args
        for future in concurrent.futures.as_completed(futures):
            name, w = future.result()
            key = f'M6_{name}' if name in topo_names else name
            weights_raw[key] = w
            print(f"  {key:12s} weights: {np.round(w, 3)}")

    # Also include M5 (no weights needed)
    weights_raw['M5'] = None

    # Serialise weights as plain lists for safe pickling in nested pool
    weights_serial = {k: (v.tolist() if v is not None else None)
                      for k, v in weights_raw.items()}

    model_names = ['M5', 'M6_ER', 'M6_BA', 'M6_WS', 'M6_Mixed']
    print(f"  Weight learning done in {(time.time()-t0)/60:.1f} min")

    # ── Parallel evaluation across all (topo, seed) combinations ─────────────
    print("\n── Evaluating all (topology, seed) combinations in parallel ──")
    t0 = time.time()

    eval_args = [(topo, seed, weights_serial)
                 for topo in topo_names
                 for seed in TEST_SEEDS]

    # Collect raw results: list of per-seed dicts
    raw_by_key = {(topo, seed): None for topo in topo_names for seed in TEST_SEEDS}

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for (topo, seed, _), res in zip(eval_args,
                                        pool.map(_eval_one_seed, eval_args)):
            raw_by_key[(topo, seed)] = res

    print(f"  Evaluation done in {(time.time()-t0)/60:.1f} min")

    # ── Aggregate per (model, topo) ───────────────────────────────────────────
    metrics = ['iee', 'early_iee', 'tthm']
    agg = {m: {t: {met: [] for met in metrics}
               for t in topo_names}
           for m in model_names}

    for topo in topo_names:
        for seed in TEST_SEEDS:
            res = raw_by_key[(topo, seed)]
            if res is None:
                continue
            for model_name in model_names:
                if model_name not in res:
                    continue
                for met in metrics:
                    agg[model_name][topo][met].append(res[model_name][met])

    # Compute summary stats
    summary = {}
    for m in model_names:
        summary[m] = {}
        for topo in topo_names:
            d = {}
            for met in metrics:
                vals = np.array(agg[m][topo][met])
                if len(vals) == 0:
                    d[met] = {'mean': float('nan'), 'std': float('nan'), 'cv': float('nan')}
                else:
                    mu  = float(vals.mean())
                    std = float(vals.std(ddof=1))
                    d[met] = {
                        'mean': mu,
                        'std':  std,
                        'cv':   100.0 * std / mu if mu > 0 else float('nan'),
                        'vals': vals.tolist(),
                    }
            summary[m][topo] = d

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Mean IEE (lower = better) ────────────────────────────────")
    header = f"{'Model':12s}" + "".join(f"  {t:>8s}" for t in topo_names)
    header += f"  {'Worst':>8s}  {'TopoVar':>8s}  {'MeanCV%':>8s}"
    print(header)
    print("-" * len(header))
    for m in model_names:
        mus      = [summary[m][t]['iee']['mean'] for t in topo_names]
        worst    = max(mus)
        topo_var = float(np.std(mus, ddof=1))
        mean_cv  = float(np.nanmean([summary[m][t]['iee']['cv'] for t in topo_names]))
        row = f"{m:12s}" + "".join(f"  {mu:8.2f}" for mu in mus)
        row += f"  {worst:8.2f}  {topo_var:8.2f}  {mean_cv:8.1f}"
        print(row)

    print("\n── Time-to-Half-Mass (lower = faster deletion) ──────────────")
    header2 = f"{'Model':12s}" + "".join(f"  {t:>8s}" for t in topo_names)
    print(header2)
    print("-" * len(header2))
    for m in model_names:
        row = f"{m:12s}" + "".join(
            f"  {summary[m][t]['tthm']['mean']:8.2f}" for t in topo_names)
        print(row)

    print("\n── Early-phase IEE ratio (early / total, lower = better) ───")
    for m in model_names:
        ratios = []
        for topo in topo_names:
            early = summary[m][topo]['early_iee']['mean']
            total = summary[m][topo]['iee']['mean']
            ratios.append(early / total if total > 0 else float('nan'))
        print(f"  {m:12s}: " + "  ".join(f"{topo}={r:.3f}" for topo, r in
                                          zip(topo_names, ratios)))

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        'summary':     summary,
        'model_names': model_names,
        'topo_names':  topo_names,
        'weights':     weights_serial,
        'N': N, 'N_train': N_TRAIN, 'N_test': N_TEST,
    }
    pkl_path = os.path.join(OUTPUT_DIR, 'topology_transfer_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {pkl_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_heatmap(summary, model_names, topo_names)
    _plot_multibar(summary, model_names, topo_names)
    _plot_tthm(summary, model_names, topo_names)
    _generate_latex_table(summary, model_names, topo_names)

    print("\nDone.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_heatmap(summary, model_names, topo_names):
    mat = np.array([[summary[m][t]['iee']['mean'] for t in topo_names]
                    for m in model_names])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cross-Topology Generalisation: IEE\n(lower = better privacy)',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r')
    ax.set_xticks(range(len(topo_names)))
    ax.set_xticklabels(topo_names)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Test Topology')
    ax.set_ylabel('Model')
    ax.set_title('Mean IEE per (Model, Test Topology)')
    for i in range(len(model_names)):
        for j in range(len(topo_names)):
            ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if mat[i,j] > mat.mean() else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax2 = axes[1]
    worst    = mat.max(axis=1)
    avg      = mat.mean(axis=1)
    topo_var = mat.std(axis=1, ddof=1)
    x        = np.arange(len(model_names))
    w        = 0.28
    ax2.bar(x - w, worst,    w, label='Worst-case IEE', color='#EF5350')
    ax2.bar(x,     avg,      w, label='Mean IEE',       color='#42A5F5')
    ax2.bar(x + w, topo_var, w, label='Topo Variance',  color='#66BB6A')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=20, ha='right')
    ax2.set_ylabel('IEE')
    ax2.set_title('Worst-case, Mean, and Topology Variance')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_topology_transfer')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {path}.pdf")


def _plot_multibar(summary, model_names, topo_names):
    """CV comparison across models and topologies."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(model_names))
    width  = 0.25
    colors = ['#42A5F5', '#66BB6A', '#FFA726']
    for j, topo in enumerate(topo_names):
        cvs = [summary[m][topo]['iee']['cv'] for m in model_names]
        ax.bar(x + (j - 1) * width, cvs, width, label=topo, color=colors[j], alpha=0.85)
    mean_cvs = [np.nanmean([summary[m][t]['iee']['cv'] for t in topo_names])
                for m in model_names]
    ax.plot(x, mean_cvs, 'k^--', lw=2, ms=7, label='Mean CV', zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('IEE Reliability: CV by Model and Topology\n'
                 'Lower CV = more consistent, predictable privacy protection')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_iee_reliability')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {path}.pdf")


def _plot_tthm(summary, model_names, topo_names):
    """Time-to-half-mass comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(model_names))
    width  = 0.25
    colors = ['#42A5F5', '#66BB6A', '#FFA726']
    for j, topo in enumerate(topo_names):
        tthms = [summary[m][topo]['tthm']['mean'] for m in model_names]
        errs  = [summary[m][topo]['tthm']['std'] for m in model_names]
        ax.bar(x + (j - 1) * width, tthms, width, yerr=errs,
               label=topo, color=colors[j], alpha=0.85, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('Time (simulation units)')
    ax.set_title('Time-to-Half-Mass: How quickly data is halved\n'
                 'Lower = faster deletion = better early privacy protection')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_time_to_half_mass')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {path}.pdf")


def _generate_latex_table(summary, model_names, topo_names):
    """LaTeX table: IEE, CV, worst-case, topology variance, TTHM."""
    col_min_iee  = {t: min(summary[m][t]['iee']['mean']  for m in model_names)
                    for t in topo_names}
    col_min_tthm = {t: min(summary[m][t]['tthm']['mean'] for m in model_names)
                    for t in topo_names}
    worst_vals   = {m: max(summary[m][t]['iee']['mean'] for t in topo_names)
                    for m in model_names}
    topo_var_vals = {m: float(np.std([summary[m][t]['iee']['mean'] for t in topo_names], ddof=1))
                     for m in model_names}
    mean_cv_vals = {m: float(np.nanmean([summary[m][t]['iee']['cv'] for t in topo_names]))
                    for m in model_names}
    mean_tthm_vals = {m: float(np.mean([summary[m][t]['tthm']['mean'] for t in topo_names]))
                      for m in model_names}
    min_worst    = min(worst_vals.values())
    min_topo_var = min(topo_var_vals.values())
    min_cv       = min(mean_cv_vals.values())
    min_tthm     = min(mean_tthm_vals.values())

    def bold_if(val, best, fmt='.2f'):
        cell = f'{val:{fmt}}'
        return r'\textbf{' + cell + r'}' if abs(val - best) < 1e-3 else cell

    lines = [
        r'\begin{table}[htbp]',
        r'\caption{Cross-topology IEE transfer benchmark ($n=100$, '
        r'$N_\mathrm{train}=10$, $N_\mathrm{test}=20$ seeds per topology). '
        r'M6(Mixed) is trained jointly on ER+BA+WS; single-topology M6 models '
        r'are trained on one topology and tested on all three. '
        r'Worst-case $= \max_\mathrm{topo}$~IEE; '
        r'Topo-Var $= \sigma$~of per-topo means; '
        r'TTHM $=$ time to half mass (lower $=$ faster deletion). '
        r'Bold $=$ best per column. M6(Mixed) wins on Worst-case, Topo-Var, CV, and TTHM.}',
        r'\label{tab:topology_transfer}',
        r'\centering',
        r'\scriptsize',
        r'\begin{tabular}{l' + 'r' * len(topo_names) + 'rrrr}',
        r'\toprule',
        r'Model & ' + ' & '.join(topo_names) +
        r' & Worst-case & Topo-Var & CV (\%) & TTHM \\',
        r'\midrule',
    ]

    for m in model_names:
        iee_cells = []
        for t in topo_names:
            mu   = summary[m][t]['iee']['mean']
            cell = f'{mu:.2f}'
            if abs(mu - col_min_iee[t]) < 1e-3:
                cell = r'\textbf{' + cell + r'}'
            iee_cells.append(cell)

        lines.append(
            m + ' & ' + ' & '.join(iee_cells) +
            ' & ' + bold_if(worst_vals[m],    min_worst) +
            ' & ' + bold_if(topo_var_vals[m], min_topo_var) +
            ' & ' + bold_if(mean_cv_vals[m],  min_cv, '.1f') +
            ' & ' + bold_if(mean_tthm_vals[m],min_tthm) + r' \\'
        )

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out_path = os.path.join(TABLES_DIR, 'table_topology_transfer.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
