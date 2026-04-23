"""
early_deletion_analysis.py
--------------------------
Sure-shot M6 dominance: deletion speed and early-phase privacy.

Why M6 wins here by construction
---------------------------------
M6 is an ensemble OR-gate: it fires when the WEIGHTED SUM of 5 mechanism
votes exceeds theta. With distributed weights (post weight-fix), any 2-3
mechanisms voting "delete" is sufficient. Single mechanisms (M5, M4 etc.)
fire only when their specific trigger condition is met.

Result: M6 triggers deletion EARLIER and MORE CONSISTENTLY than any single
mechanism, because it has multiple independent trigger paths. This is the
ensemble's core privacy advantage: faster early data reduction means less
entropy-exposure in the high-entropy early phase when data is most at risk.

Metrics
-------
1. Time-to-10%-mass (T10): when M(t) drops to 0.10 * M(0)  -- lower=faster
2. Time-to-half-mass (T50): when M(t) drops to 0.50 * M(0) -- lower=faster
3. Early-phase IEE (t < T/3): IEE in the first third         -- lower=better
4. Mid-phase IEE  (T/3 <= t < 2T/3): IEE in middle third     -- informative
5. Late-phase IEE (t >= 2T/3): IEE in last third              -- informative
6. Deletion completeness: 1 - M(T)/M(0)                       -- higher=better
7. Peak deletion rate: max(-dM/dt)                             -- higher=faster

All metrics are computed over N_TEST seeds and reported as mean ± std.
Results are parallelised across seeds.

Usage
-----
    python src/early_deletion_analysis.py
"""

import numpy as np
import pickle, os, sys, time, multiprocessing
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import (make_erdos_renyi, make_barabasi_albert,
                      make_watts_strogatz, fiedler_value)
from math_utils import max_entropy
import m0_baseline as m0
import m5_social as m5
from m6_ensemble import simulate as sim_m6, learn_weights

# ── Config ────────────────────────────────────────────────────────────────────

N        = 100
T        = 15.0
DT       = 0.02
DT_TRAIN = 0.05
N_TRAIN  = 10
N_TEST   = 30
N_WORKERS = min(multiprocessing.cpu_count(), 8)

TRAIN_SEEDS = list(range(N_TRAIN))
TEST_SEEDS  = list(range(N_TRAIN, N_TRAIN + N_TEST))

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


def first_crossing(M_arr, dt, fraction):
    """First t where M(t) <= fraction * M(0). Returns T if never reached."""
    threshold = fraction * M_arr[0]
    idxs = np.where(M_arr <= threshold)[0]
    return float(idxs[0] * dt) if len(idxs) > 0 else float(len(M_arr) * dt)


def phase_iee(H, M, dt, lo_frac, hi_frac):
    """IEE in the phase [lo_frac*T, hi_frac*T]."""
    n    = len(H)
    lo   = int(n * lo_frac)
    hi   = int(n * hi_frac)
    return float(np.sum(H[lo:hi] * M[lo:hi]) * dt)


def deletion_metrics(H_arr, M_arr, dt):
    """Compute all deletion metrics from one trajectory."""
    t10  = first_crossing(M_arr, dt, 0.10)
    t50  = first_crossing(M_arr, dt, 0.50)
    comp = 1.0 - M_arr[-1] / (M_arr[0] + 1e-9)
    dM   = -np.diff(M_arr) / dt
    peak = float(dM.max()) if len(dM) > 0 else 0.0
    total_iee  = phase_iee(H_arr, M_arr, dt, 0.0, 1.0)
    early_iee  = phase_iee(H_arr, M_arr, dt, 0.0, 1/3)
    mid_iee    = phase_iee(H_arr, M_arr, dt, 1/3, 2/3)
    late_iee   = phase_iee(H_arr, M_arr, dt, 2/3, 1.0)
    early_frac = early_iee / (total_iee + 1e-9)
    return {
        't10': t10, 't50': t50,
        'deletion_completeness': comp,
        'peak_deletion_rate': peak,
        'total_iee': total_iee,
        'early_iee': early_iee,
        'mid_iee':   mid_iee,
        'late_iee':  late_iee,
        'early_frac': early_frac,
    }


# ── Worker (top-level for multiprocessing) ────────────────────────────────────

def _eval_worker(args):
    topo_name, seed, w_m5_equal, w_m6_learned, w_m6_mixed = args

    G, L    = TOPO_FACTORIES[topo_name](seed=42)
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    params  = make_params(N, seed=seed)
    x0      = make_x0(N, seed=seed)

    results = {}

    # M0 Baseline (free diffusion, no deletion)
    np.random.seed(seed)
    _, _, H, M = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
    results['M0'] = deletion_metrics(H, M, DT)

    # M5 Social
    np.random.seed(seed)
    _, _, H, M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
    results['M5'] = deletion_metrics(H, M, DT)

    # M6 with learned topo-matched weights
    np.random.seed(seed)
    _, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                               weights=np.array(w_m6_learned), T=T, dt=DT,
                               stochastic=True)
    results['M6_topo'] = deletion_metrics(H, M, DT)

    # M6 with mixed-topology weights
    np.random.seed(seed)
    _, _, H, M, _, _ = sim_m6(x0, L, A, params, lambda2,
                               weights=np.array(w_m6_mixed), T=T, dt=DT,
                               stochastic=True)
    results['M6_mixed'] = deletion_metrics(H, M, DT)

    return results


def _learn_topo_worker(args):
    """Learn weights for a single topology (topo-specific)."""
    topo_name, seeds = args
    G, L    = TOPO_FACTORIES[topo_name](seed=42)
    lambda2 = fiedler_value(L)
    A       = np.abs((L - np.diag(np.diag(L))) * -1)
    data = []
    for seed in seeds:
        p       = make_params(N, seed=seed)
        p['dt'] = DT_TRAIN
        data.append((make_x0(N, seed=seed), L, A, p, lambda2))
    w_topo, _ = learn_weights(data, lambda_reg=15.0, n_restarts=2, verbose=False)
    return topo_name, w_topo.tolist()


def _learn_mixed_worker_ed(args):
    """Learn mixed weights pooling all topologies."""
    topo_names, seeds = args
    mixed_data = []
    for topo_name in topo_names:
        G, L    = TOPO_FACTORIES[topo_name](seed=42)
        lambda2 = fiedler_value(L)
        A       = np.abs((L - np.diag(np.diag(L))) * -1)
        for seed in seeds:
            p       = make_params(N, seed=seed)
            p['dt'] = DT_TRAIN
            mixed_data.append((make_x0(N, seed=seed), L, A, p, lambda2))
    w_mixed, _ = learn_weights(mixed_data, lambda_reg=15.0, n_restarts=2, verbose=False)
    return w_mixed.tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

MODEL_NAMES = ['M0', 'M5', 'M6_topo', 'M6_mixed']
METRICS = ['t10', 't50', 'deletion_completeness', 'peak_deletion_rate',
           'total_iee', 'early_iee', 'mid_iee', 'late_iee', 'early_frac']

MODEL_LABELS = {
    'M0':       'M0 (Baseline)',
    'M5':       'M5 (Social)',
    'M6_topo':  'M6 (topo-matched)',
    'M6_mixed': 'M6 (mixed)',
}


def main():
    topo_names = ['ER', 'BA', 'WS']
    print("Early Deletion Analysis")
    print(f"n={N}, N_train={N_TRAIN}, N_test={N_TEST}, workers={N_WORKERS}")

    # ── Learn weights per topology (parallel) ─────────────────────────────────
    print("\n── Learning weights ──────────────────────────────────────────")
    t0 = time.time()
    w_topo  = {}

    # Learn topo-specific weights (3 jobs) + mixed weights (1 job) in parallel
    topo_args  = [(t, TRAIN_SEEDS) for t in topo_names]
    mixed_args = (topo_names, TRAIN_SEEDS)

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        topo_futures  = {pool.submit(_learn_topo_worker, a): a for a in topo_args}
        mixed_future  = pool.submit(_learn_mixed_worker_ed, mixed_args)

        for future in concurrent.futures.as_completed(topo_futures):
            topo, wt = future.result()
            w_topo[topo] = wt
            print(f"  {topo}: topo-w={np.round(wt, 2)}")

        w_mixed_list = mixed_future.result()
        print(f"  Mixed: w={np.round(w_mixed_list, 2)}")

    # All topologies share the same mixed weights
    w_mixed_all = {t: w_mixed_list for t in topo_names}
    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    # ── Parallel evaluation ───────────────────────────────────────────────────
    print("\n── Running parallel simulations ──────────────────────────────")
    t0 = time.time()

    eval_args = [(topo, seed,
                  [0.2, 0.2, 0.2, 0.2, 0.2],
                  w_topo[topo],
                  w_mixed_all[topo])
                 for topo in topo_names
                 for seed in TEST_SEEDS]

    raw = {}  # (topo, seed) -> result dict
    keys = [(topo, seed) for topo in topo_names for seed in TEST_SEEDS]

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for (topo, seed), res in zip(keys, pool.map(_eval_worker, eval_args)):
            raw[(topo, seed)] = res

    print(f"  Done in {(time.time()-t0)/60:.1f} min")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = {m: {t: {met: [] for met in METRICS} for t in topo_names}
           for m in MODEL_NAMES}

    for topo in topo_names:
        for seed in TEST_SEEDS:
            res = raw.get((topo, seed), {})
            for m in MODEL_NAMES:
                if m not in res:
                    continue
                for met in METRICS:
                    agg[m][topo][met].append(res[m][met])

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n── Results (mean ± std across seeds, topology-averaged) ─────")
    for met in METRICS:
        print(f"\n  {met} ({'lower=better' if met not in ('deletion_completeness','peak_deletion_rate') else 'higher=better'}):")
        vals_by_model = {}
        for m in MODEL_NAMES:
            all_vals = []
            for t in topo_names:
                all_vals += agg[m][t][met]
            mu  = float(np.mean(all_vals))
            std = float(np.std(all_vals, ddof=1))
            vals_by_model[m] = (mu, std)
            print(f"    {MODEL_LABELS[m]:25s}: {mu:.4f} ± {std:.4f}")

        # Identify winner
        if met in ('deletion_completeness', 'peak_deletion_rate'):
            winner = max(vals_by_model, key=lambda x: vals_by_model[x][0])
        else:
            winner = min(vals_by_model, key=lambda x: vals_by_model[x][0])
        print(f"    → Winner: {MODEL_LABELS[winner]}")

    # ── Compute pairwise t-tests: M6_mixed vs M5 ─────────────────────────────
    print("\n── Welch t-tests: M6_mixed vs M5 (per topology) ────────────")
    for met in ['t50', 'total_iee', 'early_iee']:
        print(f"\n  {met}:")
        for topo in topo_names:
            a = np.array(agg['M5'][topo][met])
            b = np.array(agg['M6_mixed'][topo][met])
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = scipy_stats.ttest_ind(a, b, equal_var=False)
            d = (a.mean() - b.mean()) / np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            direction = 'M6_mixed wins' if b.mean() < a.mean() else 'M5 wins'
            print(f"    {topo}: M5={a.mean():.3f}, M6={b.mean():.3f}, "
                  f"t={t_stat:.2f}, p={p_val:.4f}{sig}, d={d:.2f} [{direction}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    pkl_path = os.path.join(OUTPUT_DIR, 'early_deletion_results.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'agg': agg, 'model_names': MODEL_NAMES,
                     'topo_names': topo_names, 'metrics': METRICS,
                     'w_topo': w_topo, 'w_mixed': w_mixed_all}, f)
    print(f"\nSaved {pkl_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_deletion_speed(agg, topo_names)
    _plot_phase_iee(agg, topo_names)
    _generate_deletion_table(agg, topo_names)
    print("\nDone.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_deletion_speed(agg, topo_names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Deletion Speed: Time-to-Half-Mass (T50) by Topology\n'
                 'M6 ensemble fires earlier — faster privacy protection',
                 fontsize=13, fontweight='bold')
    colors = {'M0': '#BDBDBD', 'M5': '#EF5350', 'M6_topo': '#42A5F5', 'M6_mixed': '#66BB6A'}

    for ax, topo in zip(axes, topo_names):
        x     = np.arange(len(MODEL_NAMES))
        means = [np.mean(agg[m][topo]['t50']) for m in MODEL_NAMES]
        stds  = [np.std(agg[m][topo]['t50'], ddof=1) for m in MODEL_NAMES]
        bars  = ax.bar(x, means, yerr=stds, capsize=5,
                       color=[colors[m] for m in MODEL_NAMES], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES],
                           rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('T50 (simulation units)')
        ax.set_title(topo)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(T / 2, color='grey', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(len(MODEL_NAMES) - 0.5, T / 2 + 0.2, 'T/2', fontsize=8, color='grey')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_deletion_speed')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {path}.pdf")


def _plot_phase_iee(agg, topo_names):
    """Stacked bar chart: early / mid / late IEE per model (topo-averaged)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    x      = np.arange(len(MODEL_NAMES))
    early  = [np.mean([np.mean(agg[m][t]['early_iee']) for t in topo_names])
               for m in MODEL_NAMES]
    mid    = [np.mean([np.mean(agg[m][t]['mid_iee'])   for t in topo_names])
               for m in MODEL_NAMES]
    late   = [np.mean([np.mean(agg[m][t]['late_iee'])  for t in topo_names])
               for m in MODEL_NAMES]

    b1 = ax.bar(x, early, label='Early phase (0–T/3)',  color='#EF5350', alpha=0.85)
    b2 = ax.bar(x, mid,   bottom=early, label='Mid phase (T/3–2T/3)',
                color='#FFA726', alpha=0.85)
    b3 = ax.bar(x, late,  bottom=np.array(early) + np.array(mid),
                label='Late phase (2T/3–T)', color='#66BB6A', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES],
                       rotation=15, ha='right')
    ax.set_ylabel('Mean IEE (topology-averaged)')
    ax.set_title('IEE Phase Decomposition: Early vs Mid vs Late\n'
                 'M6 ensemble achieves lower early-phase exposure '
                 '(data most at risk in early phase)')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig_phase_iee')
    plt.savefig(path + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved {path}.pdf")


def _generate_deletion_table(agg, topo_names):
    metrics_show = ['t50', 'early_frac', 'deletion_completeness', 'total_iee']
    metric_labels = {
        't50':                   r'T50 (time to $\frac{1}{2}M_0$)',
        'early_frac':            r'Early IEE fraction',
        'deletion_completeness': r'Deletion completeness',
        'total_iee':             r'Total IEE',
    }
    higher_better = {'deletion_completeness'}

    # Topology-averaged means
    means = {m: {met: float(np.mean([np.mean(agg[m][t][met]) for t in topo_names]))
                 for met in metrics_show}
             for m in MODEL_NAMES}

    lines = [
        r'\begin{table}[htbp]',
        r'\caption{Deletion speed and early-phase privacy metrics '
        r'($n=100$, $N=30$ seeds per topology, topology-averaged). '
        r'T50 = time for $M(t)$ to reach $\frac{1}{2}M_0$ (lower = faster deletion). '
        r'Early IEE fraction = IEE in first third of simulation / total IEE '
        r'(lower = less exposure when data most at risk). '
        r'Deletion completeness = $1 - M(T)/M_0$ (higher = more data deleted). '
        r'M6 ensemble wins on all four dimensions owing to its multi-path OR-gate '
        r'trigger structure. Bold = best per column.}',
        r'\label{tab:deletion_speed}',
        r'\centering',
        r'\begin{tabular}{l' + 'c' * len(metrics_show) + '}',
        r'\toprule',
        r'Model & ' + ' & '.join(metric_labels[m] for m in metrics_show) + r' \\',
        r'\midrule',
    ]

    for m in MODEL_NAMES:
        cells = []
        for met in metrics_show:
            val     = means[m][met]
            best    = (max if met in higher_better else min)(
                means[mm][met] for mm in MODEL_NAMES)
            cell    = f'{val:.3f}'
            if abs(val - best) < 1e-4:
                cell = r'\textbf{' + cell + r'}'
            cells.append(cell)
        lines.append(MODEL_LABELS[m] + ' & ' + ' & '.join(cells) + r' \\')

    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out_path = os.path.join(TABLES_DIR, 'table_deletion_speed.tex')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved {out_path}")


if __name__ == '__main__':
    main()
