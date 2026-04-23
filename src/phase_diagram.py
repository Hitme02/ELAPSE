"""
phase_diagram.py
----------------
SIMULATION 1: Phase diagram of IEE vs Hc/Hmax (percolation-like transition).
Sweeps Hc from 0.3*Hmax to 0.95*Hmax for ER, BA, WS topologies at n=100,200,500.
Saves fig9_phase_diagram.png and fig9_phase_diagram.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import make_erdos_renyi, make_barabasi_albert, make_watts_strogatz, fiedler_value
from math_utils import max_entropy
import m5_social as m5
from m6_ensemble import simulate as sim_m6

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

T   = 15.0
DT  = 0.02
N_SEEDS = 10
N_HC_STEPS = 15

SIZES = [100, 200, 500]

TOPOS = {
    'ER': lambda n, seed: make_erdos_renyi(n, p=0.15, seed=seed),
    'BA': lambda n, seed: make_barabasi_albert(n, m=3, seed=seed),
    'WS': lambda n, seed: make_watts_strogatz(n, k=6, p=0.1, seed=seed),
}
TOPO_COLORS = {'ER': '#2196F3', 'BA': '#E91E63', 'WS': '#4CAF50'}
TOPO_LABELS = {'ER': 'Erdős–Rényi', 'BA': 'Barabási–Albert', 'WS': 'Watts–Strogatz'}
SIZE_STYLES = {100: '-', 200: '--', 500: ':'}


def make_x0(n, seed):
    rng = np.random.default_rng(seed)
    x0 = np.zeros(n)
    idxs = rng.choice(n, max(1, n // 10), replace=False)
    x0[idxs] = rng.uniform(0.5, 1.0, len(idxs))
    return x0


def make_params(n, seed, Hc):
    H_max = max_entropy(n)
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    src = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': Hc, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def compute_iee(H_arr, M_arr, dt):
    return float(np.sum(H_arr * M_arr) * dt)


def run_phase_diagram():
    # results[topo][n] = (hc_fracs, m5_iees, m6_iees, crit_frac)
    results = {t: {} for t in TOPOS}

    for topo_name, make_fn in TOPOS.items():
        print(f"\nTopology: {topo_name}")
        for n in SIZES:
            print(f"  n={n}")
            H_max = max_entropy(n)
            hc_fracs = np.linspace(0.3, 0.95, N_HC_STEPS)
            hc_vals = hc_fracs * H_max

            m5_iees_mean = []
            m6_iees_mean = []

            # Build network once (seed=42)
            G, L = make_fn(n, 42)
            A = np.abs((L - np.diag(np.diag(L))) * -1)
            lambda2 = fiedler_value(L)
            w = np.ones(5) / 5  # equal weights for M6

            for hc in hc_vals:
                m5_run = []
                m6_run = []
                for seed in range(N_SEEDS):
                    np.random.seed(seed)
                    x0 = make_x0(n, seed)
                    params = make_params(n, seed, hc)

                    # M5
                    t_arr, _, H_arr, M_arr = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
                    m5_run.append(compute_iee(H_arr, M_arr, DT))

                    # M6
                    np.random.seed(seed)
                    t_arr, _, H_arr, M_arr, _, _ = sim_m6(
                        x0, L, A, params, lambda2, weights=w, T=T, dt=DT, stochastic=True)
                    m6_run.append(compute_iee(H_arr, M_arr, DT))

                m5_iees_mean.append(np.mean(m5_run))
                m6_iees_mean.append(np.mean(m6_run))
                print(f"    Hc/Hmax={hc/H_max:.2f}: M5={np.mean(m5_run):.3f} M6={np.mean(m6_run):.3f}")

            # Find critical threshold: steepest descent in M6
            m6_arr = np.array(m6_iees_mean)
            diffs = np.diff(m6_arr)
            crit_idx = int(np.argmin(diffs)) + 1  # index after steepest drop
            crit_frac = float(hc_fracs[crit_idx])

            results[topo_name][n] = (hc_fracs, m5_iees_mean, m6_iees_mean, crit_frac)

    return results


def plot_phase_diagram(results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Phase Diagram: IEE vs $H_c/H_{\\max}$ (Percolation-Like Transition)',
                 fontsize=13, fontweight='bold', y=1.02)

    for ax_idx, (topo_name, ax) in enumerate(zip(TOPOS.keys(), axes)):
        color = TOPO_COLORS[topo_name]
        for n in SIZES:
            hc_fracs, m5_iees, m6_iees, crit_frac = results[topo_name][n]
            ls = SIZE_STYLES[n]
            ax.plot(hc_fracs, m5_iees, color=color, linestyle=ls, alpha=0.6,
                    marker='o', markersize=3, label=f'M5 n={n}')
            ax.plot(hc_fracs, m6_iees, color=color, linestyle=ls, alpha=1.0,
                    marker='s', markersize=3, label=f'M6 n={n}')

            # Mark critical threshold for largest n only
            if n == 500:
                ax.axvline(crit_frac, color=color, linestyle=':', linewidth=1.5, alpha=0.7)
                ax.text(crit_frac + 0.01, ax.get_ylim()[1] * 0.05 if ax.get_ylim()[1] > 0 else 0.1,
                        f'$H^*_c$={crit_frac:.2f}', color=color, fontsize=8, va='bottom')

        ax.set_xlabel('$H_c / H_{\\max}$', fontsize=11)
        ax.set_ylabel('IEE', fontsize=11)
        ax.set_title(TOPO_LABELS[topo_name], fontsize=11)

        # Custom legend: n sizes only
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color='gray', linestyle='-', label='n=100'),
            Line2D([0], [0], color='gray', linestyle='--', label='n=200'),
            Line2D([0], [0], color='gray', linestyle=':', label='n=500'),
            Line2D([0], [0], color=color, linestyle='-', marker='o', markersize=5, label='M5'),
            Line2D([0], [0], color=color, linestyle='-', marker='s', markersize=5, label='M6'),
        ]
        ax.legend(handles=legend_elems, fontsize=7, loc='upper left', ncol=1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig9_phase_diagram.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig9_phase_diagram.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running phase diagram simulation...")
    results = run_phase_diagram()
    plot_phase_diagram(results)
    print("Done.")
