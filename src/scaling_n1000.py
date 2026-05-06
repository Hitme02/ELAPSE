"""
scaling_n1000.py
----------------
SIMULATION 5: n=1000 scaling validation.
Runs M5 and M6 for ER and BA at n=1000 (10 seeds).
Combines with smaller n data (recomputed) and fits power-law IEE ~ n^alpha.
Saves fig13_scaling_n1000.png and fig13_scaling_n1000.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import make_erdos_renyi, make_barabasi_albert, fiedler_value
from math_utils import max_entropy
import m5_social as m5
from m6_ensemble import simulate as sim_m6

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

T = 15.0
DT = 0.02
N_SEEDS = 10

# Sizes: existing + new n=1000
SIZES_SMALL = [50, 100, 200, 500]
SIZE_NEW = 1000
ALL_SIZES = SIZES_SMALL + [SIZE_NEW]


def make_x0(n, seed):
    rng = np.random.default_rng(seed)
    x0 = np.zeros(n)
    idxs = rng.choice(n, max(1, n // 10), replace=False)
    x0[idxs] = rng.uniform(0.5, 1.0, len(idxs))
    return x0


def make_params(n, seed):
    H_max = max_entropy(n)
    H_c = 0.65 * H_max
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    src = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def compute_iee(H_arr, M_arr, dt):
    return float(np.sum(H_arr * M_arr) * dt)


def run_single_n(n, topo_name, make_fn, seeds=None):
    if seeds is None:
        seeds = range(N_SEEDS)
    G, L = make_fn(n, 42)
    A = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    w = np.ones(5) / 5

    m5_iees = []
    m6_iees = []

    for seed in seeds:
        np.random.seed(seed)
        x0 = make_x0(n, seed)
        params = make_params(n, seed)

        t_arr, _, H_arr, M_arr = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
        m5_iees.append(compute_iee(H_arr, M_arr, DT))

        np.random.seed(seed)
        t_arr, _, H_arr, M_arr, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w, T=T, dt=DT, stochastic=True)
        m6_iees.append(compute_iee(H_arr, M_arr, DT))

    return np.mean(m5_iees), np.std(m5_iees), np.mean(m6_iees), np.std(m6_iees)


def run_scaling():
    results = {}

    TOPOS = {
        'ER': lambda n, seed: make_erdos_renyi(n, p=0.15, seed=seed),
        'BA': lambda n, seed: make_barabasi_albert(n, m=3, seed=seed),
    }

    for topo_name, make_fn in TOPOS.items():
        print(f"\nTopology: {topo_name}")
        m5_means = []
        m5_stds = []
        m6_means = []
        m6_stds = []

        for n in ALL_SIZES:
            print(f"  n={n}...", end=' ', flush=True)
            # For n=1000, use fewer seeds if slow
            seeds = range(N_SEEDS) if n <= 500 else range(N_SEEDS)
            m5m, m5s, m6m, m6s = run_single_n(n, topo_name, make_fn, seeds)
            m5_means.append(m5m)
            m5_stds.append(m5s)
            m6_means.append(m6m)
            m6_stds.append(m6s)
            print(f"M5={m5m:.3f} M6={m6m:.3f}")

        results[topo_name] = {
            'n': ALL_SIZES,
            'm5_mean': np.array(m5_means),
            'm5_std': np.array(m5_stds),
            'm6_mean': np.array(m6_means),
            'm6_std': np.array(m6_stds),
        }

    return results


def power_law_fit(n_arr, iee_arr):
    """Fit IEE ~ n^alpha via log-linear regression. Returns (alpha, c)."""
    log_n = np.log(n_arr)
    log_iee = np.log(np.maximum(iee_arr, 1e-10))
    coeffs = np.polyfit(log_n, log_iee, 1)
    alpha = coeffs[0]
    c = np.exp(coeffs[1])
    return alpha, c


def plot_scaling(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle('Scaling Validation: IEE vs Network Size $n$ (including $n=1000$)',
                 fontsize=13, fontweight='bold')

    colors_m5 = {'ER': '#64B5F6', 'BA': '#F48FB1'}
    colors_m6 = {'ER': '#1565C0', 'BA': '#AD1457'}
    labels = {'ER': 'Erdős–Rényi', 'BA': 'Barabási–Albert'}

    for ax_idx, (topo_name, ax) in enumerate(zip(['ER', 'BA'], axes)):
        data = results[topo_name]
        ns = np.array(data['n'])
        m5_mean = data['m5_mean']
        m5_std = data['m5_std']
        m6_mean = data['m6_mean']
        m6_std = data['m6_std']

        # Plot data
        ax.errorbar(ns, m5_mean, yerr=m5_std, fmt='o-', color=colors_m5[topo_name],
                    label='M5 (Social)', capsize=4, linewidth=2, markersize=6)
        ax.errorbar(ns, m6_mean, yerr=m6_std, fmt='s-', color=colors_m6[topo_name],
                    label='M6 (ELAPSE)', capsize=4, linewidth=2, markersize=6)

        # Mark n=1000 point
        ax.axvline(1000, color='gray', linestyle=':', linewidth=1, alpha=0.6)
        ax.text(0.97, 0.03, 'n=1000\n(new)', fontsize=8, color='gray',
                va='bottom', ha='right', transform=ax.transAxes)

        # Power-law fits
        alpha_m5, c_m5 = power_law_fit(ns, m5_mean)
        alpha_m6, c_m6 = power_law_fit(ns, m6_mean)

        n_fit = np.logspace(np.log10(ns[0]), np.log10(ns[-1]), 100)
        ax.plot(n_fit, c_m5 * n_fit**alpha_m5, '--', color=colors_m5[topo_name],
                alpha=0.5, linewidth=1.5, label=f'M5 fit: $n^{{{alpha_m5:.2f}}}$')
        ax.plot(n_fit, c_m6 * n_fit**alpha_m6, '--', color=colors_m6[topo_name],
                alpha=0.5, linewidth=1.5, label=f'M6 fit: $n^{{{alpha_m6:.2f}}}$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Network Size $n$', fontsize=11)
        ax.set_ylabel('IEE', fontsize=11)
        ax.set_title(f'{labels[topo_name]}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # ── Crossover n* where M6 definitively beats M5 ─────────────────
        if c_m5 > 0 and c_m6 > 0 and alpha_m5 != alpha_m6:
            # c_m5 * n^alpha_m5 = c_m6 * n^alpha_m6
            # => n* = (c_m5/c_m6)^(1/(alpha_m6-alpha_m5))
            exp_diff = alpha_m6 - alpha_m5
            if exp_diff != 0:
                n_star = (c_m5 / c_m6) ** (1.0 / exp_diff)
                if alpha_m6 < alpha_m5:
                    print(f"{topo_name}: M6 scaling exponent LOWER ({alpha_m6:.3f} < {alpha_m5:.3f})")
                    print(f"  → M6 beats M5 for n > {n_star:.0f}  (crossover n*={n_star:.1f})")
                    if n_star <= ns[-1]:
                        ax.axvline(n_star, color=colors_m6[topo_name], linestyle=':',
                                   linewidth=2, alpha=0.8)
                        ax.text(n_star * 1.05,
                                c_m5 * n_star ** alpha_m5 * 0.85,
                                f'n*={n_star:.0f}', fontsize=9,
                                color=colors_m6[topo_name], fontweight='bold')
                else:
                    print(f"{topo_name}: M5 scaling exponent lower; no crossover in tested range")
        print(f"{topo_name}: M5 alpha={alpha_m5:.3f}, M6 alpha={alpha_m6:.3f}")

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig13_scaling_n1000.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig13_scaling_n1000.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running n=1000 scaling validation...")
    results = run_scaling()
    plot_scaling(results)
    print("Done.")
