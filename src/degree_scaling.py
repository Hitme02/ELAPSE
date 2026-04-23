"""
degree_scaling.py
-----------------
SIMULATION 2: IEE vs mean degree across topology families.
BA networks m=1..5, ER networks p=0.05..0.25, n=200, 10 seeds.
Saves fig10_degree_scaling.png and fig10_degree_scaling.pdf
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
N = 200
N_SEEDS = 10


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


def run_degree_scaling():
    n = N

    # BA: m=1,2,3,4,5  => mean degree ~ 2m
    ba_params_list = [1, 2, 3, 4, 5]
    ba_mean_degrees = []
    ba_m5_means = []
    ba_m6_means = []
    ba_m5_stds = []
    ba_m6_stds = []

    print("BA networks:")
    for m in ba_params_list:
        G, L = make_barabasi_albert(n, m=m, seed=42)
        A = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)
        mean_deg = 2 * G.number_of_edges() / G.number_of_nodes()
        ba_mean_degrees.append(mean_deg)
        w = np.ones(5) / 5

        m5_iees = []
        m6_iees = []
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            x0 = make_x0(n, seed)
            params = make_params(n, seed)

            t_arr, _, H_arr, M_arr = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
            m5_iees.append(compute_iee(H_arr, M_arr, DT))

            np.random.seed(seed)
            t_arr, _, H_arr, M_arr, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w, T=T, dt=DT, stochastic=True)
            m6_iees.append(compute_iee(H_arr, M_arr, DT))

        ba_m5_means.append(np.mean(m5_iees))
        ba_m6_means.append(np.mean(m6_iees))
        ba_m5_stds.append(np.std(m5_iees))
        ba_m6_stds.append(np.std(m6_iees))
        print(f"  m={m} <k>={mean_deg:.1f}: M5={np.mean(m5_iees):.3f} M6={np.mean(m6_iees):.3f}")

    # ER: p=0.05,0.10,0.15,0.20,0.25 => mean degree ~ p*(n-1)
    er_p_list = [0.05, 0.10, 0.15, 0.20, 0.25]
    er_mean_degrees = []
    er_m5_means = []
    er_m6_means = []
    er_m5_stds = []
    er_m6_stds = []

    print("ER networks:")
    for p in er_p_list:
        G, L = make_erdos_renyi(n, p=p, seed=42)
        A = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)
        mean_deg = 2 * G.number_of_edges() / G.number_of_nodes()
        er_mean_degrees.append(mean_deg)
        w = np.ones(5) / 5

        m5_iees = []
        m6_iees = []
        for seed in range(N_SEEDS):
            np.random.seed(seed)
            x0 = make_x0(n, seed)
            params = make_params(n, seed)

            t_arr, _, H_arr, M_arr = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
            m5_iees.append(compute_iee(H_arr, M_arr, DT))

            np.random.seed(seed)
            t_arr, _, H_arr, M_arr, _, _ = sim_m6(x0, L, A, params, lambda2, weights=w, T=T, dt=DT, stochastic=True)
            m6_iees.append(compute_iee(H_arr, M_arr, DT))

        er_m5_means.append(np.mean(m5_iees))
        er_m6_means.append(np.mean(m6_iees))
        er_m5_stds.append(np.std(m5_iees))
        er_m6_stds.append(np.std(m6_iees))
        print(f"  p={p} <k>={mean_deg:.1f}: M5={np.mean(m5_iees):.3f} M6={np.mean(m6_iees):.3f}")

    return (ba_mean_degrees, ba_m5_means, ba_m5_stds, ba_m6_means, ba_m6_stds,
            er_mean_degrees, er_m5_means, er_m5_stds, er_m6_means, er_m6_stds)


def plot_degree_scaling(data):
    (ba_deg, ba_m5, ba_m5_std, ba_m6, ba_m6_std,
     er_deg, er_m5, er_m5_std, er_m6, er_m6_std) = data

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('IEE vs Mean Degree $\\langle k \\rangle$ Across Topology Families\n(n=200, 10 seeds)',
                 fontsize=13, fontweight='bold')

    # BA panel
    ax = axes[0]
    ax.errorbar(ba_deg, ba_m5, yerr=ba_m5_std, fmt='o-', color='#E91E63',
                label='M5 (Social)', capsize=4, linewidth=2, markersize=6)
    ax.errorbar(ba_deg, ba_m6, yerr=ba_m6_std, fmt='s-', color='#9C27B0',
                label='M6 (ELAPSE)', capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel('Mean Degree $\\langle k \\rangle$', fontsize=11)
    ax.set_ylabel('IEE', fontsize=11)
    ax.set_title('Barabási–Albert (m=1,2,3,4,5)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate m values
    for i, (x, y, m) in enumerate(zip(ba_deg, ba_m6, [1, 2, 3, 4, 5])):
        ax.annotate(f'm={m}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)

    # ER panel
    ax = axes[1]
    ax.errorbar(er_deg, er_m5, yerr=er_m5_std, fmt='o-', color='#2196F3',
                label='M5 (Social)', capsize=4, linewidth=2, markersize=6)
    ax.errorbar(er_deg, er_m6, yerr=er_m6_std, fmt='s-', color='#00BCD4',
                label='M6 (ELAPSE)', capsize=4, linewidth=2, markersize=6)
    ax.set_xlabel('Mean Degree $\\langle k \\rangle$', fontsize=11)
    ax.set_ylabel('IEE', fontsize=11)
    ax.set_title('Erdős–Rényi (p=0.05,0.10,0.15,0.20,0.25)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate p values
    for i, (x, y, p) in enumerate(zip(er_deg, er_m6, [0.05, 0.10, 0.15, 0.20, 0.25])):
        ax.annotate(f'p={p}', (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig10_degree_scaling.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig10_degree_scaling.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running degree scaling simulation...")
    data = run_degree_scaling()
    plot_degree_scaling(data)
    print("Done.")
