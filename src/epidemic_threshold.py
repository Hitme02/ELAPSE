"""
epidemic_threshold.py
---------------------
SIMULATION 3: M2 IEE vs basic reproduction number R0 = beta_sir/gamma.
Sweeps R0 from 0.5 to 5.0 in 10 steps, fixing gamma=0.15, varying beta_sir.
n=200, ER and BA, 10 seeds.
Saves fig11_epidemic_threshold.png and fig11_epidemic_threshold.pdf
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
import m2_epidemic as m2

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

T = 15.0
DT = 0.02
N = 200
N_SEEDS = 10
GAMMA = 0.15

# R0 from 0.5 to 5.0 in 10 steps => beta_sir = R0 * gamma
R0_VALUES = np.linspace(0.5, 5.0, 10)
BETA_VALUES = R0_VALUES * GAMMA


def make_x0(n, seed):
    rng = np.random.default_rng(seed)
    x0 = np.zeros(n)
    idxs = rng.choice(n, max(1, n // 10), replace=False)
    x0[idxs] = rng.uniform(0.5, 1.0, len(idxs))
    return x0


def make_params(n, seed, beta_sir):
    H_max = max_entropy(n)
    H_c = 0.65 * H_max
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    src = rng.choice(n, max(1, n // 5), replace=False)
    s[src] = 0.05
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': beta_sir, 'gamma': GAMMA,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def compute_iee(H_arr, M_arr, dt):
    return float(np.sum(H_arr * M_arr) * dt)


def run_epidemic_threshold():
    results = {}

    for topo_name, make_fn in [('ER', lambda seed: make_erdos_renyi(N, p=0.15, seed=seed)),
                                ('BA', lambda seed: make_barabasi_albert(N, m=3, seed=seed))]:
        print(f"\nTopology: {topo_name}")
        G, L = make_fn(42)
        A = np.abs((L - np.diag(np.diag(L))) * -1)

        iee_means = []
        iee_stds = []

        for r0, beta in zip(R0_VALUES, BETA_VALUES):
            iees = []
            for seed in range(N_SEEDS):
                np.random.seed(seed)
                x0 = make_x0(N, seed)
                params = make_params(N, seed, beta)
                t_arr, _, H_arr, M_arr = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=True)
                iees.append(compute_iee(H_arr, M_arr, DT))

            iee_means.append(np.mean(iees))
            iee_stds.append(np.std(iees))
            print(f"  R0={r0:.2f} (beta={beta:.3f}): IEE={np.mean(iees):.3f} ± {np.std(iees):.3f}")

        results[topo_name] = {
            'r0': R0_VALUES,
            'iee_mean': np.array(iee_means),
            'iee_std': np.array(iee_stds),
        }

    return results


def plot_epidemic_threshold(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {'ER': '#2196F3', 'BA': '#E91E63'}
    labels = {'ER': 'Erdős–Rényi', 'BA': 'Barabási–Albert'}

    for topo_name, data in results.items():
        r0 = data['r0']
        mean = data['iee_mean']
        std = data['iee_std']
        color = colors[topo_name]

        ax.plot(r0, mean, 'o-', color=color, linewidth=2, markersize=6,
                label=labels[topo_name])
        ax.fill_between(r0, mean - std, mean + std, color=color, alpha=0.15)

    # Mark epidemic threshold R0=1
    ax.axvline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='$R_0=1$ (epidemic threshold)')
    ymin, ymax = ax.get_ylim()
    ax.text(1.05, ymin + 0.05 * (ymax - ymin), '$R_0=1$', fontsize=10, color='black', va='bottom')

    ax.set_xlabel('Basic Reproduction Number $R_0 = \\beta_{SIR}/\\gamma$', fontsize=12)
    ax.set_ylabel('IEE (Information Entropy Exposure)', fontsize=12)
    ax.set_title('M2 IEE vs Epidemic Threshold $R_0$\n($n=200$, $\\gamma=0.15$, 10 seeds)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.4, 5.2])

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig11_epidemic_threshold.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig11_epidemic_threshold.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running epidemic threshold simulation...")
    results = run_epidemic_threshold()
    plot_epidemic_threshold(results)
    print("Done.")
