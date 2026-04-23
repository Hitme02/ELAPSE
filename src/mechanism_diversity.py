"""
mechanism_diversity.py
----------------------
SIMULATION 4: Mutual information and Pearson correlation between mechanism votes.
Run ELAPSE ensemble at n=100 ER, 30 seeds. Record vote time series v1..v5.
Compute pairwise Pearson correlation and mutual information.
Saves fig12_mechanism_diversity.png and fig12_mechanism_diversity.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy
from m6_ensemble import simulate as sim_m6

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

T = 15.0
DT = 0.02
N = 100
N_SEEDS = 30

MECH_NAMES = ['v1\nEGDM', 'v2\nEpidemic', 'v3\nFinance', 'v4\nBiology', 'v5\nSocial']


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


def discretize_votes(votes_arr, n_bins=20):
    """Discretize vote time series into bins for mutual information."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    return np.digitize(votes_arr, bins) - 1


def run_mechanism_diversity():
    G, L = make_erdos_renyi(N, p=0.15, seed=42)
    A = np.abs((L - np.diag(np.diag(L))) * -1)
    lambda2 = fiedler_value(L)
    w = np.ones(5) / 5

    all_votes = []  # list of (steps+1, 5) arrays

    print(f"Running ELAPSE ensemble n={N}, {N_SEEDS} seeds...")
    for seed in range(N_SEEDS):
        np.random.seed(seed)
        x0 = make_x0(N, seed)
        params = make_params(N, seed)

        t_arr, _, H_arr, M_arr, Delta_arr, votes_arr = sim_m6(
            x0, L, A, params, lambda2, weights=w, T=T, dt=DT, stochastic=True)
        all_votes.append(votes_arr)
        if seed % 10 == 0:
            print(f"  Seed {seed} done. votes shape: {votes_arr.shape}")

    # Stack all seeds: (N_SEEDS * steps, 5) -- concatenate time series across seeds
    combined_votes = np.concatenate(all_votes, axis=0)  # (N_SEEDS * steps, 5)
    print(f"Combined votes shape: {combined_votes.shape}")

    # Compute pairwise Pearson correlation
    corr_matrix = np.corrcoef(combined_votes.T)  # (5, 5)
    print("Pearson correlation matrix:")
    print(np.round(corr_matrix, 3))

    # Compute pairwise mutual information (using discretized votes)
    disc_votes = discretize_votes(combined_votes)  # (steps, 5)
    mi_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            mi_matrix[i, j] = mutual_info_score(disc_votes[:, i], disc_votes[:, j])
    print("Mutual information matrix:")
    print(np.round(mi_matrix, 4))

    return corr_matrix, mi_matrix


def plot_mechanism_diversity(corr_matrix, mi_matrix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Mechanism Diversity: Pairwise Correlation and Mutual Information\n'
                 '(n=100 ER, 30 seeds, votes v1–v5 concatenated)',
                 fontsize=12, fontweight='bold')

    # Pearson correlation heatmap
    ax = axes[0]
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(MECH_NAMES, fontsize=9)
    ax.set_yticklabels(MECH_NAMES, fontsize=9)
    ax.set_title('Pearson Correlation', fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    for i in range(5):
        for j in range(5):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

    # Mutual information heatmap
    ax = axes[1]
    mi_max = mi_matrix.max()
    im2 = ax.imshow(mi_matrix, cmap='YlOrRd', vmin=0, vmax=mi_max if mi_max > 0 else 1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(MECH_NAMES, fontsize=9)
    ax.set_yticklabels(MECH_NAMES, fontsize=9)
    ax.set_title('Mutual Information (nats)', fontsize=11)
    plt.colorbar(im2, ax=ax, shrink=0.8)

    for i in range(5):
        for j in range(5):
            val = mi_matrix[i, j]
            color = 'white' if val > 0.6 * mi_max else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=color)

    # Add interpretation note
    fig.text(0.5, -0.03,
             'Low off-diagonal values indicate non-redundant mechanisms (desirable for ensemble diversity)',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig12_mechanism_diversity.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig12_mechanism_diversity.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running mechanism diversity analysis...")
    corr_matrix, mi_matrix = run_mechanism_diversity()
    plot_mechanism_diversity(corr_matrix, mi_matrix)
    print("Done.")
