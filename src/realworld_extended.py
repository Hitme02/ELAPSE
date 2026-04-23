"""
realworld_extended.py
---------------------
SIMULATION 6: Extended real-world validation using NetworkX built-in graphs.
Uses: karate_club_graph(), les_miserables_graph(), florentine_families_graph().
Runs M0, M5, M6 with 10 seeds, reports IEE mean ± 95% CI.
Saves fig14_realworld_extended.png and fig14_realworld_extended.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import networkx as nx
import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from networks import fiedler_value
from math_utils import max_entropy
import m0_baseline as m0
import m5_social as m5
from m6_ensemble import simulate as sim_m6

FIGURES_DIR = os.path.join(ROOT_DIR, 'output', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

T = 15.0
DT = 0.02
N_SEEDS = 10


def load_real_networks():
    """Load real-world networks from NetworkX built-ins."""
    networks = {}

    # 1. Karate Club (n=34)
    G = nx.karate_club_graph()
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    L = nx.laplacian_matrix(G).toarray().astype(float)
    networks['Karate Club\n(n=34)'] = (G, L)
    print(f"Karate Club: n={G.number_of_nodes()}, m={G.number_of_edges()}")

    # 2. Les Misérables (n=77)
    try:
        G = nx.les_miserables_graph()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        L = nx.laplacian_matrix(G).toarray().astype(float)
        networks['Les Misérables\n(n=77)'] = (G, L)
        print(f"Les Misérables: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    except AttributeError:
        print("les_miserables_graph not available in this NetworkX version")

    # 3. Davis Southern Women (n=18 events)
    try:
        G = nx.davis_southern_women_graph()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        L = nx.laplacian_matrix(G).toarray().astype(float)
        networks['Davis Women\n(n=32)'] = (G, L)
        print(f"Davis Southern Women: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    except Exception as e:
        print(f"Davis Women graph failed: {e}")

    # 4. Florentine Families (n=15)
    try:
        G = nx.florentine_families_graph()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        G = nx.convert_node_labels_to_integers(G)
        L = nx.laplacian_matrix(G).toarray().astype(float)
        networks['Florentine\nFamilies (n=15)'] = (G, L)
        print(f"Florentine Families: n={G.number_of_nodes()}, m={G.number_of_edges()}")
    except Exception as e:
        print(f"Florentine Families graph failed: {e}")

    return networks


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


def compute_ci(iees):
    arr = np.array(iees)
    n = len(arr)
    mu = float(arr.mean())
    if n > 1:
        ci = stats.t.interval(0.95, df=n - 1, loc=mu, scale=stats.sem(arr))
    else:
        ci = (mu, mu)
    half_ci = (ci[1] - ci[0]) / 2
    return mu, half_ci


def run_realworld():
    networks = load_real_networks()
    results = {}

    for net_name, (G, L) in networks.items():
        n = G.number_of_nodes()
        A = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)
        w = np.ones(5) / 5

        print(f"\nNetwork: {net_name.replace(chr(10), ' ')}  n={n}")

        m0_iees = []
        m5_iees = []
        m6_iees = []

        for seed in range(N_SEEDS):
            np.random.seed(seed)
            x0 = make_x0(n, seed)
            params = make_params(n, seed)

            # M0
            t_arr, _, H_arr, M_arr = m0.simulate(x0, L, params, T=T, dt=DT, stochastic=True)
            m0_iees.append(compute_iee(H_arr, M_arr, DT))

            # M5
            np.random.seed(seed)
            t_arr, _, H_arr, M_arr = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=True)
            m5_iees.append(compute_iee(H_arr, M_arr, DT))

            # M6
            np.random.seed(seed)
            t_arr, _, H_arr, M_arr, _, _ = sim_m6(x0, L, A, params, lambda2,
                                                    weights=w, T=T, dt=DT, stochastic=True)
            m6_iees.append(compute_iee(H_arr, M_arr, DT))

        m0_mu, m0_ci = compute_ci(m0_iees)
        m5_mu, m5_ci = compute_ci(m5_iees)
        m6_mu, m6_ci = compute_ci(m6_iees)

        results[net_name] = {
            'n': n,
            'M0': (m0_mu, m0_ci),
            'M5': (m5_mu, m5_ci),
            'M6': (m6_mu, m6_ci),
        }
        print(f"  M0={m0_mu:.3f}±{m0_ci:.3f}  M5={m5_mu:.3f}±{m5_ci:.3f}  M6={m6_mu:.3f}±{m6_ci:.3f}")

    return results


def plot_realworld(results):
    net_names = list(results.keys())
    n_nets = len(net_names)

    x = np.arange(n_nets)
    width = 0.25

    colors = {'M0': '#90A4AE', 'M5': '#EF5350', 'M6': '#42A5F5'}
    labels_map = {'M0': 'M0 (Baseline)', 'M5': 'M5 (Social)', 'M6': 'M6 (ELAPSE)'}

    fig, ax = plt.subplots(figsize=(max(10, n_nets * 2.5), 6))

    for i, model in enumerate(['M0', 'M5', 'M6']):
        means = [results[net][model][0] for net in net_names]
        cis = [results[net][model][1] for net in net_names]
        bars = ax.bar(x + (i - 1) * width, means, width, label=labels_map[model],
                      color=colors[model], alpha=0.85, edgecolor='black', linewidth=0.5)
        ax.errorbar(x + (i - 1) * width, means, yerr=cis, fmt='none',
                    color='black', capsize=4, linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(net_names, fontsize=10)
    ax.set_xlabel('Real-World Network', fontsize=12)
    ax.set_ylabel('IEE (Information Entropy Exposure)', fontsize=12)
    ax.set_title('Extended Real-World Validation: M0 vs M5 vs M6 (ELAPSE)\n'
                 '(10 seeds, mean ± 95% CI)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # Add n values below x-axis labels
    for i, net in enumerate(net_names):
        n = results[net]['n']
        ax.text(i, -0.08 * ax.get_ylim()[1], f'n={n}', ha='center', fontsize=9, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    out_png = os.path.join(FIGURES_DIR, 'fig14_realworld_extended.png')
    out_pdf = os.path.join(FIGURES_DIR, 'fig14_realworld_extended.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Running extended real-world validation...")
    results = run_realworld()
    plot_realworld(results)
    print("Done.")
