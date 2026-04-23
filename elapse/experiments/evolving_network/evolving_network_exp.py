"""
evolving_network_exp.py
-----------------------
Experiment: evolving network (ER -> BA transition).

At each dt_rewire=1.0 time unit:
  - Add 2 edges via preferential attachment
  - Remove 2 random edges (maintaining connectivity)

Tests hypothesis: M6 advantage over M5 INCREASES on evolving graphs.
"""

import numpy as np
import networkx as nx
import pickle, os, sys, multiprocessing
import concurrent.futures

N_WORKERS = min(multiprocessing.cpu_count(), 8)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

from networks import make_erdos_renyi, fiedler_value
from math_utils import max_entropy, entropy
import m0_baseline as m0, m5_social as m5
from m6_ensemble import simulate as sim_m6, learn_weights, collect_votes, ensemble_signal


T = 15.0
DT = 0.02
dt_rewire = 1.0
N_TEST = 20
N_TRAIN = 10


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
        'kappa': 0.3, 'sigma_noise': 0.015, 's': s, 'T_train': T, 'dt': DT,
    }


def make_x0(n, seed=None):
    rng = np.random.default_rng(seed)
    x0  = np.zeros(n)
    idx = rng.choice(n, max(1, n//10), replace=False)
    x0[idx] = rng.uniform(0.5, 1.0, len(idx))
    return x0


def compute_iee(H, M, dt):
    return float(np.sum(H * M) * dt)


def evolve_graph(G, rng, n_add=2, n_remove=2):
    """
    Evolve graph one step:
    - Add n_add edges via preferential attachment
    - Remove n_remove random edges (maintaining connectivity)
    Returns updated G.
    """
    nodes = list(G.nodes())
    degrees = np.array([G.degree(nd) for nd in nodes], dtype=float)

    # Preferential attachment: add edges
    for _ in range(n_add):
        deg_sum = degrees.sum()
        if deg_sum <= 0:
            probs = np.ones(len(nodes)) / len(nodes)
        else:
            probs = degrees / deg_sum
        u_idx = rng.choice(len(nodes), p=probs)
        v_idx = rng.choice(len(nodes))
        u, v = nodes[u_idx], nodes[v_idx]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            degrees[u_idx] += 1
            degrees[v_idx] += 1

    # Remove edges (avoiding bridges to maintain connectivity)
    edges = list(G.edges())
    removed = 0
    attempts = 0
    while removed < n_remove and attempts < 100:
        if not edges:
            break
        e_idx = rng.integers(0, len(edges))
        u, v = edges[e_idx]
        G.remove_edge(u, v)
        if nx.is_connected(G):
            edges.pop(e_idx)
            removed += 1
        else:
            G.add_edge(u, v)  # restore
        attempts += 1

    return G


def simulate_evolving_m5(x0, G_init, params, T=T, dt=DT, dt_rewire=1.0, seed=0):
    """Run M5 on an evolving graph."""
    from math_utils import sigma_egdm
    n     = len(x0)
    steps = int(T / dt)
    rewire_steps = int(dt_rewire / dt)

    G = G_init.copy()
    rng = np.random.default_rng(seed + 1000)

    x = x0.copy()
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)
    H_arr[0] = entropy(x)
    M_arr[0]  = x.sum()

    sigma_n = params.get('sigma_noise', 0.015)
    H_max   = max_entropy(n)
    H_c     = params['H_c']
    mu      = params['mu']
    alpha   = params['alpha']
    beta    = params['beta']
    kappa   = params.get('kappa', 0.3)
    s       = params.get('s', np.zeros(n))

    for i in range(steps):
        # Rewire if needed
        if i > 0 and i % rewire_steps == 0:
            G = evolve_graph(G, rng)

        L = nx.laplacian_matrix(G).toarray().astype(float)
        A = np.abs((L - np.diag(np.diag(L))) * -1)

        H = entropy(x)
        M = x.sum()

        # M5 local cascade pressure vote
        degrees_arr = np.array([G.degree(j) for j in range(n)], dtype=float)
        degrees_arr = np.maximum(degrees_arr, 1)
        nbr_pressure = np.zeros(n)
        for j in range(n):
            neighbours = list(G.neighbors(j))
            if neighbours:
                nbr_pressure[j] = sum(x[k] for k in neighbours) / (degrees_arr[j] * M + 1e-6)

        v5 = np.mean(nbr_pressure) * sigma_egdm(H, H_c, H_max, beta)
        v5 = float(np.clip(v5 * kappa, 0, 1))

        dxdt = -alpha * (L @ x) - mu * v5 * x + s
        x = x + dxdt * dt

        dW = np.random.default_rng(seed + i).standard_normal(n) * np.sqrt(dt)
        x  = x + sigma_n * np.sqrt(np.maximum(x, 0)) * dW
        x  = np.maximum(x, 0)

        H_arr[i+1] = entropy(x)
        M_arr[i+1] = x.sum()

    t_arr = np.arange(steps+1) * dt
    return t_arr, H_arr, M_arr


def simulate_evolving_m6(x0, G_init, params, w_learned, T=T, dt=DT, dt_rewire=1.0, seed=0):
    """Run M6 ensemble on evolving graph, updating L/A at each rewire."""
    import m2_epidemic as m2_mod

    n     = len(x0)
    steps = int(T / dt)
    rewire_steps = int(dt_rewire / dt)

    G = G_init.copy()
    rng = np.random.default_rng(seed + 2000)

    x = x0.copy()
    H_arr = np.zeros(steps + 1)
    M_arr = np.zeros(steps + 1)
    H_arr[0] = entropy(x)
    M_arr[0]  = x.sum()

    sigma_n = params.get('sigma_noise', 0.015)
    alpha   = params['alpha']
    mu      = params['mu']
    s       = params.get('s', np.zeros(n))
    theta   = 0.4

    # M2 state
    I0 = x0.copy()
    S0 = np.maximum(np.ones(n) - I0, 0)
    state2 = np.concatenate([S0, I0, np.zeros(n)])

    for i in range(steps):
        if i > 0 and i % rewire_steps == 0:
            G = evolve_graph(G, rng)

        L       = nx.laplacian_matrix(G).toarray().astype(float)
        A       = np.abs((L - np.diag(np.diag(L))) * -1)
        lambda2 = fiedler_value(L)

        t_curr = i * dt
        votes = collect_votes(x, state2, L, A, params, t_curr, lambda2)
        Delta, fires, eff_sig = ensemble_signal(votes, w_learned, theta)

        dxdt = -alpha * (L @ x) - mu * eff_sig * x + s
        x = x + dxdt * dt

        dstate2 = m2_mod.derivatives(state2, A, params)
        state2  = np.maximum(state2 + dstate2 * dt, 0)

        dW = np.random.default_rng(seed + i + 500).standard_normal(n) * np.sqrt(dt)
        x  = x + sigma_n * np.sqrt(np.maximum(x, 0)) * dW
        x  = np.maximum(x, 0)

        H_arr[i+1] = entropy(x)
        M_arr[i+1] = x.sum()

    t_arr = np.arange(steps+1) * dt
    return t_arr, H_arr, M_arr


def _evolving_seed_worker(args):
    """Top-level worker: run static+evolving M5 and M6 for one seed."""
    n, seed, G_init, L_init, lambda2, w_learned = args
    A_init = np.abs((L_init - np.diag(np.diag(L_init))) * -1)
    params = make_params(n, seed=seed)
    x0     = make_x0(n, seed=seed)

    np.random.seed(seed)
    _, _, H, M = m5.simulate(x0, L_init, A_init, params, T=T, dt=DT, stochastic=True)
    static_m5 = compute_iee(H, M, DT)

    np.random.seed(seed)
    _, _, H, M, _, _ = sim_m6(x0, L_init, A_init, params, lambda2,
                               weights=w_learned, T=T, dt=DT, stochastic=True)
    static_m6 = compute_iee(H, M, DT)

    _, H5e, M5e = simulate_evolving_m5(x0, G_init, params, seed=seed)
    evolv_m5 = compute_iee(H5e, M5e, DT)

    _, H6e, M6e = simulate_evolving_m6(x0, G_init, params, w_learned, seed=seed)
    evolv_m6 = compute_iee(H6e, M6e, DT)

    return seed, static_m5, static_m6, evolv_m5, evolv_m6


def run_evolving_network_experiment(n=100, n_seeds=N_TEST, verbose=True):
    """
    Compare M5 vs M6 on static vs evolving graph.
    Returns dict with IEE comparisons.
    """
    G_init, L_init = make_erdos_renyi(n, seed=42)
    lambda2 = fiedler_value(L_init)

    # Learn M6 weights on static graph (train seeds)
    A_init = np.abs((L_init - np.diag(np.diag(L_init))) * -1)
    train_data = []
    for seed in range(N_TRAIN):
        params = make_params(n, seed=seed)
        params['dt'] = 0.05
        x0 = make_x0(n, seed=seed)
        train_data.append((x0, L_init, A_init, params, lambda2))

    w_learned, _ = learn_weights(train_data, lambda_reg=15.0, n_restarts=2, verbose=False)

    if verbose:
        print(f"Evolving network experiment: n={n}, lambda2={lambda2:.4f}")
        print(f"Learned weights: {np.round(w_learned, 3)}")

    results = {
        'static':   {'M5': [], 'M6': []},
        'evolving': {'M5': [], 'M6': []},
    }

    test_seeds = list(range(N_TRAIN, N_TRAIN + n_seeds))
    worker_args = [(n, seed, G_init, L_init, lambda2, w_learned)
                   for seed in test_seeds]

    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for seed, sm5, sm6, em5, em6 in pool.map(_evolving_seed_worker, worker_args):
            results['static']['M5'].append(sm5)
            results['static']['M6'].append(sm6)
            results['evolving']['M5'].append(em5)
            results['evolving']['M6'].append(em6)

    summary = {}
    for scenario in ['static', 'evolving']:
        for model in ['M5', 'M6']:
            arr = np.array(results[scenario][model])
            summary[f'{scenario}_{model}_mean'] = float(arr.mean())
            summary[f'{scenario}_{model}_std']  = float(arr.std())

        m5_mean = summary[f'{scenario}_M5_mean']
        m6_mean = summary[f'{scenario}_M6_mean']
        summary[f'{scenario}_M6_advantage_pct'] = 100.0 * (m5_mean - m6_mean) / (m5_mean + 1e-6)

    if verbose:
        for sc in ['static', 'evolving']:
            m5m = summary[f'{sc}_M5_mean']
            m6m = summary[f'{sc}_M6_mean']
            adv = summary[f'{sc}_M6_advantage_pct']
            print(f"  {sc:10s}: M5={m5m:.2f}, M6={m6m:.2f}, M6 advantage={adv:+.1f}%")

    summary['raw'] = results
    return summary


if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running evolving network experiment...")
    results = run_evolving_network_experiment(n=100, n_seeds=20, verbose=True)

    with open(os.path.join(OUTPUT_DIR, 'evolving_network_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    print("Done.")
