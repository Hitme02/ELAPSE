"""
gossip_entropy.py
-----------------
Phase 2E: Distributed gossip-protocol entropy estimation.

Each node maintains a local estimate of H(t) using only k-hop neighbourhood
information (k=2, k=3). Compare estimated H(t) against the true global H(t).

Gossip estimation procedure:
  1. Each node i collects x_j for all j within k hops of i
  2. Normalises the local mass to estimate a local probability distribution
  3. Computes local Shannon entropy H_i^{(k)} from this local view
  4. The global estimate H_est^{(k)}(t) = mean over all nodes of H_i^{(k)}(t)

Metrics:
  - Estimation error: |H_est^{(k)}(t) - H_true(t)|
  - False early triggers: timesteps where H_est > H_c but H_true < H_c
  - Missed triggers    : timesteps where H_est < H_c but H_true > H_c
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from math_utils import entropy, max_entropy, sigma_egdm
from networks   import make_erdos_renyi, make_barabasi_albert, make_watts_strogatz
import m1_egdm as m1


T        = 15.0
DT       = 0.02
N_SEEDS  = 30


def get_k_hop_neighbors(G, node, k):
    """Return set of all nodes within k hops of `node`, including itself."""
    return set(nx.ego_graph(G, node, radius=k).nodes())


def precompute_k_hop_sets(G, k_values=(2, 3)):
    """
    Precompute k-hop neighborhood sets for all nodes.
    Returns dict: k -> list of sets, one per node.
    """
    neighborhoods = {}
    for k in k_values:
        neighborhoods[k] = [get_k_hop_neighbors(G, v, k) for v in G.nodes()]
    return neighborhoods


def gossip_entropy_estimate(x, neighborhoods, k):
    """
    Estimate global H from k-hop local views.

    x             : data concentration vector (n,)
    neighborhoods : precomputed k-hop sets (list of sets)
    k             : hop count (for indexing)

    Returns scalar estimated entropy.
    """
    n = len(x)
    local_H = np.zeros(n)
    for i, nbrs in enumerate(neighborhoods[k]):
        nbr_list = list(nbrs)
        x_local  = x[nbr_list]
        local_H[i] = entropy(x_local)
    return float(local_H.mean())


def make_params(n):
    H_max = max_entropy(n)
    H_c   = 0.65 * H_max
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'sigma_noise': 0.015, 's': np.zeros(n),
    }


def run_one_gossip_sim(x0, G, L, params, k_values=(2, 3), T=T, dt=DT):
    """
    Run M1 simulation and simultaneously track gossip-estimated H(t).

    Returns:
      t_arr            : time array
      H_true_arr       : true global H(t)
      H_est_arr        : dict k -> estimated H(t) from gossip
      trigger_stats    : dict k -> {false_early, missed, total_steps}
    """
    n     = len(x0)
    steps = int(T / dt)
    x     = x0.copy()

    H_c   = params['H_c']
    alpha = params['alpha']
    mu    = params['mu']
    beta  = params.get('beta', 2.0)
    H_max = max_entropy(n)
    s     = params.get('s', np.zeros(n))

    # Precompute k-hop neighborhoods
    neighborhoods = precompute_k_hop_sets(G, k_values=k_values)

    t_arr      = np.zeros(steps + 1)
    H_true_arr = np.zeros(steps + 1)
    H_est_arr  = {k: np.zeros(steps + 1) for k in k_values}

    H_true_arr[0] = entropy(x)
    for k in k_values:
        H_est_arr[k][0] = gossip_entropy_estimate(x, neighborhoods, k)

    for i in range(steps):
        dxdt = m1.derivatives(x, L, params)
        x    = x + dxdt * dt

        # Add stochastic noise
        sigma_n = params.get('sigma_noise', 0.015)
        noise   = sigma_n * np.sqrt(np.maximum(x, 0)) * np.random.normal(0, np.sqrt(dt), n)
        x       = x + noise
        x       = np.maximum(x, 0)

        t_arr[i + 1]      = (i + 1) * dt
        H_true_arr[i + 1] = entropy(x)
        for k in k_values:
            H_est_arr[k][i + 1] = gossip_entropy_estimate(x, neighborhoods, k)

    # ── Trigger statistics ─────────────────────────────────────────────
    trigger_stats = {}
    for k in k_values:
        true_above  = H_true_arr >= H_c
        est_above   = H_est_arr[k] >= H_c
        false_early = int(np.sum(est_above & ~true_above))   # est triggers, true doesn't
        missed      = int(np.sum(true_above & ~est_above))   # true triggers, est doesn't
        trigger_stats[k] = {
            'false_early': false_early,
            'missed':      missed,
            'total_steps': steps + 1,
            'false_early_rate': false_early / (steps + 1),
            'missed_rate':      missed / (steps + 1),
        }

    return t_arr, H_true_arr, H_est_arr, trigger_stats


def find_min_k_for_accuracy(results, missed_threshold=0.05):
    """
    For each topology, find the minimum k achieving missed_rate < missed_threshold.
    Returns dict: topo -> min_k (or None if no k achieves threshold).
    """
    min_k_table = {}
    for topo, data in results.items():
        found = None
        for k in sorted(data['trigger'].keys()):
            if data['trigger'][k]['missed_rate'] < missed_threshold:
                found = k
                break
        min_k_table[topo] = found
    return min_k_table


def run_gossip_study(n=100, k_values=(2, 3, 4, 5), n_seeds=N_SEEDS, verbose=True):
    """
    Run gossip estimation study across three topologies.

    Returns nested dict:
      results[topo] = {
        't_arr'       : mean time array,
        'H_true_mean' : mean true H(t) trajectory,
        'H_true_ci'   : 95% CI band (lower, upper),
        'H_est_mean'  : {k: mean estimated H(t)},
        'H_est_ci'    : {k: (lower, upper)},
        'error_mean'  : {k: mean |H_est - H_true|},
        'trigger'     : {k: {false_early_rate, missed_rate}},
      }
    """
    from scipy import stats

    make_fns = {
        'Erdos-Renyi':     lambda s: make_erdos_renyi(n, seed=s),
        'Barabasi-Albert': lambda s: make_barabasi_albert(n, seed=s),
        'Watts-Strogatz':  lambda s: make_watts_strogatz(n, seed=s),
    }

    results = {}

    steps = int(T / DT)

    for topo_name, make_fn in make_fns.items():
        if verbose:
            print(f"\n  Topology: {topo_name}")

        # Use seed=42 for the graph (same graph across runs, different x0)
        G, L = make_fn(42)
        params = make_params(n)
        params['s'] = np.zeros(n)

        all_H_true  = np.zeros((n_seeds, steps + 1))
        all_H_est   = {k: np.zeros((n_seeds, steps + 1)) for k in k_values}
        all_trig_fe = {k: [] for k in k_values}
        all_trig_ms = {k: [] for k in k_values}

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            x0  = np.zeros(n)
            sns = rng.choice(n, max(1, n // 10), replace=False)
            x0[sns] = rng.uniform(0.5, 1.0, len(sns))

            np.random.seed(seed)
            t_arr, H_true, H_ests, trig = run_one_gossip_sim(
                x0, G, L, params, k_values=k_values
            )

            all_H_true[seed] = H_true
            for k in k_values:
                all_H_est[k][seed]    = H_ests[k]
                all_trig_fe[k].append(trig[k]['false_early_rate'])
                all_trig_ms[k].append(trig[k]['missed_rate'])

            if verbose and (seed + 1) % 10 == 0:
                print(f"    Seed {seed+1}/{n_seeds} done")

        # ── Aggregate statistics ───────────────────────────────────────
        ht_mean = all_H_true.mean(axis=0)
        ht_sem  = all_H_true.std(axis=0) / np.sqrt(n_seeds)
        ht_ci   = (ht_mean - 1.96 * ht_sem, ht_mean + 1.96 * ht_sem)

        H_est_mean = {}
        H_est_ci   = {}
        error_mean = {}
        trig_stats = {}

        for k in k_values:
            he_m   = all_H_est[k].mean(axis=0)
            he_sem = all_H_est[k].std(axis=0) / np.sqrt(n_seeds)
            err    = np.abs(all_H_est[k] - all_H_true).mean(axis=0)

            H_est_mean[k] = he_m
            H_est_ci[k]   = (he_m - 1.96 * he_sem, he_m + 1.96 * he_sem)
            error_mean[k] = err
            trig_stats[k] = {
                'false_early_rate': float(np.mean(all_trig_fe[k])),
                'missed_rate':      float(np.mean(all_trig_ms[k])),
                'false_early_ci':   (
                    float(np.mean(all_trig_fe[k]) - 1.96 * np.std(all_trig_fe[k]) / np.sqrt(n_seeds)),
                    float(np.mean(all_trig_fe[k]) + 1.96 * np.std(all_trig_fe[k]) / np.sqrt(n_seeds)),
                ),
            }

        results[topo_name] = {
            't_arr':       t_arr,
            'H_true_mean': ht_mean,
            'H_true_ci':   ht_ci,
            'H_est_mean':  H_est_mean,
            'H_est_ci':    H_est_ci,
            'error_mean':  error_mean,
            'trigger':     trig_stats,
            'H_c':         params['H_c'],
        }

        if verbose:
            for k in k_values:
                fe = trig_stats[k]['false_early_rate']
                ms = trig_stats[k]['missed_rate']
                me = error_mean[k].mean()
                print(f"    k={k}: mean error={me:.4f}  false_early={fe:.4f}  missed={ms:.4f}")

    return results


if __name__ == '__main__':
    r = run_gossip_study(n=100, verbose=True)
