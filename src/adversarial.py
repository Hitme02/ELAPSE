"""
adversarial.py
--------------
Phase 2D: Adversarial evasion model.

Models an attacker who controls fraction f in [0,1] of nodes and
reduces edge weights incident to adversarial nodes by factor (1-f),
keeping H(t) low to prevent ELAPSE from triggering deletion.

Attack model (symmetric formulation, Fix 6B):
  - Attacker controls a random fraction f of nodes
  - For each edge (i,j) where j is adversarial, reduce A[i,j]=A[j,i] by (1-f)
  - Recompute symmetric L_adv = D_adv - A_adv
  - This preserves Laplacian symmetry (required for real eigenvalues)

Countermeasure:
  - Adaptive H_c: if dH/dt is anomalously slow, lower H_c
  - H_c_eff(t) = H_c * (1 - gamma_adapt * slowness(t))

Simulations run for:
  - f in {0.0, 0.1, 0.2, 0.3}
  - H_c_frac in {0.65 (conservative), 0.50 (aggressive)}
  - topologies: ER and BA at n=200
  - 30 random seeds each
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_utils import entropy, max_entropy, sigma_egdm
from networks   import make_erdos_renyi, make_barabasi_albert, fiedler_value


T  = 15.0
DT = 0.02
N_SEEDS = 30


def make_params(n, H_c_frac=0.65, beta=2.0):
    H_max = max_entropy(n)
    H_c   = H_c_frac * H_max
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': beta,
        'H_c_frac': H_c_frac,
        'n_hill': 4.0, 'sigma_noise': 0.015,
        's': np.zeros(n), 'T_train': T, 'dt': DT,
    }


def make_symmetric_adversarial_laplacian(L, adversarial_nodes, f):
    """
    Build a symmetric adversarial Laplacian.

    For each edge (i,j) where j is adversarial, scale A[i,j]=A[j,i]
    by (1-f). Then recompute L_adv = D_adv - A_adv.
    This preserves symmetry of L, ensuring real eigenvalues.

    Parameters
    ----------
    L               : (n,n) symmetric Laplacian
    adversarial_nodes : boolean array (n,)
    f               : throttle fraction in [0,1]
    """
    n = L.shape[0]
    # Recover adjacency from Laplacian: A = -L + diag(L)
    A = -(L - np.diag(np.diag(L)))

    A_adv = A.copy()
    # Scale all edges incident to adversarial nodes
    if f > 0 and adversarial_nodes.any():
        adv_idx = np.where(adversarial_nodes)[0]
        for j in adv_idx:
            A_adv[:, j] *= (1.0 - f)
            A_adv[j, :] *= (1.0 - f)
        # Fix double-counted adversarial-adversarial edges
        for j in adv_idx:
            for k in adv_idx:
                if A[j, k] > 0:
                    # Was scaled by (1-f)^2, restore to (1-f)
                    A_adv[j, k] = A[j, k] * (1.0 - f)

    # Recompute Laplacian
    D_adv = np.diag(A_adv.sum(axis=1))
    L_adv = D_adv - A_adv
    return L_adv


def simulate_adversarial(x0, L, params, adversarial_nodes, f,
                          T=T, dt=DT, stochastic=True,
                          use_adaptive_Hc=False, adaptive_gamma=0.4,
                          window=50):
    """
    Simulate M1 (EGDM) with symmetric adversarial Laplacian.

    adversarial_nodes : boolean array (n,), True if node is adversarial
    f                 : throttle fraction (edges incident to adv nodes scaled by 1-f)
    use_adaptive_Hc   : if True, lower H_c when entropy growth is anomalously slow
    """
    n     = len(x0)
    steps = int(T / dt)
    x     = x0.copy()

    alpha  = params['alpha']
    mu     = params['mu']
    H_c    = params['H_c']
    beta   = params.get('beta', 2.0)
    H_max  = max_entropy(n)
    s      = params.get('s', np.zeros(n))

    # Build symmetric adversarial Laplacian
    L_adv = make_symmetric_adversarial_laplacian(L, adversarial_nodes, f)

    t_arr     = np.zeros(steps + 1)
    H_arr     = np.zeros(steps + 1)
    M_arr     = np.zeros(steps + 1)
    Hc_arr    = np.zeros(steps + 1)

    H_arr[0]  = entropy(x)
    M_arr[0]  = x.sum()
    Hc_arr[0] = H_c
    H_c_eff   = H_c

    for i in range(steps):
        H_curr = entropy(x)

        # ── Adaptive H_c countermeasure ────────────────────────────────
        if use_adaptive_Hc and i >= window:
            dH_window   = (H_arr[i] - H_arr[max(0, i - window)]) / (window * dt)
            dH_baseline = max_entropy(n) / (T * 0.3)
            if dH_window < dH_baseline * 0.5:
                slowness  = max(0.0, dH_baseline - dH_window) / (dH_baseline + 1e-8)
                H_c_eff   = H_c * (1.0 - adaptive_gamma * slowness)
                H_c_eff   = max(H_c_eff, 0.3 * H_max)
            else:
                H_c_eff = H_c
        else:
            H_c_eff = H_c

        Hc_arr[i] = H_c_eff

        diffusion  = -alpha * (L_adv @ x)
        sig        = sigma_egdm(H_curr, H_c_eff, H_max, beta)
        mortality  = -mu * sig * x
        dx         = diffusion + mortality + s

        x = x + dx * dt

        if stochastic:
            sigma_n = params.get('sigma_noise', 0.015)
            noise   = sigma_n * np.sqrt(np.maximum(x, 0)) * np.random.normal(0, np.sqrt(dt), n)
            x       = x + noise

        x = np.maximum(x, 0)

        t_arr[i + 1]  = (i + 1) * dt
        H_arr[i + 1]  = entropy(x)
        M_arr[i + 1]  = x.sum()
        Hc_arr[i + 1] = H_c_eff

    M0  = M_arr[0] if M_arr[0] > 0 else 1.0
    IEE = float(np.sum(H_arr * M_arr) * dt)

    below_half = np.where(M_arr < 0.5 * M0)[0]
    t_star     = float(t_arr[below_half[0]]) if len(below_half) > 0 else T

    return {
        't_arr': t_arr, 'H_arr': H_arr, 'M_arr': M_arr,
        'Hc_arr': Hc_arr, 'IEE': IEE, 't_star': t_star,
    }


def run_adversarial_study(n=200, f_values=(0.0, 0.1, 0.2, 0.3),
                          n_seeds=N_SEEDS, H_c_frac=0.65, verbose=True):
    """
    Run adversarial degradation study across ER and BA topologies.

    Returns nested dict:
      results[topo][f]['plain']    — without countermeasure
      results[topo][f]['adaptive'] — with adaptive H_c
    Each entry has 'iee_mean', 'iee_ci', 'tstar_mean', 'tstar_ci', 'all_iees'
    """
    from scipy import stats

    topos = {
        'Erdos-Renyi':     lambda: make_erdos_renyi(n, seed=42),
        'Barabasi-Albert': lambda: make_barabasi_albert(n, seed=42),
    }

    results = {}

    for topo_name, make_net in topos.items():
        G, L = make_net()
        results[topo_name] = {}

        if verbose:
            print(f"\n  Topology: {topo_name}  (H_c_frac={H_c_frac})")

        for f in f_values:
            results[topo_name][f] = {'plain': {}, 'adaptive': {}}

            plain_iees    = []
            plain_tstars  = []
            adapt_iees    = []
            adapt_tstars  = []

            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)

                x0 = np.zeros(n)
                seed_nodes = rng.choice(n, max(1, n // 10), replace=False)
                x0[seed_nodes] = rng.uniform(0.5, 1.0, len(seed_nodes))

                adv_nodes = np.zeros(n, dtype=bool)
                if f > 0:
                    n_adv   = max(1, int(f * n))
                    adv_idx = rng.choice(n, n_adv, replace=False)
                    adv_nodes[adv_idx] = True

                params = make_params(n, H_c_frac=H_c_frac)
                params['s'] = np.zeros(n)

                r_plain = simulate_adversarial(
                    x0, L, params, adv_nodes, f,
                    use_adaptive_Hc=False, stochastic=True
                )
                plain_iees.append(r_plain['IEE'])
                plain_tstars.append(r_plain['t_star'])

                r_adapt = simulate_adversarial(
                    x0, L, params, adv_nodes, f,
                    use_adaptive_Hc=True, stochastic=True
                )
                adapt_iees.append(r_adapt['IEE'])
                adapt_tstars.append(r_adapt['t_star'])

            for tag, iees, tstars, key in [
                ('plain', plain_iees, plain_tstars, 'plain'),
                ('adaptive', adapt_iees, adapt_tstars, 'adaptive'),
            ]:
                arr_i = np.array(iees)
                arr_t = np.array(tstars)
                ci_i  = stats.t.interval(0.95, df=n_seeds-1,
                                          loc=arr_i.mean(), scale=stats.sem(arr_i))
                ci_t  = stats.t.interval(0.95, df=n_seeds-1,
                                          loc=arr_t.mean(), scale=stats.sem(arr_t))
                results[topo_name][f][key] = {
                    'iee_mean':   float(arr_i.mean()),
                    'iee_std':    float(arr_i.std()),
                    'iee_ci':     (float(ci_i[0]), float(ci_i[1])),
                    'tstar_mean': float(arr_t.mean()),
                    'tstar_std':  float(arr_t.std()),
                    'tstar_ci':   (float(ci_t[0]), float(ci_t[1])),
                    'all_iees':   arr_i.tolist(),
                }

            if verbose:
                pi = results[topo_name][f]['plain']['iee_mean']
                ai = results[topo_name][f]['adaptive']['iee_mean']
                print(f"    f={f:.1f}  plain={pi:.2f}  adaptive={ai:.2f}")

    return results


def run_adversarial_both_thresholds(n=200, f_values=(0.0, 0.1, 0.2, 0.3),
                                     n_seeds=N_SEEDS, verbose=True):
    """
    Run adversarial study for both conservative (H_c=0.65) and aggressive
    (H_c=0.50) thresholds. Returns dict keyed by H_c_frac.
    """
    results = {}
    for H_c_frac in [0.65, 0.50]:
        label = f'H_c_{int(H_c_frac*100)}'
        if verbose:
            print(f"\n{'='*50}")
            print(f"  H_c_frac = {H_c_frac}")
            print(f"{'='*50}")
        results[label] = run_adversarial_study(
            n=n, f_values=f_values, n_seeds=n_seeds,
            H_c_frac=H_c_frac, verbose=verbose
        )
    return results


if __name__ == '__main__':
    r = run_adversarial_both_thresholds(n=200, verbose=True)
    import json
    for hc_label, hc_res in r.items():
        print(f"\n{hc_label}:")
        print(json.dumps({
            topo: {str(f): {k: v['iee_mean'] for k, v in d.items()}
                   for f, d in hc_res[topo].items()}
            for topo in hc_res
        }, indent=2))
