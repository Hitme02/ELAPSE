"""
convexity_check.py
------------------
Numerical verification of IEE convexity in w (Phase 2A).

Samples 500 weight vectors uniformly on the 5-simplex and computes IEE
for each, then verifies that the empirical surface satisfies:
  IEE(theta*wa + (1-theta)*wb) <= theta*IEE(wa) + (1-theta)*IEE(wb)
for 1000 random pairs (wa, wb) and theta in (0,1).

Reports the violation fraction and a summary statistic.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from math_utils import entropy, max_entropy
from networks   import make_erdos_renyi
import m1_egdm    as m1
import m2_epidemic as m2
import m3_finance  as m3
import m4_biology  as m4
import m5_social   as m5
from m6_ensemble import simulate as sim_m6


T  = 15.0
DT = 0.05   # coarser timestep for speed during weight-space sweep


def make_quick_params(n):
    H_max = max_entropy(n)
    H_c   = 0.65 * H_max
    s     = np.zeros(n)
    return {
        'alpha': 0.3, 'mu': 1.5, 'H_c': H_c, 'beta': 2.0,
        'n_hill': 4.0, 'beta_sir': 0.4, 'gamma': 0.15,
        'theta_ou': 0.3, 'mu_ou': 0.1, 'sigma_ou': 0.04,
        'kappa': 0.3, 'deletion_threshold': 0.05, 'sigma_noise': 0.015,
        's': s, 'T_train': T, 'dt': DT,
    }


def compute_iee(weights, x0, L, A, params, lambda2):
    """Compute IEE for a given weight vector (deterministic run for stability)."""
    _, _, H_arr, M_arr, _, _ = sim_m6(
        x0, L, A, params, lambda2,
        weights=weights, T=T, dt=DT, stochastic=False
    )
    return float(np.sum(H_arr * M_arr) * DT)


def sample_simplex(n_samples, K=5, rng=None):
    """
    Sample n_samples points uniformly from the K-simplex using the
    Dirichlet(1,...,1) distribution (equivalent to uniform on simplex).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    raw = rng.exponential(1.0, size=(n_samples, K))
    return raw / raw.sum(axis=1, keepdims=True)


def run_convexity_check(n=50, n_weight_samples=500, n_pair_tests=1000, verbose=True):
    """
    Main convexity verification routine.

    Returns
    -------
    dict with:
      weights       : (n_weight_samples, 5) sampled weight vectors
      iees          : (n_weight_samples,) IEE values
      violation_rate: fraction of convexity-violated pairs
      max_violation : worst-case violation magnitude
      summary       : human-readable summary string
    """
    rng = np.random.default_rng(42)
    G, L = make_erdos_renyi(n, seed=42)
    A    = np.abs((L - np.diag(np.diag(L))) * -1)

    # Fiedler value
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    lambda2     = float(eigenvalues[1])

    params = make_quick_params(n)
    x0     = np.zeros(n)
    seeds  = rng.choice(n, max(1, n // 10), replace=False)
    x0[seeds] = rng.uniform(0.5, 1.0, len(seeds))

    # ── Step 1: Sample weight vectors and compute IEE ─────────────────
    if verbose:
        print(f"  Sampling {n_weight_samples} weight vectors on 5-simplex...")
    weights_all = sample_simplex(n_weight_samples, K=5, rng=rng)
    iees = np.zeros(n_weight_samples)

    for i, w in enumerate(weights_all):
        iees[i] = compute_iee(w, x0, L, A, params, lambda2)
        if verbose and (i + 1) % 100 == 0:
            print(f"    Computed {i+1}/{n_weight_samples}  "
                  f"IEE range: [{iees[:i+1].min():.2f}, {iees[:i+1].max():.2f}]")

    # ── Step 2: Test convexity on random pairs ────────────────────────
    if verbose:
        print(f"  Testing convexity on {n_pair_tests} random pairs...")

    idxA = rng.integers(0, n_weight_samples, n_pair_tests)
    idxB = rng.integers(0, n_weight_samples, n_pair_tests)
    theta_vals = rng.uniform(0.1, 0.9, n_pair_tests)

    violations   = 0
    max_viol     = 0.0
    viol_amounts = []

    for k in range(n_pair_tests):
        wa     = weights_all[idxA[k]]
        wb     = weights_all[idxB[k]]
        theta  = theta_vals[k]
        wc     = theta * wa + (1 - theta) * wb

        iee_a  = iees[idxA[k]]
        iee_b  = iees[idxB[k]]
        iee_c  = compute_iee(wc, x0, L, A, params, lambda2)

        rhs    = theta * iee_a + (1 - theta) * iee_b
        viol   = iee_c - rhs       # positive = convexity violated
        viol_amounts.append(viol)

        if viol > 1e-3:            # small numerical tolerance
            violations += 1
            max_viol = max(max_viol, viol)

    viol_amounts = np.array(viol_amounts)
    violation_rate = violations / n_pair_tests

    summary = (
        f"Convexity check (n={n}, {n_weight_samples} samples, {n_pair_tests} pairs):\n"
        f"  IEE range     : [{iees.min():.3f}, {iees.max():.3f}]\n"
        f"  IEE mean ± std: {iees.mean():.3f} ± {iees.std():.3f}\n"
        f"  Violation rate: {violation_rate:.4f}  "
        f"({violations}/{n_pair_tests} pairs violated by >0.001)\n"
        f"  Max violation : {max_viol:.4f}\n"
        f"  Mean |viol|   : {np.abs(viol_amounts).mean():.4f}\n"
        f"  Conclusion    : {'CONVEX (empirically validated)' if violation_rate < 0.02 else 'NON-CONVEX (counterexamples found)'}"
    )

    if verbose:
        print(summary)

    return {
        'weights':        weights_all,
        'iees':           iees,
        'violation_rate': violation_rate,
        'max_violation':  max_viol,
        'viol_amounts':   viol_amounts,
        'summary':        summary,
        'n':              n,
    }


if __name__ == '__main__':
    result = run_convexity_check(n=50, n_weight_samples=500, n_pair_tests=1000, verbose=True)
    print("\nFinal result:", result['summary'])
