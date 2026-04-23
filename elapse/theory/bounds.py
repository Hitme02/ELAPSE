"""
bounds.py
---------
Numerical verification of Theorem 2 (stochastic IEE upper bound).

For each (n, topology) combination, runs 1000 Monte Carlo SDE paths and
confirms that the empirical E[IEE] lies strictly below the closed-form
stochastic bound derived via the second-moment Gronwall argument.

Saves results to output/theorem2_stochastic_verification.csv.

Usage
-----
    python elapse/theory/bounds.py
"""

import numpy as np
import csv
import os
import sys

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))
sys.path.insert(0, ROOT_DIR)

from elapse.core.sde.stochastic_bound import (
    stochastic_iee_upper_bound,
    verify_stochastic_bound,
)


# Simulation parameters (matching paper's Table 1)
MU          = 1.5
DELTA_MIN   = 0.05    # conservative lower bound on ensemble signal
SIGMA_N     = 0.015
S_FRAC      = 0.05    # ||s||_1
TAU_STAR    = 2.0     # conservative spectral trigger time
T           = 15.0
N_PATHS     = 1000
DT          = 0.02


def run_verification():
    """
    Run stochastic bound verification for all (n, topology) combinations.
    Returns list of result dicts.
    """
    ns = [50, 100, 200]
    topologies = ['erdos_renyi', 'barabasi_albert', 'watts_strogatz']

    all_results = []
    all_passed  = True

    print("Theorem 2 Stochastic IEE Bound Verification")
    print("=" * 60)
    print(f"Parameters: mu={MU}, delta_min={DELTA_MIN}, sigma_n={SIGMA_N}")
    print(f"            s_norm={S_FRAC}, tau*={TAU_STAR}, T={T}, n_paths={N_PATHS}")
    print()

    for n in ns:
        for topo in topologies:
            print(f"  n={n:4d}, topology={topo}...", end=' ', flush=True)

            result = verify_stochastic_bound(
                n=n,
                topology=topo,
                mu=MU,
                delta_min=DELTA_MIN,
                sigma_n=SIGMA_N,
                s_frac=S_FRAC,
                tau_star=TAU_STAR,
                T=T,
                n_paths=N_PATHS,
                dt=DT,
                seed=42,
            )

            status = "OK" if result['bound_holds'] else "VIOLATION"
            print(
                f"E[IEE]={result['empirical_mean_IEE']:.3f}, "
                f"bound={result['stochastic_bound']:.3f}, "
                f"slack={result['bound_slack_pct']:.1f}%  [{status}]"
            )

            if not result['bound_holds']:
                all_passed = False

            all_results.append(result)

    print()
    print(f"Overall: {'ALL BOUNDS HOLD' if all_passed else 'VIOLATIONS FOUND'}")
    return all_results


def save_csv(results, output_dir):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'theorem2_stochastic_verification.csv')

    fieldnames = [
        'n', 'topology', 'empirical_mean_IEE', 'stochastic_bound',
        'bound_slack_pct', 'n_paths', 'CI_95_lower', 'CI_95_upper', 'bound_holds',
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"Saved {out_path}")
    return out_path


if __name__ == '__main__':
    output_dir = os.path.join(ROOT_DIR, 'output')
    results = run_verification()
    save_csv(results, output_dir)
