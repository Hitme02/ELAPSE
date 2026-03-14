"""
run_all.py
----------
Single entry point for the ELAPSE framework.

Usage:
    python run_all.py            # Full run: n = [50, 100, 200, 500]
    python run_all.py --quick    # Quick smoke test: n = [50] only
    python run_all.py --medium   # Medium run: n = [50, 100] only
    python run_all.py --no-sens  # Skip sensitivity analysis

Outputs:
    output/results.pkl       — main simulation results
    output/sensitivity.pkl   — sensitivity analysis results
    output/figures/          — 8 PNG figures
"""

import sys, os, argparse, pickle, time
import numpy as np

# ── Ensure the src directory is on the path ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

from run_simulation import run_all, run_sensitivity, OUTPUT_DIR, FIGURES_DIR
from plot_results   import generate_all_figures

MODEL_ORDER = ['M0_Baseline', 'M1_EGDM', 'M2_Epidemic', 'M3_Finance',
               'M4_Biology', 'M5_Social', 'M6_Ensemble']
TOPO_ORDER  = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
TOPO_SHORT  = {'Erdos-Renyi': 'ER', 'Barabasi-Albert': 'BA', 'Watts-Strogatz': 'WS'}


def print_banner(text, char='=', width=70):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_summary(results):
    print_banner("SIMULATION SUMMARY")

    for n in sorted(results.keys()):
        print(f"\n── n = {n} ──")
        print(f"{'Model':<22} {'Topology':<20} {'IEE':>9} {'t*':>8} {'Final M':>10}")
        print("-" * 72)

        for topo in TOPO_ORDER:
            for m in MODEL_ORDER:
                r    = results[n].get(topo, {}).get(m, {})
                iee  = r.get('IEE', float('nan'))
                ts   = r.get('t_star', float('nan'))
                fm   = r.get('final_mass', float('nan'))
                flag = " ◀ ELAPSE" if m == 'M6_Ensemble' else (
                       " ◀ WORST"  if m == 'M0_Baseline' else "")
                print(f"  {m:<20} {TOPO_SHORT[topo]:<20} {iee:>9.3f} {ts:>8.3f} {fm:>10.4f}{flag}")
            print()

    # Highlight weight collapse status
    print("\nEnsemble Weight Diagnosis:")
    for n in sorted(results.keys()):
        for topo in TOPO_ORDER:
            w = results[n].get(topo, {}).get('M6_Ensemble', {}).get('learned_weights', None)
            if w is not None:
                wmax = w.max()
                status = "✓ distributed" if wmax < 0.8 else "⚠ concentrated"
                print(f"  n={n:<4} {TOPO_SHORT[topo]:<4}  max_w={wmax:.3f}  weights={np.round(w, 3)}  {status}")


def main():
    parser = argparse.ArgumentParser(description="ELAPSE Framework — Run All Simulations")
    parser.add_argument('--quick',   action='store_true', help='Quick test: n=50 only')
    parser.add_argument('--medium',  action='store_true', help='Medium run: n=50,100')
    parser.add_argument('--no-sens', action='store_true', help='Skip sensitivity analysis')
    args = parser.parse_args()

    if args.quick:
        sizes = [50]
        print_banner("ELAPSE Framework — QUICK SMOKE TEST (n=50)")
    elif args.medium:
        sizes = [50, 100]
        print_banner("ELAPSE Framework — MEDIUM RUN (n=50,100)")
    else:
        sizes = [50, 100, 200, 500]
        print_banner("ELAPSE Framework — FULL SIMULATION RUN")

    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── Step 1: Main simulations ──────────────────────────────────────────
    t0 = time.time()
    results = run_all(sizes=sizes, verbose=True)
    t1 = time.time()
    print(f"\n  Simulation time: {t1 - t0:.1f}s")

    results_path = os.path.join(OUTPUT_DIR, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"  Results saved → {results_path}")

    print_summary(results)

    # ── Step 2: Sensitivity analysis ─────────────────────────────────────
    if not args.no_sens:
        print_banner("Sensitivity Analysis  (H_c × β  →  IEE)", char='-')
        sens_n = 50 if args.quick else 100
        t2 = time.time()
        sens = run_sensitivity(n=sens_n, verbose=True)
        t3 = time.time()
        print(f"\n  Sensitivity time: {t3 - t2:.1f}s")

        sens_path = os.path.join(OUTPUT_DIR, 'sensitivity.pkl')
        with open(sens_path, 'wb') as f:
            pickle.dump(sens, f)
        print(f"  Sensitivity saved → {sens_path}")

    # ── Step 3: Figures ───────────────────────────────────────────────────
    print_banner("Generating Figures", char='-')
    # Use largest available n for trajectory plots, unless quick test
    n_traj = min(max(results.keys()), 100)
    figure_paths = generate_all_figures(n_for_trajectories=n_traj)

    # ── Final summary ─────────────────────────────────────────────────────
    print_banner("Done")
    print(f"  Results : {results_path}")
    print(f"  Figures : {FIGURES_DIR}/")
    for p in figure_paths:
        print(f"    {os.path.basename(p)}")
    print()


if __name__ == '__main__':
    main()
