"""
run_extended.py
---------------
Master runner for the extended ELAPSE paper (AMM Elsevier submission).

Runs:
  Phase 2A  : Numerical convexity check of IEE surface
  Phase 2B  : N=30 seed simulations with 95% CI for all combinations
  Phase 2C  : SNAP real-world network validation
  Phase 2D  : Adversarial evasion study
  Phase 2E  : Gossip entropy estimation study
  Phase 2F  : M3 vote trajectory analysis

Generates: All 8 publication-quality figures (PDF + PNG @ 300 DPI)

Usage:
  python run_extended.py               # Full run (all phases)
  python run_extended.py --quick       # Quick test (n=50, 5 seeds)
  python run_extended.py --no-snap     # Skip SNAP download
  python run_extended.py --phase 2B    # Run specific phase only

Outputs (all in output/):
  ci_results.pkl         Phase 2B: main CI results
  ci_snap.pkl            Phase 2C: SNAP network results
  adversarial_results.pkl Phase 2D
  gossip_results.pkl     Phase 2E
  m3_analysis.pkl        Phase 2F
  convexity_check.pkl    Phase 2A
  figures/*.pdf, *.png   All 8 figures
"""

import sys, os, argparse, pickle, time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, 'src')
sys.path.insert(0, SRC_DIR)

OUTPUT_DIR  = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def banner(text, char='=', width=68):
    print(f"\n{char*width}")
    print(f"  {text}")
    print(f"{char*width}")


def save(obj, name):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"  Saved → {path}")
    return path


def load(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def parse_args():
    p = argparse.ArgumentParser(description='ELAPSE Extended Runner')
    p.add_argument('--quick',    action='store_true', help='Quick test (n=50, 5 seeds)')
    p.add_argument('--no-snap',  action='store_true', help='Skip SNAP dataset download')
    p.add_argument('--phase',    type=str, default=None,
                   help='Run specific phase: 2A, 2B, 2C, 2D, 2E, 2F, figs')
    p.add_argument('--sizes',    type=int, nargs='+', default=None,
                   help='Network sizes to run (default: 50 100 200 500)')
    p.add_argument('--seeds',    type=int, default=None,
                   help='Number of seeds (default: 30)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        sizes   = [50]
        n_seeds = 5
        banner("ELAPSE Extended — QUICK TEST (n=50, 5 seeds)")
    else:
        sizes   = args.sizes or [50, 100, 200, 500]
        n_seeds = args.seeds or 30
        banner(f"ELAPSE Extended — FULL RUN (n={sizes}, {n_seeds} seeds)")

    phase_filter = args.phase
    run_all      = phase_filter is None

    # ── Phase 2A: Convexity check ─────────────────────────────────────────────
    if run_all or phase_filter == '2A':
        banner("Phase 2A: Numerical Convexity Check of IEE", char='-')
        from convexity_check import run_convexity_check

        t0 = time.time()
        n_ws = 50 if args.quick else 500
        n_pt = 100 if args.quick else 1000
        conv = run_convexity_check(n=50, n_weight_samples=n_ws,
                                   n_pair_tests=n_pt, verbose=True)
        print(f"  Time: {time.time()-t0:.1f}s")
        save(conv, 'convexity_check.pkl')

    # ── Phase 2B: Main CI simulations ────────────────────────────────────────
    if run_all or phase_filter == '2B':
        banner(f"Phase 2B: Main Simulations (N={n_seeds} seeds)", char='-')
        from run_simulation_ci import run_all_ci

        t0 = time.time()
        ci_results, raw_results = run_all_ci(sizes=sizes, n_seeds=n_seeds, verbose=True)
        print(f"\n  Total time: {time.time()-t0:.0f}s")
        save(ci_results, 'ci_results.pkl')

    # ── Phase 2C: SNAP real-world validation ──────────────────────────────────
    if (run_all or phase_filter == '2C') and not args.no_snap:
        banner("Phase 2C: SNAP P2P Network Validation", char='-')
        from run_simulation_ci import run_snap_ci

        t0 = time.time()
        ci_snap = run_snap_ci(n_seeds=n_seeds, n_sample=500, verbose=True)
        print(f"  Time: {time.time()-t0:.0f}s")
        save(ci_snap, 'ci_snap.pkl')

    # ── Phase 2D: Adversarial study ───────────────────────────────────────────
    if run_all or phase_filter == '2D':
        banner("Phase 2D: Adversarial Evasion Study (both H_c thresholds)", char='-')
        from adversarial import run_adversarial_both_thresholds

        f_values = [0.0, 0.1, 0.2, 0.3]
        adv_n    = 50 if args.quick else 200
        t0 = time.time()
        adv_results = run_adversarial_both_thresholds(n=adv_n, f_values=f_values,
                                                       n_seeds=n_seeds, verbose=True)
        print(f"  Time: {time.time()-t0:.0f}s")
        save(adv_results, 'adversarial_results.pkl')

    # ── Phase 2E: Gossip entropy estimation ───────────────────────────────────
    if run_all or phase_filter == '2E':
        banner("Phase 2E: Gossip Entropy Estimation Study", char='-')
        from gossip_entropy import run_gossip_study

        gossip_n = 50 if args.quick else 100
        t0 = time.time()
        gossip_results = run_gossip_study(n=gossip_n, k_values=(2, 3, 4, 5),
                                          n_seeds=n_seeds, verbose=True)
        print(f"  Time: {time.time()-t0:.0f}s")
        save(gossip_results, 'gossip_results.pkl')

    # ── Phase 2F: M3 analysis ─────────────────────────────────────────────────
    if run_all or phase_filter == '2F':
        banner("Phase 2F: M3 Vote Trajectory Analysis", char='-')
        from run_simulation_ci import run_m3_analysis

        m3_n = 50 if args.quick else 200
        t0 = time.time()
        m3_results = run_m3_analysis(n=m3_n, n_seeds=n_seeds, verbose=True)
        print(f"  Time: {time.time()-t0:.0f}s")
        save(m3_results, 'm3_analysis.pkl')

    # ── Figures ───────────────────────────────────────────────────────────────
    if run_all or phase_filter == 'figs':
        banner("Generating Publication Figures", char='-')
        from plot_extended import generate_all_extended

        n_traj = min(max(sizes), 100) if sizes else 100
        paths  = generate_all_extended(n_traj=n_traj)
        print(f"\n  {len(paths)} figure files generated in {FIGURES_DIR}/")

    # ── Print summary table ───────────────────────────────────────────────────
    if run_all or phase_filter == '2B':
        ci_results = load('ci_results.pkl')
        if ci_results:
            banner("Summary: Mean IEE ± 95% CI", char='-')
            models = ['M0_Baseline', 'M1_EGDM', 'M2_Epidemic', 'M3_Finance',
                      'M4_Biology', 'M5_Social', 'M6_Ensemble']
            topos  = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
            ts     = {'Erdos-Renyi': 'ER', 'Barabasi-Albert': 'BA', 'Watts-Strogatz': 'WS'}

            for n in sorted(ci_results.keys()):
                print(f"\n  n = {n}")
                print(f"  {'Model':<22} {'Topo':<5} {'IEE':>10}  {'95% CI':^20}  {'t*':>8}")
                print("  " + "-"*67)
                for topo in topos:
                    for m in models:
                        r = ci_results[n].get(topo, {}).get(m, {})
                        if not r:
                            continue
                        iee = r.get('iee_mean', float('nan'))
                        ci  = r.get('iee_ci', (float('nan'), float('nan')))
                        ts_ = r.get('tstar_mean', float('nan'))
                        flag = " ◀ ELAPSE" if m == 'M6_Ensemble' else ""
                        print(f"  {m:<22} {ts[topo]:<5} {iee:>10.2f}  "
                              f"[{ci[0]:>8.2f}, {ci[1]:>8.2f}]  {ts_:>8.2f}{flag}")

    banner("Run complete")


if __name__ == '__main__':
    main()
