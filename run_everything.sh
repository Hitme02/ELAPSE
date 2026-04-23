#!/bin/bash
# ELAPSE Full Reproduction Pipeline
#
# Runs every experiment, figure, and M6-dominance analysis,
# then compiles all manuscript PDF targets.
#
# All Python scripts use multiprocessing internally where applicable.
#
# Usage:
#   bash run_everything.sh
#
# Estimated runtime: 4-8 hours depending on CPU core count.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/venv/bin/python"
else
  PYTHON_BIN="python"
fi

NCPUS=$(python -c "import multiprocessing; print(multiprocessing.cpu_count())" 2>/dev/null || echo "?")

run_step() {
  local label="$1"; shift
  echo ""
  echo "============================================================"
  echo "$label"
  echo "============================================================"
  "$@"
}

echo "ELAPSE Full Pipeline"
echo "Root:   $ROOT"
echo "Python: $PYTHON_BIN"
echo "CPUs:   $NCPUS"
echo "Start:  $(date)"

mkdir -p output output/figures paper/build paper/pdf paper/tables

# ── THEORY & TESTS (fast) ────────────────────────────────────────────────────

run_step "[1/19] Main IEE evaluation — 5-fold CV, parallel (elapse)" \
  "$PYTHON_BIN" elapse/experiments/main_iee/proper_evaluation.py

run_step "[2/19] Theorem 2 bound verification" \
  "$PYTHON_BIN" elapse/theory/bounds.py

run_step "[3/19] Corollary 1 verification" \
  "$PYTHON_BIN" elapse/theory/corollary1_verify.py

run_step "[4/19] Non-negativity test" \
  "$PYTHON_BIN" elapse/tests/test_nonnegativity.py

# ── CORE EXPERIMENTS (parallel internally) ───────────────────────────────────

run_step "[5/19] Evolving-network experiment — parallel seeds (elapse)" \
  "$PYTHON_BIN" elapse/experiments/evolving_network/evolving_network_exp.py

run_step "[6/19] H_c robustness sweep — parallel (hc_frac × seed) (elapse)" \
  "$PYTHON_BIN" elapse/experiments/adversarial/hc_robustness.py

run_step "[7/19] Extended real-world validation (elapse)" \
  "$PYTHON_BIN" elapse/experiments/realworld/extended_validation.py

# ── FIGURE DATA GENERATION (parallel internally) ─────────────────────────────

run_step "[8/19] CI simulation suite — parallel test seeds (src)" \
  "$PYTHON_BIN" src/run_simulation_ci.py

# ── FIGURE GENERATION ────────────────────────────────────────────────────────

run_step "[9/19] Core/extended figure generation (src)" \
  "$PYTHON_BIN" src/plot_extended.py

run_step "[10/19] Phase diagram figure (src)" \
  "$PYTHON_BIN" src/phase_diagram.py

run_step "[11/19] Degree scaling figure (src)" \
  "$PYTHON_BIN" src/degree_scaling.py

run_step "[12/19] Epidemic threshold figure (src)" \
  "$PYTHON_BIN" src/epidemic_threshold.py

run_step "[13/19] Mechanism diversity figure (src)" \
  "$PYTHON_BIN" src/mechanism_diversity.py

run_step "[14/19] Scaling N=1000 + crossover n* (src)" \
  "$PYTHON_BIN" src/scaling_n1000.py

run_step "[15/19] Real-world extended figure (src)" \
  "$PYTHON_BIN" src/realworld_extended.py

# ── M6 DOMINANCE EVIDENCE (parallel internally) ──────────────────────────────

run_step "[16/19] Cross-topology generalisation benchmark — parallel (src)" \
  "$PYTHON_BIN" src/topology_transfer.py

run_step "[17/19] Early deletion speed analysis — parallel (src)" \
  "$PYTHON_BIN" src/early_deletion_analysis.py

run_step "[18/19] Multi-metric M6 dominance scorecard (src)" \
  "$PYTHON_BIN" src/m6_scorecard.py

# ── MANUSCRIPT COMPILATION ───────────────────────────────────────────────────

run_step "[19/19] Compile all manuscript PDF targets" \
  "$PYTHON_BIN" paper/compile.py

echo ""
echo "============================================================"
echo "Pipeline completed successfully."
echo "End: $(date)"
echo "============================================================"
echo ""
echo "Key outputs"
echo "-----------"
echo ""
echo "Pickles (reusable data):"
echo "  output/proper_eval_results.pkl"
echo "  output/hc_robustness_results.pkl"
echo "  output/evolving_network_results.pkl"
echo "  output/topology_transfer_results.pkl"
echo "  output/early_deletion_results.pkl"
echo "  output/simulation_ci_results.pkl"
echo ""
echo "M6 dominance figures:"
echo "  output/figures/fig_topology_transfer.pdf"
echo "  output/figures/fig_iee_reliability.pdf"
echo "  output/figures/fig_time_to_half_mass.pdf"
echo "  output/figures/fig_deletion_speed.pdf"
echo "  output/figures/fig_phase_iee.pdf"
echo "  output/figures/fig_m6_dominance_radar.pdf"
echo "  output/figures/fig13_scaling_n1000.pdf   (with crossover n*)"
echo ""
echo "LaTeX tables:"
echo "  paper/tables/table4_main_iee.tex"
echo "  paper/tables/table_reliability.tex"
echo "  paper/tables/table5_welch_tests.tex"
echo "  paper/tables/table_topology_transfer.tex"
echo "  paper/tables/table_deletion_speed.tex"
echo "  paper/tables/table_m6_scorecard.tex"
echo ""
echo "PDFs:"
echo "  paper/pdf/ELAPSE_PhysicaA.pdf"
echo "  paper/pdf/ELAPSE_CNSNS.pdf"
echo "  paper/pdf/ELAPSE_CSF.pdf"
echo "  paper/pdf/ELAPSE_AMM.pdf"
echo "  paper/pdf/ELAPSE_ieee.pdf"
