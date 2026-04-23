# ELAPSE

Entropy-Linked Autonomous Propagation and Self-Erasure (ELAPSE) is a
networked stochastic dynamics framework for controlled information propagation.
The core modeling idea is entropy-triggered deletion: once normalized entropy
crosses a threshold, deletion pressure is activated and propagation is damped.

This repository contains:

- the active experiment and theorem workflow in `elapse/`
- the still-required mechanism and network implementation layer in `src/`
- manuscript sources and build tooling in `paper/`
- reproducibility scripts and datasets

## 1. Repository At A Glance

Top-level layout:

- `elapse/` active package for experiments, theory, tests, and core SDE utilities
- `src/` mechanism implementations and helper modules imported by active scripts
- `paper/` manuscript sources, build tooling, generated tables, and built PDFs
- `data/snap/` SNAP network cache for real-world validation
- `assets/` supporting static assets
- `reproduce_all.sh` one-command end-to-end reproduction script
- `requirements.txt` Python dependency pin set

## 2. Environment Setup

### Python

Recommended: Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

Current pinned dependencies:

- numpy==1.26.4
- scipy==1.12.0
- matplotlib==3.8.3
- networkx==3.2.1
- requests==2.31.0
- seaborn==0.13.2

### LaTeX Toolchain

For paper builds, you need these executables on PATH:

- `pdflatex`
- `bibtex`

## 3. Fastest Reproduction Path

Run everything from repository root:

```bash
bash reproduce_all.sh
```

Pipeline stages in `reproduce_all.sh`:

1. Main 5-fold evaluation (`elapse/experiments/main_iee/proper_evaluation.py`)
2. Theorem 2 stochastic bound verification (`elapse/theory/bounds.py`)
3. Corollary 1 Ito-corrected verification (`elapse/theory/corollary1_verify.py`)
4. Non-negativity check (`elapse/tests/test_nonnegativity.py`)
5. Evolving-network experiment (`elapse/experiments/evolving_network/evolving_network_exp.py`)
6. Hc robustness sweep (`elapse/experiments/adversarial/hc_robustness.py`)
7. Extended real-world validation + PhysicaA manuscript compile

## 4. Complete Script And Module Reference

### 4.1 Active ELAPSE Package (`elapse/`)

Core package files:

- `elapse/__init__.py`

Core algorithms:

- `elapse/core/__init__.py`
- `elapse/core/ensemble/__init__.py`
- `elapse/core/mechanisms/__init__.py`
- `elapse/core/mechanisms/m3_finance_derived.py`
- `elapse/core/network/__init__.py`
- `elapse/core/sde/__init__.py`
- `elapse/core/sde/stochastic_bound.py`

Experiments:

- `elapse/experiments/__init__.py`
- `elapse/experiments/main_iee/__init__.py`
- `elapse/experiments/main_iee/proper_evaluation.py`
- `elapse/experiments/adversarial/__init__.py`
- `elapse/experiments/adversarial/hc_robustness.py`
- `elapse/experiments/evolving_network/__init__.py`
- `elapse/experiments/evolving_network/evolving_network_exp.py`
- `elapse/experiments/realworld/__init__.py`
- `elapse/experiments/realworld/extended_validation.py`
- `elapse/experiments/realworld/random_walk_sampler.py`
- `elapse/experiments/gossip/__init__.py`
- `elapse/experiments/phase_diagram/__init__.py`
- `elapse/experiments/scaling/__init__.py`

Theory verification:

- `elapse/theory/__init__.py`
- `elapse/theory/bounds.py`
- `elapse/theory/corollary1_verify.py`

Tests:

- `elapse/tests/__init__.py`
- `elapse/tests/test_ensemble.py`
- `elapse/tests/test_mechanisms.py`
- `elapse/tests/test_nonnegativity.py`
- `elapse/tests/test_theorems.py`

Figures/tables package placeholders:

- `elapse/figures/__init__.py`
- `elapse/tables/__init__.py`

### 4.2 Legacy-But-Required Mechanism Layer (`src/`)

These files are still actively imported by `elapse/experiments/*` and
`elapse/tests/*`, so they are part of the working runtime.

- `src/adversarial.py`
- `src/convexity_check.py`
- `src/degree_scaling.py`
- `src/epidemic_threshold.py`
- `src/generate_tables.py`
- `src/gossip_entropy.py`
- `src/m0_baseline.py`
- `src/m1_egdm.py`
- `src/m2_epidemic.py`
- `src/m3_finance.py`
- `src/m4_biology.py`
- `src/m5_social.py`
- `src/m6_ensemble.py`
- `src/m7_timer.py`
- `src/math_utils.py`
- `src/mechanism_diversity.py`
- `src/networks.py`
- `src/phase_diagram.py`
- `src/plot_extended.py`
- `src/plot_results.py`
- `src/realworld_extended.py`
- `src/run_simulation.py`
- `src/run_simulation_ci.py`
- `src/scaling_n1000.py`
- `src/snap_loader.py`

### 4.3 Paper System (`paper/`)

Build tooling:

- `paper/compile.py`

Build directories:

- `paper/tex/` source `.tex` files and bibliography material
- `paper/tables/` generated and hand-maintained table snippets used by TeX
- `paper/build/` moved intermediate TeX build artifacts (`.aux`, `.bbl`, etc.)
- `paper/pdf/` final compiled PDF outputs

Compiled targets supported by `paper/compile.py`:

- AMM -> `ELAPSE_AMM.tex`
- CNSNS -> `ELAPSE_CNSNS.tex`
- PhysicaA -> `ELAPSE_PhysicaA.tex`
- CSF -> `ELAPSE_CSF.tex`
- IEEE -> `ELAPSE_ieee.tex`
- changes -> `changes_summary.tex`

### 4.4 Reproducibility Scripts

- `reproduce_all.sh` full pipeline orchestrator

## 5. Command Cookbook

### Main evaluation and tables

```bash
python elapse/experiments/main_iee/proper_evaluation.py
```

Writes:

- `output/proper_eval_results.pkl`
- `paper/tables/table4_main_iee.tex`
- `paper/tables/table5_welch_tests.tex`

### Theorem 2 stochastic bound verification

```bash
python elapse/theory/bounds.py
```

Writes:

- `output/theorem2_stochastic_verification.csv`

### Corollary 1 Ito-corrected verification

```bash
python elapse/theory/corollary1_verify.py
```

Writes:

- `output/corollary1_n50.pkl`
- `output/corollary1_n100.pkl`
- `output/corollary1_n200.pkl`

### Non-negativity check

```bash
python elapse/tests/test_nonnegativity.py
```

### Robustness to threshold misspecification

```bash
python elapse/experiments/adversarial/hc_robustness.py
```

Writes:

- `output/hc_robustness_results.pkl`

### Extended real-world validation

```bash
python elapse/experiments/realworld/extended_validation.py
```

Writes:

- `output/table_realworld_extended.csv`
- `output/realworld_extended_results.pkl`
- `paper/tables/table_realworld_extended.tex`

### Evolving-network experiment

```bash
python elapse/experiments/evolving_network/evolving_network_exp.py
```

### Full test suite

```bash
pytest elapse/tests -q
```

### Build paper PDFs

All targets:

```bash
python paper/compile.py
```

Single target example:

```bash
python paper/compile.py PhysicaA
```

## 6. Data, Outputs, And Artifacts Policy

### Input data

- `data/snap/` stores downloaded SNAP resources (for example CA-GrQc cache)

### Generated outputs

- `output/` contains numerical experiment/theory outputs
- `paper/pdf/` contains generated manuscript PDFs
- `paper/build/` contains intermediate TeX build files

### Source of truth

- Code and manuscript sources are the source of truth.
- Generated outputs can be deleted and regenerated from scripts in this README.
- PDFs are intentionally placed in `paper/pdf/` as a dedicated distribution folder.

## 7. Architecture Notes

The repository intentionally keeps both `elapse/` and `src/`:

- `elapse/` gives cleaner package organization for current experiments/theory.
- `src/` still contains implementation modules imported by active entry points.

Removing `src/` currently breaks the runtime unless imports are fully migrated.

## 8. Reproducibility Expectations

Typical runtime (machine-dependent):

- Main 5-fold evaluation can take hours.
- Theory checks and robustness studies can take tens of minutes to hours.
- Paper compilation is usually minutes once dependencies are installed.

For quicker smoke checks, run individual scripts rather than the full pipeline.

## 9. Troubleshooting

If a script fails with missing module errors:

1. Confirm dependencies: `pip install -r requirements.txt`
2. Run commands from repository root (important for relative imports)

If paper build fails:

1. Confirm `pdflatex` and `bibtex` are available in shell PATH
2. Re-run compile command; artifacts are managed by `paper/compile.py`

If real-world validation cannot download SNAP data:

- The code includes fallback logic for CA-GrQc loading and synthetic substitutes.

## 10. Minimal Daily Workflow

If you only want the current paper-critical path:

1. Run `python elapse/experiments/main_iee/proper_evaluation.py`
2. Run `python elapse/theory/bounds.py`
3. Run `python elapse/theory/corollary1_verify.py`
4. Run `python elapse/experiments/realworld/extended_validation.py`
5. Run `python paper/compile.py PhysicaA`

That sequence regenerates the core validation artifacts and the PhysicaA PDF.
