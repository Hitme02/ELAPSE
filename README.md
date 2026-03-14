# ELAPSE (Entropy-Linked Autonomous Propagation and Self-Erasure)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains the official codebase and simulation models for **ELAPSE**, an active data containment framework designed for decentralized networks. The research introduces a method for controlling digital proliferation through geometric entropy metrics, substituting traditional chronological decay mechanisms with spatial-topological self-erasure constraints.

## Abstract

The restriction of information flow across stochastic, decentralized networks presents a significant challenge for ongoing data privacy. Current cryptographic access controls are frequently bypassed post-exfiltration, and static self-destructing protocols generally rely on chronological decay sequences that do not account for the topological realities of viral spread. In this study, we introduce the ELAPSE framework, shifting from chronological data lifetimes toward spatial-topological containment. By measuring the Shannon entropy of the data's spatial dispersion, ELAPSE links the mortality rate directly to structural privacy loss. The core system optimization blends five distinct mechanistic domains (biological, epidemiological, financial, social, and physical) via entropy-regularised Nelder-Mead simplex optimization.

## Repository Structure

```
ELAPSE/
├── src/                # Core Python implementation
│   ├── networks.py     # Graph topologies (ER, BA, WS)
│   ├── math_utils.py   # Matrix Laplacians, entropy equations
│   ├── m0_baseline.py  # Uncontained theoretical diffusion
│   ├── m1_egdm.py      # Entropy-Gated Diffusion Model (Physical)
│   ├── m2_epidemic.py  # Compartmental SIR (Epidemiological)
│   ├── m3_finance.py   # Stochastic First-Passage (Financial)
│   ├── m4_biology.py   # Cooperative Binding (Systems Biology)
│   ├── m5_social.py    # Cascade Pressure (Sociological)
│   ├── m6_ensemble.py  # ELAPSE Optimized Weighted Projection
│   ├── run_simulation.py
│   └── plot_results.py
├── paper/              # LaTeX source code and full compiled PDF
├── output/             # Auto-generated experiment dumps and graphs
├── run_all.py          # Master CLI entrypoint
└── requirements.txt    # Python dependencies
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Full Scale Study** (Generates data and figures for $n \in [50, 100, 200, 500]$)
   ```bash
   python run_all.py
   ```

3. **Run Smoke Test** (Verifies build viability sequentially at $n=50$ limit)
   ```bash
   python run_all.py --quick
   ```

## Results Extraction
Submitting the master simulation triggers internal plotting capabilities housed in `src/plot_results.py`. Result arrays output into `output/results.pkl`. Matplotlib representations directly correlating to spatial graphs are written to `output/figures/` (e.g. `fig3_iee_n500.png`).
