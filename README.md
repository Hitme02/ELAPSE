# ELAPSE: Entropy-Linked Autonomous Propagation and Self-Erasure

A mathematical framework for active data containment in distributed networks,
submitted to *Applied Mathematical Modelling* (Elsevier).

**Core idea:** instead of time-based deletion, ELAPSE triggers data erasure
when network-wide Shannon entropy $H(t)$ exceeds a threshold $H_c$, meaning
the data has spread too widely to be recalled.  An ensemble of five deletion
mechanisms (epidemic, stochastic-finance, biology, social cascade, and
diffusion) learns optimal voting weights via regularised Nelder-Mead
optimisation to minimise time-integrated entropy exposure (IEE).

---

## Repository layout

```
EL/
├── src/                     Python simulation codebase
│   ├── math_utils.py        entropy, sigma, Hill, first-passage functions
│   ├── networks.py          ER, BA, WS graph generators; Fiedler value
│   ├── m0_baseline.py       M0: pure diffusion (no deletion)
│   ├── m1_egdm.py           M1: EGDM entropy-gate (polynomial threshold)
│   ├── m2_epidemic.py       M2: SIR epidemic mortality
│   ├── m3_finance.py        M3: OU first-passage deletion
│   ├── m4_biology.py        M4: Hill-function cooperative mortality
│   ├── m5_social.py         M5: social cascade threshold deletion
│   ├── m6_ensemble.py       M6: ELAPSE ensemble (entropy-regularised Nelder-Mead)
│   ├── run_simulation.py    original single-seed runner (legacy)
│   ├── run_simulation_ci.py N=30 seeds + 95 % CI runner (primary)
│   ├── snap_loader.py       Stanford SNAP Gnutella downloader / parser
│   ├── convexity_check.py   Phase 2A: IEE convexity numerical check
│   ├── adversarial.py       Phase 2D: adversarial hub-throttling model
│   ├── gossip_entropy.py    Phase 2E: k-hop gossip entropy estimator
│   ├── plot_extended.py     publication figures (300 DPI, PDF + PNG)
│   └── generate_tables.py   LaTeX table cells for \input{} commands
├── run_extended.py          Master runner (phase-by-phase or all at once)
├── paper/
│   ├── ELAPSE_AMM.tex       Compilable AMM Elsevier paper (32 pages)
│   ├── ELAPSE_AMM.bib       Bibliography (35 references)
│   ├── elsarticle.cls       Elsevier article class (local copy for compilation)
│   ├── elsarticle-num.bst   Numeric citation style
│   └── tables/              Auto-generated LaTeX cell files (t{n}_{topo}_{model}.tex)
├── output/
│   ├── ci_results.pkl       Main simulation results (N=30 seeds, n=50/100/200/500)
│   ├── ci_snap.pkl          SNAP P2P real-world validation results
│   ├── adversarial_results.pkl  Phase 2D adversarial study
│   ├── gossip_results.pkl   Phase 2E gossip estimation study
│   ├── m3_analysis.pkl      Phase 2F M3 mechanistic analysis
│   ├── convexity_check.pkl  Phase 2A convexity violation data
│   ├── iee_summary.csv      Human-readable IEE table (all n, topo, model)
│   └── figures/             All publication figures (PDF + PNG, 300 DPI)
└── data/
    └── snap/                Downloaded SNAP Gnutella datasets (auto-downloaded)
```

---

## Reproducing all results

### Prerequisites

```bash
pip install numpy scipy networkx matplotlib
```

A working TeX Live installation with `pdflatex` is needed to compile the paper.
`elsarticle.cls` is included locally so no separate package installation is
required.

### Step 1 — Run all phases (approx. 100 min on a modern laptop)

```bash
python3 run_extended.py
```

To run individual phases:

```bash
python3 run_extended.py --phase 2A    # convexity check (Dirichlet simplex sampling)
python3 run_extended.py --phase 2B    # main N=30 CI simulation, all sizes/topologies
python3 run_extended.py --phase 2C    # SNAP real-world validation (downloads data)
python3 run_extended.py --phase 2D    # adversarial hub-throttling model
python3 run_extended.py --phase 2E    # k-hop gossip entropy estimation
python3 run_extended.py --phase 2F    # M3 mechanistic analysis
python3 run_extended.py --phase figs  # generate all publication figures
```

### Step 2 — Generate LaTeX table cells

```bash
python3 src/generate_tables.py
```

Creates `paper/tables/t{n}_{topo}_{model}.tex` (individual cells) and
`paper/tables/iee_table_n{n}.tex` (full tables) used by the paper's
`\input{}` commands.

### Step 3 — Compile the paper

```bash
cd paper
pdflatex ELAPSE_AMM.tex
bibtex   ELAPSE_AMM
pdflatex ELAPSE_AMM.tex
pdflatex ELAPSE_AMM.tex   # second pass resolves cross-references
```

Output: `paper/ELAPSE_AMM.pdf` (32 pages).

---

## Key empirical results

All statistics are mean ± 95 % CI over N=30 independent seeds.

### IEE at n=500 (lower is better)

| Mechanism         |       ER |       BA |       WS |
|-------------------|----------|----------|----------|
| M0 (no deletion)  | 7016 ± 32| 6960 ± 32| 6939 ± 31|
| M1 (EGDM)         |  462 ± 1 |  -- |  -- |
| M4 (Biology-Hill) |  532 ± 2 |  521 ± 2 |  522 ± 2 |
| M5 (Social)       |  454 ± 2 |  -- |  -- |
| **M6 (ELAPSE)**   | **469 ± 2**| **627 ± 2**| **624 ± 2**|

M6 achieves ≥ 90 % IEE reduction vs M0 on all combinations.
At n=500, weight collapse (M4 weight ≈ 0.88–1.00 on BA/WS) means M6
approximates M4 rather than a true ensemble — a known limitation documented
in the paper (§4.4).

### SNAP real-world validation (n=500 subgraphs)

| Mechanism | Gnutella08 (λ₂=0.303) | Gnutella31 (λ₂=0.071) |
|-----------|----------------------|----------------------|
| M0        |    6960 ± 32         |     6780 ± 30        |
| M4        |     523 ± 2          |      518 ± 2         |
| M5        |     473 ± 3          |      533 ± 9         |
| M6        |     619 ± 2          |      670 ± 3         |

M6/M0 IEE ratio ≥ 9× on both real-world datasets.

---

## Simulation design

| Parameter | Value |
|-----------|-------|
| Integration | Euler-Maruyama SDE |
| Time horizon T | 15 time units |
| Step size dt | 0.02 |
| Noise | CIR-type: σ_n √max(x,0) dW, σ_n=0.015 |
| Seeds N | 30 per (topology × size × mechanism) |
| CI | scipy.stats.t.interval, α=0.05, df=29 |
| Weight learning | Nelder-Mead, 2 restarts, λ_reg=15, dt=0.05 for speed |
| Network sizes | 50, 100, 200, 500 |
| Topologies | ER (p=0.15), BA (m=3), WS (k=6, p=0.1) |
| SNAP subgraphs | BFS-seeded, 500 nodes from Gnutella08/31 |

---

## Mathematical notes

**Lemma 1 (convexity):** The paper restricts the IEE convexity proof to the
*linearised* system (σ(H) = const).  A numerical check on 1000 random weight
pairs drawn from the Dirichlet(1,...,1) simplex finds a **14.1 % Jensen
inequality violation rate** (maximum violation ≈ 109) under the full nonlinear
SDE.  Weight learning is therefore a non-convex problem; multiple Nelder-Mead
restarts are used to avoid local minima.

**M3 underperformance:** The Finance-OU vote v₃(t) = 1−exp(−λ_fp t) uses the
*static* Fiedler value λ₂.  On BA networks with small λ₂ ≈ 1.28, v₃ grows
too slowly to match rapid hub-driven entropy growth.  A time-varying spectral
gap is proposed as a fix (§8 of the paper).

**Adversarial model:** At f=0.3 (30 % adversarial nodes throttling outgoing
diffusion by 70 %), IEE increases by 4.5 % on BA and 2.5 % on ER.  An
adaptive H_c countermeasure (lowering the deletion threshold when entropy
growth is anomalously slow) recovers ≈35 % of the degradation on BA.

**Gossip estimation:** k=3 hop neighbourhood estimates achieve 0 % false-early
and < 0.1 % missed-trigger rates on ER and BA.  The Watts-Strogatz topology
requires larger k due to ring-lattice clustering (23.5 % missed at k=3).
