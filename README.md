# ELAPSE: Entropy-Linked Autonomous Propagation and Self-Erasure

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Welcome to the official repository for the **ELAPSE** framework. This project is a mathematical simulation engine designed to solve a major problem in data privacy: **how do we stop sensitive data from spreading forever once it leaves our hands?**

---

## 🛑 The Problem: Why Current Systems Fail

Imagine you send a highly sensitive photo or document over a decentralized network. 
Normally, people try to secure this by using **Time-Based Deletion** (like Snapchat, or protocols like *Vanish*). They attach a timer: *"Delete this file after 8 hours."*

**The flaw:** On a massive, highly connected network, 8 hours is an eternity. A viral file can be copied to thousands of computers in minutes. By the time the 8-hour clock runs out, the data is already everywhere. The "timer" has no idea what is actually happening in the real world.

## 💡 The ELAPSE Solution: Deleting Based on "Space", Not "Time"

**ELAPSE** throws away the clock. Instead of deleting data based on *time*, it deletes data based on *space* (how widely it has spread). 

We measure this spread using a mathematical concept called **Shannon Entropy**. If the data stays contained to a few trusted computers, the entropy is low, and the data stays alive. But the second the data starts "going viral" and leaking to too many computers, the entropy spikes. **ELAPSE detects this spike and triggers a network-wide self-erasure protocol.** It's like a fire sprinkler system that turns on the exact moment it senses too much smoke.

---

## ⚙️ How It Works: The 5 Core Mechanisms

Because computer networks come in all shapes and sizes, a single "delete switch" doesn't always work. To fix this, ELAPSE acts as an "ensemble"—it combines the math from **5 different scientific fields** to vote on when the data should kill itself:

1. **The Physics Model (Diffusion):** Treats data like heat spreading across a metal plate.
2. **The Epidemiology Model (SIR):** Treats data like a virus spreading through a population.
3. **The Finance Model (First-Passage Time):** Treats the data's survival like a fluctuating stock market price hitting a stop-loss limit.
4. **The Systems Biology Model (Hill Function):** Treats the deletion trigger like a cooperative protein binding in a living cell.
5. **The Sociology Model (Information Cascade):** Assumes computers will delete the data if they see enough of their "neighbors" deleting it (herd behavior).

### The "Brain" (Nelder-Mead Optimization)
Instead of guessing which of these 5 models is best, ELAPSE uses an AI optimization algorithm. It continuously tests the network and assigns weights to each of the 5 models, blending them together. This guarantees the tightest possible containment no matter what network the data is on.

---

## 📊 Key Results

Through extensive simulations on networks of up to 500 nodes (using Erdős–Rényi, Barabási–Albert, and Watts–Strogatz graphs):
- **Doing Nothing (Baseline):** Privacy exposure skyrockets to a massive score of **7,055**.
- **ELAPSE Ensemble:** Privacy exposure is rapidly crushed and bounded to just **470**.
- **Result:** A **93.3% reduction** in privacy loss compared to uncontained networks.

---

## 📂 Repository Structure

```text
ELAPSE/
├── src/                # Core Python Simulation Engine
│   ├── networks.py     # Graph topologies (ER, BA, WS)
│   ├── math_utils.py   # Matrix Laplacians, entropy equations
│   ├── m0_baseline.py  # Model 0: Uncontained theoretical diffusion
│   ├── m1_egdm.py      # Model 1: Physical Diffusion
│   ├── m2_epidemic.py  # Model 2: Epidemiological SIR
│   ├── m3_finance.py   # Model 3: Financial Stochastic
│   ├── m4_biology.py   # Model 4: Biological Switch
│   ├── m5_social.py    # Model 5: Sociological Cascade
│   ├── m6_ensemble.py  # Model 6: The ELAPSE Brain (Optimization)
│   ├── run_simulation.py
│   └── plot_results.py
├── paper/              # LaTeX source code and the final compiled Publication PDF
├── output/             # Auto-generated experiment dumps and graphical charts
├── run_all.py          # Master CLI entrypoint to run the entire project
└── requirements.txt    # Python dependencies
```

---

## 🚀 Quick Start Guide

Want to run the simulations yourself and generate the graphs from the paper? It's incredibly easy.

**1. Install Dependencies**
Make sure you have Python installed, then install the required math and graphing libraries:
```bash
pip install -r requirements.txt
```

**2. Run a Quick Smoke Test**
If you just want to verify the code works without waiting for large graphs to render, run a quick limit test (50 computers):
```bash
python run_all.py --quick
```

**3. Run the Full Scale Study**
This will execute the entire ELAPSE framework across all network topologies and all mathematical models. **(Note: This takes a few minutes to complete).**
```bash
python run_all.py
```

## 📈 Viewing Results
Once the simulation finishes running, head into the `output/figures/` folder. The engine will automatically generate all the beautiful Trajectory, Bar, Heatmap, and Scaling charts seen in the official paper!
