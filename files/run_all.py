"""
run_all.py  —  QUORUM Framework: full simulation + plots in one script.
"""

import sys, os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, '/home/claude/quorum')
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── imports ────────────────────────────────────────────────────────────────
from utils.networks   import get_all_networks, fiedler_value
from utils.math_utils import entropy, max_entropy, sigma_egdm, sigma_hill, first_passage_prob
import models.m1_egdm    as m1
import models.m2_epidemic as m2
import models.m3_finance  as m3
import models.m4_biology  as m4
import models.m5_social   as m5
from ensemble.m6_ensemble import simulate as sim_m6, learn_weights, DEFAULT_THETA

OUT = '/home/claude/quorum/output'
FIG = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

# ── Shared config ──────────────────────────────────────────────────────────
SIZES      = [50, 100]
T          = 10.0
DT         = 0.05
STOCHASTIC = True

MODEL_ORDER  = ['M1_EGDM','M2_Epidemic','M3_Finance','M4_Biology','M5_Social','M6_Ensemble']
TOPO_ORDER   = ['Erdos-Renyi','Barabasi-Albert','Watts-Strogatz']
TOPO_SHORT   = {'Erdos-Renyi':'ER','Barabasi-Albert':'BA','Watts-Strogatz':'WS'}
MODEL_LABELS = {
    'M1_EGDM':     'M1: EGDM (Baseline)',
    'M2_Epidemic': 'M2: Epidemic-Entropy',
    'M3_Finance':  'M3: Stochastic-Finance',
    'M4_Biology':  'M4: Bio-Switch (Hill)',
    'M5_Social':   'M5: Cascade-Entropy',
    'M6_Ensemble': 'M6: Ensemble (QUORUM)',
}
COLORS = {
    'M1_EGDM':     '#1A56A4',
    'M2_Epidemic': '#E05A2B',
    'M3_Finance':  '#2A9D8F',
    'M4_Biology':  '#D4A017',
    'M5_Social':   '#9B59B6',
    'M6_Ensemble': '#E63946',
}

# ── Parameter builder ──────────────────────────────────────────────────────
def make_params(n, rng):
    H_max = max_entropy(n)
    H_c   = 0.65 * H_max
    s     = np.zeros(n)
    s[rng.choice(n, max(1, n//5), replace=False)] = 0.05
    return dict(
        alpha=0.3, mu=1.5, H_c=H_c, beta=2.0,
        n_hill=4.0, beta_sir=0.4, gamma=0.15,
        theta_ou=0.3, mu_ou=0.1, sigma_ou=0.04,
        kappa=0.3, deletion_threshold=0.05,
        sigma_noise=0.015, s=s,
        T_train=T, dt=DT,
    )

def make_x0(n, rng):
    x0 = np.zeros(n)
    seeds = rng.choice(n, max(1, n//10), replace=False)
    x0[seeds] = rng.uniform(0.5, 1.0, len(seeds))
    return x0

def adj_from_lap(L):
    A = -(L - np.diag(np.diag(L)))
    return np.abs(A)

def metrics(t_arr, H_arr, M_arr):
    M0     = max(M_arr[0], 1e-9)
    below  = np.where(M_arr < 0.5 * M0)[0]
    t_star = float(t_arr[below[0]]) if len(below) else float(t_arr[-1])
    IEE    = float(np.sum(H_arr * M_arr) * DT)
    return dict(t_star=t_star, IEE=IEE,
                final_mass=float(M_arr[-1]), final_H=float(H_arr[-1]),
                t_arr=t_arr, H_arr=H_arr, M_arr=M_arr)

# ══════════════════════════════════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════
results = {}

for n in SIZES:
    results[n] = {}
    rng  = np.random.default_rng(42 + n)
    nets = get_all_networks(n)
    print(f"\n{'='*60}  n={n}")

    for topo, (G, L) in nets.items():
        results[n][topo] = {}
        A       = adj_from_lap(L)
        lam2    = fiedler_value(L)
        params  = make_params(n, rng)
        x0      = make_x0(n, rng)
        print(f"\n  {topo}  λ₂={lam2:.3f}")

        # M1
        print("    M1 EGDM ...", end=' ', flush=True)
        t,_,H,M = m1.simulate(x0, L, params, T=T, dt=DT, stochastic=STOCHASTIC)
        r = metrics(t,H,M); r['lambda2'] = lam2
        results[n][topo]['M1_EGDM'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

        # M2
        print("    M2 Epidemic ...", end=' ', flush=True)
        t,_,H,M = m2.simulate(x0, A, L, params, T=T, dt=DT, stochastic=STOCHASTIC)
        r = metrics(t,H,M)
        results[n][topo]['M2_Epidemic'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

        # M3
        print("    M3 Finance ...", end=' ', flush=True)
        t,_,H,M = m3.simulate(x0, L, params, lam2, T=T, dt=DT, stochastic=STOCHASTIC)
        r = metrics(t,H,M)
        results[n][topo]['M3_Finance'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

        # M4
        print("    M4 Biology ...", end=' ', flush=True)
        t,_,H,M = m4.simulate(x0, L, params, T=T, dt=DT, stochastic=STOCHASTIC)
        r = metrics(t,H,M)
        results[n][topo]['M4_Biology'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

        # M5
        print("    M5 Social ...", end=' ', flush=True)
        t,_,H,M = m5.simulate(x0, L, A, params, T=T, dt=DT, stochastic=STOCHASTIC)
        r = metrics(t,H,M)
        results[n][topo]['M5_Social'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

        # M6 — learn weights then simulate
        print("    M6 learning weights ...", flush=True)
        train = [(x0, L, A, params, lam2)]
        w, _  = learn_weights(train, verbose=True)

        print("    M6 Ensemble ...", end=' ', flush=True)
        t,_,H,M,Delta,votes = sim_m6(
            x0, L, A, params, lam2,
            weights=w, T=T, dt=DT, stochastic=STOCHASTIC
        )
        r = metrics(t,H,M)
        r['learned_weights'] = w
        r['Delta_arr']       = Delta
        r['votes_arr']       = votes
        results[n][topo]['M6_Ensemble'] = r
        print(f"IEE={r['IEE']:.2f}  t*={r['t_star']:.2f}")

# Save
with open(f'{OUT}/results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("\nResults saved.")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)
for n in SIZES:
    print(f"\n── n = {n} ──")
    print(f"  {'Model':<22} {'Topology':<22} {'IEE':>8} {'t*':>8} {'FinalM':>8}")
    print("  " + "-"*68)
    for topo in TOPO_ORDER:
        for m in MODEL_ORDER:
            r   = results[n][topo].get(m, {})
            iee = r.get('IEE', float('nan'))
            ts  = r.get('t_star', float('nan'))
            fm  = r.get('final_mass', float('nan'))
            tag = " ◀" if m == 'M6_Ensemble' else ""
            print(f"  {m:<22} {topo:<22} {iee:>8.2f} {ts:>8.2f} {fm:>8.4f}{tag}")
        print()

# ══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.grid':True,'grid.alpha':0.25,'figure.dpi':150,
})

n_plot = max(results.keys())
data   = results[n_plot]

# ── Fig 1: Entropy trajectories ───────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(f'Entropy Trajectories H(t)   n = {n_plot}', fontsize=14, fontweight='bold')
H_c_line = 0.65 * max_entropy(n_plot)

for ax, topo in zip(axes, TOPO_ORDER):
    lam2 = data[topo]['M1_EGDM'].get('lambda2','?')
    for m in MODEL_ORDER:
        if m not in data[topo]: continue
        r  = data[topo][m]
        lw = 2.8 if m == 'M6_Ensemble' else 1.6
        ls = '-'  if m in ('M6_Ensemble','M1_EGDM') else '-'
        zo = 5    if m == 'M6_Ensemble' else 2
        ax.plot(r['t_arr'], r['H_arr'], color=COLORS[m],
                label=MODEL_LABELS[m], lw=lw, ls=ls, alpha=0.9, zorder=zo)
    ax.axhline(H_c_line, color='#333', ls=':', lw=1.2, alpha=0.6, label='$H_c$')
    ax.set_title(f"{TOPO_SHORT[topo]}  (λ₂={lam2:.3f})", fontsize=12)
    ax.set_xlabel('Time t')
axes[0].set_ylabel('H(p(t))')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5,-0.08), fontsize=9, frameon=True)
plt.tight_layout()
p = f'{FIG}/fig1_entropy_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"\nSaved {p}")

# ── Fig 2: Mass decay ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(f'Mass Decay M(t)/M(0)   n = {n_plot}', fontsize=14, fontweight='bold')
for ax, topo in zip(axes, TOPO_ORDER):
    for m in MODEL_ORDER:
        if m not in data[topo]: continue
        r  = data[topo][m]
        M0 = max(r['M_arr'][0], 1e-9)
        lw = 2.8 if m == 'M6_Ensemble' else 1.6
        zo = 5   if m == 'M6_Ensemble' else 2
        ax.plot(r['t_arr'], r['M_arr']/M0, color=COLORS[m],
                label=MODEL_LABELS[m], lw=lw, alpha=0.9, zorder=zo)
    ax.axhline(0.5, color='#333', ls=':', lw=1.0, alpha=0.5, label='50% mass')
    ax.set_title(TOPO_SHORT[topo], fontsize=12)
    ax.set_xlabel('Time t')
    ax.set_ylim(-0.05, 1.15)
axes[0].set_ylabel('M(t) / M(0)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5,-0.08), fontsize=9, frameon=True)
plt.tight_layout()
p = f'{FIG}/fig2_mass_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"Saved {p}")

# ── Fig 3: IEE bar chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x_pos   = np.arange(len(TOPO_ORDER))
nm      = len(MODEL_ORDER)
width   = 0.12
offsets = np.linspace(-(nm-1)/2, (nm-1)/2, nm) * width
for j, m in enumerate(MODEL_ORDER):
    vals = [data[topo].get(m,{}).get('IEE',0) for topo in TOPO_ORDER]
    lw   = 1.5 if m == 'M6_Ensemble' else 0.4
    ec   = 'black' if m == 'M6_Ensemble' else 'white'
    bars = ax.bar(x_pos + offsets[j], vals, width=width*0.92,
                  color=COLORS[m], label=MODEL_LABELS[m],
                  alpha=0.88, edgecolor=ec, linewidth=lw)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7, rotation=40)
ax.set_xticks(x_pos)
ax.set_xticklabels([TOPO_SHORT[t] for t in TOPO_ORDER], fontsize=12)
ax.set_ylabel('Time-Integrated Entropy Exposure (IEE)')
ax.set_title(f'IEE Comparison — All Models & Topologies  (n={n_plot})\nLower = better privacy protection', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
p = f'{FIG}/fig3_iee_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"Saved {p}")

# ── Fig 4: t* bar chart ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
for j, m in enumerate(MODEL_ORDER):
    vals = [data[topo].get(m,{}).get('t_star',0) for topo in TOPO_ORDER]
    lw   = 1.5 if m == 'M6_Ensemble' else 0.4
    ec   = 'black' if m == 'M6_Ensemble' else 'white'
    ax.bar(x_pos + offsets[j], vals, width=width*0.92,
           color=COLORS[m], label=MODEL_LABELS[m],
           alpha=0.88, edgecolor=ec, linewidth=lw)
ax.set_xticks(x_pos)
ax.set_xticklabels([TOPO_SHORT[t] for t in TOPO_ORDER], fontsize=12)
ax.set_ylabel('Mortality Activation Time t*')
ax.set_title(f'Mortality Activation Time  (n={n_plot})\nLower = faster deletion', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
p = f'{FIG}/fig4_tstar_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"Saved {p}")

# ── Fig 5: Vote evolution (M6 on ER network) ──────────────────────────────
topo_v = 'Erdos-Renyi'
m6d    = data[topo_v].get('M6_Ensemble', {})
if 'votes_arr' in m6d:
    votes  = m6d['votes_arr']
    t_arr  = m6d['t_arr']
    Delta  = m6d['Delta_arr']
    vlabels = ['v₁: EGDM','v₂: Epidemic','v₃: Finance','v₄: Biology','v₅: Social']
    vcols   = [COLORS[m] for m in MODEL_ORDER[:5]]

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(11,7), sharex=True)
    fig.suptitle(f'Ensemble Vote Evolution — {TOPO_SHORT[topo_v]}  n={n_plot}',
                 fontsize=13, fontweight='bold')
    for k in range(5):
        ax1.plot(t_arr, votes[:,k], label=vlabels[k],
                 color=vcols[k], lw=1.8, alpha=0.9)
    ax1.axhline(DEFAULT_THETA, color='#333', ls=':', lw=1.2, alpha=0.6, label=f'θ={DEFAULT_THETA}')
    ax1.set_ylabel('Individual Vote vₖ(t)')
    ax1.legend(fontsize=9, ncol=3)
    ax1.set_ylim(-0.02, 1.08)

    ax2.plot(t_arr, Delta, color=COLORS['M6_Ensemble'], lw=2.5, label='Δ(t) weighted vote')
    ax2.fill_between(t_arr, DEFAULT_THETA, Delta,
                     where=Delta>=DEFAULT_THETA, alpha=0.25,
                     color=COLORS['M6_Ensemble'], label='Deletion active')
    ax2.axhline(DEFAULT_THETA, color='#333', ls=':', lw=1.2, alpha=0.6, label=f'θ={DEFAULT_THETA}')
    ax2.set_ylabel('Δ(t) Ensemble Vote')
    ax2.set_xlabel('Time t')
    ax2.legend(fontsize=9)
    ax2.set_ylim(-0.02, 1.08)

    plt.tight_layout()
    p = f'{FIG}/fig5_votes_n{n_plot}.png'
    plt.savefig(p, bbox_inches='tight'); plt.close()
    print(f"Saved {p}")

# ── Fig 6: Learned weights per topology ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle(f'Learned Ensemble Weights per Topology  (n={n_plot})',
             fontsize=13, fontweight='bold')
vlabels = ['EGDM','Epidemic','Finance','Biology','Social']
vcols   = [COLORS[m] for m in MODEL_ORDER[:5]]
for ax, topo in zip(axes, TOPO_ORDER):
    w = data[topo].get('M6_Ensemble',{}).get('learned_weights', np.ones(5)/5)
    bars = ax.bar(vlabels, w, color=vcols, alpha=0.88, edgecolor='white', lw=0.6)
    for bar, v in zip(bars, w):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                f'{v:.3f}', ha='center', fontsize=9)
    ax.axhline(0.2, color='gray', ls='--', alpha=0.4, lw=1, label='Equal weight')
    ax.set_title(TOPO_SHORT[topo], fontsize=12)
    ax.set_ylim(0, max(w)*1.3+0.05)
    ax.set_ylabel('wₖ' if ax==axes[0] else '')
    ax.legend(fontsize=8)
plt.tight_layout()
p = f'{FIG}/fig6_weights_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"Saved {p}")

# ── Fig 7: IEE reduction vs M1 baseline ──────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
models_vs_m1 = [m for m in MODEL_ORDER if m != 'M1_EGDM']
x2 = np.arange(len(TOPO_ORDER))
nm2 = len(models_vs_m1)
offsets2 = np.linspace(-(nm2-1)/2,(nm2-1)/2,nm2) * 0.13
for j, m in enumerate(models_vs_m1):
    reductions = []
    for topo in TOPO_ORDER:
        base = data[topo].get('M1_EGDM',{}).get('IEE',1)
        val  = data[topo].get(m,{}).get('IEE', base)
        pct  = 100*(base - val)/max(base,1e-9)
        reductions.append(pct)
    lw = 1.5 if m == 'M6_Ensemble' else 0.4
    ec = 'black' if m == 'M6_Ensemble' else 'white'
    bars = ax.bar(x2+offsets2[j], reductions, width=0.12*0.92,
                  color=COLORS[m], label=MODEL_LABELS[m],
                  alpha=0.88, edgecolor=ec, linewidth=lw)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x2)
ax.set_xticklabels([TOPO_SHORT[t] for t in TOPO_ORDER], fontsize=12)
ax.set_ylabel('IEE Reduction vs M1 Baseline (%)')
ax.set_title(f'Relative IEE Reduction vs EGDM Baseline  (n={n_plot})\nPositive = better than EGDM', fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
p = f'{FIG}/fig7_iee_reduction_n{n_plot}.png'
plt.savefig(p, bbox_inches='tight'); plt.close()
print(f"Saved {p}")

print("\n✅ All done.")
