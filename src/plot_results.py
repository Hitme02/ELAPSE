"""
plot_results.py
---------------
Generates all 8 figures for the ELAPSE paper from simulation results.

Figure 1: Entropy trajectories H(t) — all 7 models, one panel per topology
Figure 2: Mass decay M(t)           — all 7 models, one panel per topology
Figure 3: IEE bar chart             — all 7 models across topologies
Figure 4: Mortality activation time t* comparison
Figure 5: Ensemble vote evolution over time (M6 only)
Figure 6: Learned weights visualisation (bar chart per topology)
Figure 7: IEE vs network size n     — scaling study
Figure 8: Sensitivity heatmap       — H_c_frac × beta → IEE (M1 and M5)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from math_utils import max_entropy

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'figure.dpi':         150,
})

MODEL_COLORS = {
    'M0_Baseline': '#888888',
    'M1_EGDM':     '#1A56A4',
    'M2_Epidemic': '#E05A2B',
    'M3_Finance':  '#2A9D8F',
    'M4_Biology':  '#E9C46A',
    'M5_Social':   '#9B59B6',
    'M6_Ensemble': '#E63946',
}

MODEL_LABELS = {
    'M0_Baseline': 'M0: No Deletion (Baseline)',
    'M1_EGDM':     'M1: EGDM (Entropy Threshold)',
    'M2_Epidemic': 'M2: Epidemic-Entropy (SIR)',
    'M3_Finance':  'M3: Stochastic-Finance (OU)',
    'M4_Biology':  'M4: Bio-Switch (Hill)',
    'M5_Social':   'M5: Cascade-Entropy',
    'M6_Ensemble': 'M6: ELAPSE Ensemble',
}

MODEL_ORDER = ['M0_Baseline', 'M1_EGDM', 'M2_Epidemic', 'M3_Finance',
               'M4_Biology', 'M5_Social', 'M6_Ensemble']
TOPO_ORDER  = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
TOPO_SHORT  = {'Erdos-Renyi': 'ER', 'Barabasi-Albert': 'BA', 'Watts-Strogatz': 'WS'}

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')


def load_results(name='results.pkl'):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_n(results):
    """Return the largest network size available."""
    return max(results.keys())


# ── Figure 1: Entropy Trajectories ────────────────────────────────────────

def plot_entropy_trajectories(results, n=None):
    if n is None:
        n = get_n(results)
    data = results[n]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(f'Entropy Trajectories H(t)  —  n = {n}', fontsize=14, fontweight='bold', y=1.01)

    for ax, topo in zip(axes, TOPO_ORDER):
        topo_data = data[topo]
        lambda2   = topo_data.get('M1_EGDM', {}).get('lambda2', 0)

        for m in MODEL_ORDER:
            if m not in topo_data:
                continue
            r  = topo_data[m]
            lw = 2.5 if m in ('M6_Ensemble', 'M0_Baseline') else 1.5
            ls = '-' if m in ('M6_Ensemble', 'M0_Baseline') else '--' if m == 'M1_EGDM' else '-'
            ax.plot(r['t_arr'], r['H_arr'],
                    color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                    linewidth=lw, linestyle=ls, alpha=0.9)

        # Mark H_c
        H_c = 0.65 * max_entropy(n)
        ax.axhline(H_c, color='black', linestyle=':', linewidth=1.2, alpha=0.5, label=r'$H_c$')

        ax.set_title(f'{TOPO_SHORT[topo]}  ($\\lambda_2$={lambda2:.3f})', fontsize=12)
        ax.set_xlabel('Time t', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Entropy H(p(t))', fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig1_entropy_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 2: Mass Decay ───────────────────────────────────────────────────

def plot_mass_decay(results, n=None):
    if n is None:
        n = get_n(results)
    data = results[n]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(f'Total Mass M(t)  —  n = {n}', fontsize=14, fontweight='bold', y=1.01)

    for ax, topo in zip(axes, TOPO_ORDER):
        topo_data = data[topo]

        for m in MODEL_ORDER:
            if m not in topo_data:
                continue
            r  = topo_data[m]
            M0 = r['M_arr'][0]
            lw = 2.5 if m in ('M6_Ensemble', 'M0_Baseline') else 1.5
            ax.plot(r['t_arr'], r['M_arr'] / max(M0, 1e-6),
                    color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                    linewidth=lw, alpha=0.9)

        ax.axhline(0.5, color='black', linestyle=':', linewidth=1.0, alpha=0.4, label='50% mass')
        ax.set_title(f'{TOPO_SHORT[topo]}', fontsize=12)
        ax.set_xlabel('Time t', fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel('Normalised mass M(t)/M(0)', fontsize=11)
        ax.set_ylim(0, 1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig2_mass_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 3: IEE Bar Chart ────────────────────────────────────────────────

def plot_iee_comparison(results, n=None):
    if n is None: n = get_n(results)
    data = results[n]

    fig, ax = plt.subplots(figsize=(13, 5))

    x_pos    = np.arange(len(TOPO_ORDER))
    n_models = len(MODEL_ORDER)
    width    = 0.11
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, m in enumerate(MODEL_ORDER):
        iees = [data[topo].get(m, {}).get('IEE', 0) for topo in TOPO_ORDER]
        bars = ax.bar(x_pos + offsets[j], iees,
                      width=width * 0.9,
                      color=MODEL_COLORS[m],
                      label=MODEL_LABELS[m],
                      alpha=0.85,
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, iees):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=45)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TOPO_SHORT[t] for t in TOPO_ORDER], fontsize=12)
    ax.set_ylabel('Time-Integrated Entropy Exposure (IEE)', fontsize=11)
    ax.set_title(f'IEE Comparison Across Models and Topologies  (n={n})\nLower = better privacy protection',
                 fontsize=12)
    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig3_iee_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 4: Mortality Activation Time t* ────────────────────────────────

def plot_tstar_comparison(results, n=None):
    if n is None: n = get_n(results)
    data = results[n]

    fig, ax = plt.subplots(figsize=(11, 5))

    x_pos    = np.arange(len(TOPO_ORDER))
    n_models = len(MODEL_ORDER)
    width    = 0.11
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    for j, m in enumerate(MODEL_ORDER):
        tstars = [data[topo].get(m, {}).get('t_star', 0) for topo in TOPO_ORDER]
        ax.bar(x_pos + offsets[j], tstars,
               width=width * 0.9,
               color=MODEL_COLORS[m],
               label=MODEL_LABELS[m],
               alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([TOPO_SHORT[t] for t in TOPO_ORDER], fontsize=12)
    ax.set_ylabel('Mortality Activation Time t*', fontsize=11)
    ax.set_title(f'Mortality Activation Time  (n={n})\nLower = faster deletion onset', fontsize=12)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig4_tstar_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 5: Ensemble Vote Evolution ─────────────────────────────────────

def plot_vote_evolution(results, topo='Erdos-Renyi', n=None):
    if n is None: n = get_n(results)
    data = results[n][topo]

    if 'M6_Ensemble' not in data or 'votes_arr' not in data['M6_Ensemble']:
        print("  No ensemble vote data found, skipping Fig 5.")
        return None

    votes = data['M6_Ensemble']['votes_arr']   # (T, 5)
    t_arr = data['M6_Ensemble']['t_arr']
    Delta = data['M6_Ensemble']['Delta_arr']

    vote_names  = ['v₁: EGDM', 'v₂: Epidemic', 'v₃: Finance', 'v₄: Biology', 'v₅: Social']
    vote_colors = [MODEL_COLORS[m] for m in
                   ['M1_EGDM', 'M2_Epidemic', 'M3_Finance', 'M4_Biology', 'M5_Social']]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f'Ensemble Vote Evolution  —  {TOPO_SHORT[topo]}  (n={n})', fontsize=13)

    for k in range(5):
        ax1.plot(t_arr, votes[:, k], label=vote_names[k],
                 color=vote_colors[k], linewidth=1.8, alpha=0.9)

    ax1.axhline(0.4, color='black', linestyle=':', alpha=0.5, label='θ threshold')
    ax1.set_ylabel('Individual Vote vₖ(t)', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylim(0, 1.05)

    ax2.plot(t_arr, Delta, color=MODEL_COLORS['M6_Ensemble'],
             linewidth=2.5, label='Δ(t) ensemble vote')
    ax2.axhline(0.4, color='black', linestyle=':', alpha=0.5, label='θ = 0.4')
    ax2.fill_between(t_arr, 0.4, Delta,
                     where=Delta >= 0.4, alpha=0.2,
                     color=MODEL_COLORS['M6_Ensemble'], label='Deletion active')
    ax2.set_ylabel('Weighted Vote Δ(t)', fontsize=11)
    ax2.set_xlabel('Time t', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig5_votes_{TOPO_SHORT[topo]}_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 6: Learned Weights ─────────────────────────────────────────────

def plot_learned_weights(results, n=None):
    if n is None: n = get_n(results)
    data = results[n]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('Learned ELAPSE Ensemble Weights per Topology\n'
                 '(Entropy regularisation prevents weight collapse)',
                 fontsize=12, fontweight='bold')

    vote_labels = ['EGDM\n(M1)', 'Epidemic\n(M2)', 'Finance\n(M3)',
                   'Biology\n(M4)', 'Social\n(M5)']
    vote_colors = [MODEL_COLORS[m] for m in
                   ['M1_EGDM', 'M2_Epidemic', 'M3_Finance', 'M4_Biology', 'M5_Social']]

    for ax, topo in zip(axes, TOPO_ORDER):
        if 'M6_Ensemble' not in data[topo]:
            continue
        w = data[topo]['M6_Ensemble'].get('learned_weights', np.ones(5) / 5)
        bars = ax.bar(vote_labels, w, color=vote_colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, w):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=9)
        ax.set_title(TOPO_SHORT[topo], fontsize=12)
        ax.set_ylim(0, max(max(w) * 1.3, 0.5))
        if ax == axes[0]:
            ax.set_ylabel('Weight wₖ')
        # Equal weight reference line
        ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1,
                   label='Equal (0.2)')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f'fig6_weights_n{n}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 7: IEE vs Network Size (Scaling) ───────────────────────────────

def plot_scaling(results):
    """
    Mean IEE per model across the 3 topologies, plotted against network size n.
    Shows how each model's privacy performance scales with network complexity.
    """
    sizes = sorted(results.keys())
    if len(sizes) < 2:
        print("  Need at least 2 network sizes for scaling plot, skipping Fig 7.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    markers = ['o', 's', '^', 'D', 'v', 'P', '*']
    for j, m in enumerate(MODEL_ORDER):
        mean_iees = []
        for n in sizes:
            iees = []
            for topo in TOPO_ORDER:
                v = results[n].get(topo, {}).get(m, {}).get('IEE', None)
                if v is not None:
                    iees.append(v)
            mean_iees.append(np.mean(iees) if iees else np.nan)

        lw = 2.5 if m in ('M6_Ensemble', 'M0_Baseline') else 1.5
        ax.plot(sizes, mean_iees,
                color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                linewidth=lw, marker=markers[j], markersize=7, alpha=0.9)

    ax.set_xlabel('Network size n', fontsize=12)
    ax.set_ylabel('Mean IEE (across 3 topologies)', fontsize=11)
    ax.set_title('IEE Scaling with Network Size\nM0 (no deletion) upper bound shows value of ELAPSE',
                 fontsize=12)
    ax.set_xticks(sizes)
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig7_scaling.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Figure 8: Sensitivity Heatmap ─────────────────────────────────────────

def plot_sensitivity(sensitivity_path=None):
    """
    2-panel heatmap: H_c_frac × beta → IEE for M1 and M5.
    """
    if sensitivity_path is None:
        sensitivity_path = os.path.join(OUTPUT_DIR, 'sensitivity.pkl')

    if not os.path.exists(sensitivity_path):
        print("  No sensitivity results found, skipping Fig 8.")
        return None

    with open(sensitivity_path, 'rb') as f:
        sens = pickle.load(f)

    H_c_fracs = sens['H_c_fracs']
    betas     = sens['betas']
    M1_IEE    = sens['M1_IEE']
    M5_IEE    = sens['M5_IEE']
    n         = sens['n']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Sensitivity Analysis: IEE vs H_c and β  (n={n}, ER topology)\n'
                 f'Lower IEE = better privacy. Darker = worse.',
                 fontsize=12, fontweight='bold')

    vmin = min(M1_IEE.min(), M5_IEE.min())
    vmax = max(M1_IEE.max(), M5_IEE.max())

    for ax, mat, title in zip(axes, [M1_IEE, M5_IEE],
                               ['M1: EGDM — H_c × β sensitivity',
                                'M5: Social — H_c × β sensitivity']):
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r',
                       vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([str(b) for b in betas])
        ax.set_yticks(range(len(H_c_fracs)))
        ax.set_yticklabels([f'{h:.2f}' for h in H_c_fracs])
        ax.set_xlabel('β (EGDM sharpness)', fontsize=11)
        ax.set_ylabel('H_c / H_max', fontsize=11)
        ax.set_title(title, fontsize=11)

        # Annotate cells
        for i in range(len(H_c_fracs)):
            for j in range(len(betas)):
                ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center',
                        fontsize=8, color='black')

    plt.colorbar(im, ax=axes, label='IEE', shrink=0.85)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'fig8_sensitivity.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ── Run all figures ────────────────────────────────────────────────────────

def generate_all_figures(results_path=None):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_results('results.pkl')

    print(f"\nGenerating figures for all network sizes...")
    paths = []

    for n in sorted(results.keys()):
        print(f"  --> Processing n={n}...")
        paths.append(plot_entropy_trajectories(results, n))
        paths.append(plot_mass_decay(results, n))
        paths.append(plot_iee_comparison(results, n))
        paths.append(plot_tstar_comparison(results, n))
        # Generate vote evolution specific to ER since it shows the cleanest diffusion dynamics
        paths.append(plot_vote_evolution(results, topo='Erdos-Renyi', n=n))
        paths.append(plot_learned_weights(results, n=n))

    paths.append(plot_scaling(results))
    paths.append(plot_sensitivity())

    print(f"\nAll figures saved to {FIGURES_DIR}")
    return [p for p in paths if p is not None]


if __name__ == '__main__':
    generate_all_figures()
