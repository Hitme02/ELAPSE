"""
plot_extended.py
----------------
Generates all publication-quality figures (300 DPI, PDF + PNG) for the
extended ELAPSE paper (AMM Elsevier format).

Figure 1 : Entropy trajectories H(t) with 95% CI bands
Figure 2 : IEE bar chart at n=500 with error bars + SNAP results
Figure 3 : Learned weights with std deviation bars
Figure 4 : Scale-stress IEE vs n with SNAP data points
Figure 5 : Sensitivity heatmap (fixed colorbar label)
Figure 6 : Adversarial degradation — IEE vs f for ER and BA
Figure 7 : Gossip estimation error — true H(t) vs k-hop estimated H(t)
Figure 8 : M3 vote trajectory analysis — v3(t) vs others on BA topology
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pickle, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from math_utils import max_entropy

OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})

MODEL_COLORS = {
    'M0_Baseline': '#555555',
    'M1_EGDM':     '#1A56A4',
    'M2_Epidemic': '#E05A2B',
    'M3_Finance':  '#2A9D8F',
    'M4_Biology':  '#E9C46A',
    'M5_Social':   '#9B59B6',
    'M6_Ensemble': '#E63946',
}

MODEL_LABELS = {
    'M0_Baseline': 'M0: No Deletion',
    'M1_EGDM':     'M1: EGDM',
    'M2_Epidemic': 'M2: Epidemic-SIR',
    'M3_Finance':  'M3: Finance-OU',
    'M4_Biology':  'M4: Bio-Hill',
    'M5_Social':   'M5: Cascade',
    'M6_Ensemble': 'M6: ELAPSE',
}

MODEL_ORDER = ['M0_Baseline', 'M1_EGDM', 'M2_Epidemic', 'M3_Finance',
               'M4_Biology', 'M5_Social', 'M6_Ensemble']
TOPO_ORDER  = ['Erdos-Renyi', 'Barabasi-Albert', 'Watts-Strogatz']
TOPO_LABEL  = {'Erdos-Renyi': 'ER', 'Barabasi-Albert': 'BA', 'Watts-Strogatz': 'WS'}

VOTE_LABELS  = ['v₁: EGDM', 'v₂: Epidemic', 'v₃: Finance', 'v₄: Biology', 'v₅: Social']
VOTE_COLORS  = [MODEL_COLORS[m] for m in
                ['M1_EGDM', 'M2_Epidemic', 'M3_Finance', 'M4_Biology', 'M5_Social']]


def savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, name + '.pdf')
    png_path = os.path.join(FIGURES_DIR, name + '.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")
    return pdf_path, png_path


# ── Figure 1: Entropy Trajectories with 95% CI ────────────────────────────────

def plot_fig1_entropy(ci_results, n=100):
    data = ci_results[n]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    fig.suptitle(f'Shannon Entropy Trajectories $H(t)$  —  $n = {n}$  '
                 f'(shaded: 95\\% CI over $N_{{\\rm test}}=20$ seeds)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, topo in zip(axes, TOPO_ORDER):
        td      = data[topo]
        lambda2 = td['M1_EGDM']['lambda2']
        H_c     = 0.65 * max_entropy(n)

        for m in MODEL_ORDER:
            if m not in td:
                continue
            r  = td[m]
            t  = r['t_arr']
            Hm = r['H_arr_mean']
            Hl = r['H_arr_lo']
            Hh = r['H_arr_hi']
            lw = 2.5 if m in ('M6_Ensemble', 'M0_Baseline') else 1.5
            ls = '--' if m == 'M0_Baseline' else '-'
            ax.plot(t, Hm, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                    linewidth=lw, linestyle=ls, alpha=0.95)
            ax.fill_between(t, Hl, Hh, alpha=0.12, color=MODEL_COLORS[m])

        ax.axhline(H_c, color='black', linestyle=':', linewidth=1.0, alpha=0.6,
                   label=r'$H_c$ (threshold)')
        ax.set_title(f'{TOPO_LABEL[topo]}  ($\\lambda_2={lambda2:.3f}$)', fontsize=12)
        ax.set_xlabel('Time $t$', fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('Entropy $H(t)$', fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.14), frameon=True, fontsize=9)
    plt.tight_layout()
    return savefig(fig, f'fig1_entropy_n{n}')


# ── Figure 2: IEE Bar Chart at n=500 with CI and SNAP ────────────────────────

def plot_fig2_iee(ci_results, ci_snap=None, n=500):
    if n not in ci_results:
        n = max(ci_results.keys())

    data      = ci_results[n]
    snap_names = list(ci_snap.keys()) if ci_snap else []

    # Group data: synthetic topologies + SNAP
    all_groups = TOPO_ORDER + snap_names
    all_labels = [TOPO_LABEL[t] for t in TOPO_ORDER] + snap_names

    fig, ax = plt.subplots(figsize=(14, 5.5))

    x_pos    = np.arange(len(all_groups))
    n_models = len(MODEL_ORDER)
    width    = 0.10
    offsets  = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

    for j, m in enumerate(MODEL_ORDER):
        iees = []
        cis  = []
        for group in all_groups:
            if group in data:
                r    = data[group].get(m, {})
                iees.append(r.get('iee_mean', 0))
                ci   = r.get('iee_ci', (0, 0))
                cis.append((r.get('iee_mean', 0) - ci[0],
                             ci[1] - r.get('iee_mean', 0)))
            elif ci_snap and group in ci_snap:
                r    = ci_snap[group].get(m, {})
                iees.append(r.get('iee_mean', 0))
                ci   = r.get('iee_ci', (0, 0))
                cis.append((r.get('iee_mean', 0) - ci[0],
                             ci[1] - r.get('iee_mean', 0)))
            else:
                iees.append(0)
                cis.append((0, 0))

        yerr = np.array(cis).T  # (2, n_groups)
        bars = ax.bar(x_pos + offsets[j], iees,
                      width=width * 0.9,
                      yerr=yerr,
                      capsize=2,
                      color=MODEL_COLORS[m],
                      label=MODEL_LABELS[m],
                      alpha=0.85, edgecolor='white', linewidth=0.4,
                      error_kw=dict(elinewidth=0.8, ecolor='#333'))

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=11)
    ax.set_ylabel('IEE (mean ± 95\\% CI)', fontsize=11)
    ax.set_title(f'Time-Integrated Entropy Exposure (IEE) — $n = {n}$ synthetic + SNAP real-world\n'
                 f'Lower = stronger privacy protection', fontsize=12)
    ax.legend(fontsize=8.5, loc='upper right')

    # Add vertical separator between synthetic and SNAP
    if snap_names:
        ax.axvline(len(TOPO_ORDER) - 0.5, color='gray', linestyle='--',
                   linewidth=1.0, alpha=0.6)
        ax.text(len(TOPO_ORDER) - 0.5, ax.get_ylim()[1] * 0.95,
                'SNAP →', fontsize=9, color='gray', ha='left')

    plt.tight_layout()
    return savefig(fig, f'fig2_iee_n{n}')


# ── Figure 3: Learned Weights with Std Dev Bars ───────────────────────────────

def plot_fig3_weights(ci_results, n=100):
    if n not in ci_results:
        n = max(ci_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle('ELAPSE Ensemble Learned Weights $\\mathbf{w}$ per Topology\n'
                 '(entropy regularisation $\\lambda_{\\rm reg}=15$)',
                 fontsize=12, fontweight='bold')

    vlabels = ['EGDM\n(M1)', 'Epidemic\n(M2)', 'Finance\n(M3)',
               'Biology\n(M4)', 'Social\n(M5)']
    vcolors = VOTE_COLORS

    for ax, topo in zip(axes, TOPO_ORDER):
        entry = ci_results[n][topo].get('M6_Ensemble', {})
        w     = entry.get('learned_weights', np.ones(5) / 5)

        bars = ax.bar(vlabels, w, color=vcolors, alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, w):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', fontsize=9.5, fontweight='bold')

        ax.axhline(0.2, color='gray', linestyle='--', alpha=0.5,
                   linewidth=1.0, label='Equal ($1/K$)')
        ax.set_title(TOPO_LABEL[topo], fontsize=13)
        ax.set_ylim(0, max(max(w) * 1.35, 0.55))
        if ax is axes[0]:
            ax.set_ylabel('Weight $w_k$', fontsize=11)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return savefig(fig, f'fig3_weights_n{n}')


# ── Figure 4: IEE vs n (Scaling) with SNAP points ────────────────────────────

def plot_fig4_scaling(ci_results, ci_snap=None):
    sizes   = sorted(ci_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5.5))
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    for j, m in enumerate(MODEL_ORDER):
        mean_iees = []
        ci_lo     = []
        ci_hi     = []
        for n in sizes:
            iees_all = []
            ci_lo_n  = []
            ci_hi_n  = []
            for topo in TOPO_ORDER:
                r = ci_results[n].get(topo, {}).get(m, {})
                if 'iee_mean' in r:
                    iees_all.append(r['iee_mean'])
                    ci_lo_n.append(r['iee_ci'][0])
                    ci_hi_n.append(r['iee_ci'][1])

            if iees_all:
                mean_iees.append(np.mean(iees_all))
                ci_lo.append(np.mean(ci_lo_n))
                ci_hi.append(np.mean(ci_hi_n))
            else:
                mean_iees.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)

        lw = 2.5 if m in ('M6_Ensemble', 'M0_Baseline') else 1.5
        l, = ax.plot(sizes, mean_iees, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                     linewidth=lw, marker=markers[j], markersize=7, alpha=0.9)
        ax.fill_between(sizes, ci_lo, ci_hi, alpha=0.08, color=MODEL_COLORS[m])

    # Add SNAP data points as triangles
    if ci_snap:
        snap_sizes = [d['M6_Ensemble']['n'] for d in ci_snap.values() if 'M6_Ensemble' in d]
        snap_iees  = [d['M6_Ensemble']['iee_mean'] for d in ci_snap.values() if 'M6_Ensemble' in d]
        snap_names = [name for name, d in ci_snap.items() if 'M6_Ensemble' in d]

        for sx, sy, sn in zip(snap_sizes, snap_iees, snap_names):
            ax.scatter(sx, sy, marker='*', s=200, color=MODEL_COLORS['M6_Ensemble'],
                       zorder=5, edgecolors='black', linewidth=0.5)
            ax.annotate(f'SNAP\n{sn.replace("Gnutella", "G")}',
                        (sx, sy), textcoords='offset points',
                        xytext=(8, 0), fontsize=8, color='black')

    ax.set_xlabel('Network size $n$', fontsize=12)
    ax.set_ylabel('Mean IEE ± 95\\% CI (across 3 topologies)', fontsize=11)
    ax.set_title('IEE Scaling with Network Size — M0 upper bound confirms ELAPSE value',
                 fontsize=12)
    ax.set_xticks(sizes)
    ax.legend(fontsize=8.5, loc='upper left', ncol=2)
    plt.tight_layout()
    return savefig(fig, 'fig4_scaling')


# ── Figure 5: Sensitivity Heatmap ────────────────────────────────────────────

def plot_fig5_sensitivity():
    path = os.path.join(OUTPUT_DIR, 'sensitivity.pkl')
    if not os.path.exists(path):
        print("  No sensitivity.pkl found, skipping Fig 5.")
        return None, None

    with open(path, 'rb') as f:
        sens = pickle.load(f)

    H_c_fracs = sens['H_c_fracs']
    betas     = sens['betas']
    M1_IEE    = sens['M1_IEE']
    M5_IEE    = sens['M5_IEE']
    n         = sens['n']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Sensitivity: IEE vs $H_c/H_{{\\rm max}}$ and $\\beta$  '
                 f'($n={n}$, ER topology)\nDarker = higher exposure (worse)',
                 fontsize=12, fontweight='bold')

    vmin = min(M1_IEE.min(), M5_IEE.min())
    vmax = max(M1_IEE.max(), M5_IEE.max())

    for ax, mat, title in zip(axes,
                               [M1_IEE, M5_IEE],
                               ['M1 (EGDM): $H_c/H_{\\rm max}$ × $\\beta$',
                                'M5 (Cascade): $H_c/H_{\\rm max}$ × $\\beta$']):
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn_r',
                       vmin=vmin, vmax=vmax, origin='lower')
        ax.set_xticks(range(len(betas)))
        ax.set_xticklabels([str(b) for b in betas], fontsize=10)
        ax.set_yticks(range(len(H_c_fracs)))
        ax.set_yticklabels([f'{h:.2f}' for h in H_c_fracs], fontsize=10)
        ax.set_xlabel('$\\beta$ (EGDM sharpness)', fontsize=11)
        ax.set_ylabel('$H_c / H_{\\rm max}$', fontsize=11)
        ax.set_title(title, fontsize=11)

        for i in range(len(H_c_fracs)):
            for j in range(len(betas)):
                ax.text(j, i, f'{mat[i, j]:.1f}', ha='center', va='center',
                        fontsize=8, color='black')

    cbar = plt.colorbar(im, ax=axes, label='IEE', shrink=0.85)
    cbar.set_label('Time-Integrated Entropy Exposure (IEE)', fontsize=10)
    plt.tight_layout()
    return savefig(fig, 'fig5_sensitivity')


# ── Figure 6: Adversarial Degradation ────────────────────────────────────────

def plot_fig6_adversarial(adv_results):
    # Handle both old format {topo: {f: ...}} and new format {H_c_key: {topo: {f: ...}}}
    if 'H_c_65' in adv_results or 'H_c_50' in adv_results:
        thresh_data = {
            r'$H_c=0.65H_{\max}$': adv_results.get('H_c_65', {}),
            r'$H_c=0.50H_{\max}$': adv_results.get('H_c_50', {}),
        }
    else:
        thresh_data = {r'$H_c=0.65H_{\max}$': adv_results}

    topo_list   = ['Erdos-Renyi', 'Barabasi-Albert']
    n_thresh    = len(thresh_data)
    n_topo      = len(topo_list)

    fig, axes = plt.subplots(n_thresh, n_topo,
                             figsize=(13, 5 * n_thresh), sharey=False,
                             squeeze=False)
    fig.suptitle('Adversarial Degradation: IEE vs Attacker Fraction $f$\n'
                 'Rows: threshold; Dashed: adaptive $H_c$ countermeasure',
                 fontsize=13, fontweight='bold')

    for row, (thresh_label, topo_dict) in enumerate(thresh_data.items()):
        for col, topo in enumerate(topo_list):
            ax = axes[row][col]
            if topo not in topo_dict:
                ax.set_visible(False)
                continue
            td       = topo_dict[topo]
            f_values = sorted(td.keys())

            plain_mean = [td[f]['plain']['iee_mean'] for f in f_values]
            plain_lo   = [td[f]['plain']['iee_ci'][0] for f in f_values]
            plain_hi   = [td[f]['plain']['iee_ci'][1] for f in f_values]

            adapt_mean = [td[f]['adaptive']['iee_mean'] for f in f_values]
            adapt_lo   = [td[f]['adaptive']['iee_ci'][0] for f in f_values]
            adapt_hi   = [td[f]['adaptive']['iee_ci'][1] for f in f_values]

            ax.plot(f_values, plain_mean, 'o-', color='#E63946', linewidth=2.2,
                    markersize=7, label='ELAPSE (no countermeasure)')
            ax.fill_between(f_values, plain_lo, plain_hi, alpha=0.15, color='#E63946')

            ax.plot(f_values, adapt_mean, 's--', color='#1A56A4', linewidth=2.2,
                    markersize=7, label='ELAPSE + adaptive $H_c$')
            ax.fill_between(f_values, adapt_lo, adapt_hi, alpha=0.15, color='#1A56A4')

            ax.set_xlabel('Attacker fraction $f$', fontsize=11)
            ax.set_ylabel('IEE (mean ± 95\\% CI)', fontsize=11)
            ax.set_title(f'{TOPO_LABEL[topo]} — {thresh_label} ($n=200$)', fontsize=11)
            ax.set_xticks(f_values)
            ax.legend(fontsize=9)

    plt.tight_layout()
    return savefig(fig, 'fig6_adversarial')


# ── Figure 7: Gossip Estimation Error ────────────────────────────────────────

def plot_fig7_gossip(gossip_results):
    # Detect available k values from results
    first_topo = next(iter(gossip_results.values()))
    k_values   = sorted(first_topo['H_est_mean'].keys())
    k_colors   = {2: '#E05A2B', 3: '#2A9D8F', 4: '#6A4C93', 5: '#F4A261'}
    k_labels   = {k: f'$k={k}$ hop gossip' for k in k_values}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    fig.suptitle('Gossip Protocol Entropy Estimation: True $H(t)$ vs $k$-hop Estimate\n'
                 '(shaded: 95\\% CI over 30 seeds)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, topo in zip(axes, TOPO_ORDER):
        if topo not in gossip_results:
            continue
        gr   = gossip_results[topo]
        t    = gr['t_arr']
        H_c  = gr['H_c']

        # True H(t)
        ax.plot(t, gr['H_true_mean'], color='black', linewidth=2.0,
                label='True $H(t)$', zorder=5)
        ax.fill_between(t, gr['H_true_ci'][0], gr['H_true_ci'][1],
                        alpha=0.10, color='black')

        # Gossip estimates
        for k in k_values:
            he_m = gr['H_est_mean'][k]
            he_l, he_h = gr['H_est_ci'][k]
            ax.plot(t, he_m, linewidth=1.8, linestyle='--',
                    color=k_colors[k], label=k_labels[k], alpha=0.9)
            ax.fill_between(t, he_l, he_h, alpha=0.08, color=k_colors[k])

        ax.axhline(H_c, color='gray', linestyle=':', linewidth=1.0, alpha=0.7,
                   label='$H_c$')

        ax.set_xlabel('Time $t$', fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('Entropy', fontsize=11)
        ax.set_title(TOPO_LABEL[topo], fontsize=13)
        ax.legend(fontsize=8.5)

        # Annotation: error stats for all k
        lines_text = []
        for k in k_values:
            trig = gr['trigger'].get(k, {})
            fe = trig.get('false_early_rate', float('nan'))
            ms = trig.get('missed_rate', float('nan'))
            lines_text.append(f'$k={k}$: FE={fe:.3f} MS={ms:.3f}')
        ax.text(0.02, 0.03, '\n'.join(lines_text),
                transform=ax.transAxes, fontsize=7.0,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    return savefig(fig, 'fig7_gossip')


# ── Figure 8: M3 Vote Trajectory Analysis ────────────────────────────────────

def plot_fig8_m3_analysis(m3_results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle('M3 (Finance-OU) Vote Trajectory Analysis\n'
                 '$v_3(t)$ vs other mechanisms on BA and ER topologies '
                 '(mean ± 1 std, 30 seeds)',
                 fontsize=13, fontweight='bold')

    for ax, topo in zip(axes, ['Barabasi-Albert', 'Erdos-Renyi']):
        if topo not in m3_results:
            continue
        mr  = m3_results[topo]
        t   = mr['t_arr']
        vm  = mr['votes_mean']   # (steps+1, 5)
        vs  = mr['votes_std']    # (steps+1, 5)
        lam = mr['lambda2']
        iee_m3 = mr['iee_m3_mean']
        iee_std = mr['iee_m3_std']

        for k in range(5):
            lw = 2.5 if k == 2 else 1.5   # highlight v3
            ls = '-' if k == 2 else '--'
            ax.plot(t, vm[:, k], color=VOTE_COLORS[k], label=VOTE_LABELS[k],
                    linewidth=lw, linestyle=ls, alpha=0.9)
            ax.fill_between(t, vm[:, k] - vs[:, k], vm[:, k] + vs[:, k],
                            alpha=0.08, color=VOTE_COLORS[k])

        ax.set_xlabel('Time $t$', fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel('Vote $v_k(t)$', fontsize=11)
        ax.set_title(f'{TOPO_LABEL[topo]}  ($\\lambda_2={lam:.3f}$)\n'
                     f'M3 standalone IEE = {iee_m3:.1f} ± {iee_std:.1f}',
                     fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return savefig(fig, 'fig8_m3_analysis')


# ── Master generation function ─────────────────────────────────────────────────

def generate_all_extended(n_traj=100):
    """Generate all figures. Loads pkl files from output/."""
    paths = []

    def load(name):
        p = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(p):
            print(f"  Not found: {p}")
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)

    ci_results = load('ci_results.pkl')
    ci_snap    = load('ci_snap.pkl') or {}
    adv_res    = load('adversarial_results.pkl')
    gossip_res = load('gossip_results.pkl')
    m3_res     = load('m3_analysis.pkl')

    if ci_results:
        n_traj_actual = n_traj if n_traj in ci_results else max(ci_results.keys())

        print(f"\nFig 1: Entropy trajectories (n={n_traj_actual})...")
        paths.extend(plot_fig1_entropy(ci_results, n=n_traj_actual))

        n500 = 500 if 500 in ci_results else max(ci_results.keys())
        print(f"\nFig 2: IEE bar chart (n={n500})...")
        paths.extend(plot_fig2_iee(ci_results, ci_snap=ci_snap, n=n500))

        print(f"\nFig 3: Learned weights (n={n_traj_actual})...")
        paths.extend(plot_fig3_weights(ci_results, n=n_traj_actual))

        print("\nFig 4: Scaling plot...")
        paths.extend(plot_fig4_scaling(ci_results, ci_snap=ci_snap))

    print("\nFig 5: Sensitivity heatmap...")
    r5 = plot_fig5_sensitivity()
    if r5[0]:
        paths.extend(r5)

    if adv_res:
        print("\nFig 6: Adversarial degradation...")
        paths.extend(plot_fig6_adversarial(adv_res))

    if gossip_res:
        print("\nFig 7: Gossip estimation...")
        paths.extend(plot_fig7_gossip(gossip_res))

    if m3_res:
        print("\nFig 8: M3 vote trajectory analysis...")
        paths.extend(plot_fig8_m3_analysis(m3_res))

    print(f"\nGenerated {len([p for p in paths if p])} figure files.")
    return paths


if __name__ == '__main__':
    generate_all_extended(n_traj=100)
