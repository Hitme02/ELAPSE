"""
generate_tables.py
------------------
Generates LaTeX table cells from ci_results.pkl and ci_snap.pkl.

Outputs:
  paper/tables/t{n}_{topo}_{model}.tex   -- single cell "value ± CI"
  paper/tables/iee_full_table.tex        -- full formatted table
"""

import pickle, os, sys, numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(BASE_DIR, 'output')
TAB_DIR  = os.path.join(BASE_DIR, 'paper', 'tables')
os.makedirs(TAB_DIR, exist_ok=True)

MODEL_MAP = {
    'M0_Baseline': 'M0',
    'M1_EGDM':     'M1',
    'M2_Epidemic': 'M2',
    'M3_Finance':  'M3',
    'M4_Biology':  'M4',
    'M5_Social':   'M5',
    'M6_Ensemble': 'M6',
}
TOPO_SHORT = {
    'Erdos-Renyi':     'ER',
    'Barabasi-Albert': 'BA',
    'Watts-Strogatz':  'WS',
}
MODEL_ORDER = list(MODEL_MAP.keys())
TOPO_ORDER  = list(TOPO_SHORT.keys())


def fmt(mean, ci):
    """Format 'mean ± half-CI' for LaTeX."""
    half = (ci[1] - ci[0]) / 2
    return f'${mean:.1f} \\pm {half:.1f}$'


def write_cell(n, topo_short, model_short, text):
    fname = os.path.join(TAB_DIR, f't{n}_{topo_short}_{model_short}.tex')
    with open(fname, 'w') as f:
        f.write(text)


def generate_main_tables(ci_results):
    """Generate per-n full IEE tables."""
    for n in sorted(ci_results.keys()):
        data = ci_results[n]

        # Single-cell files for paper \input{} commands
        for topo, ts in TOPO_SHORT.items():
            for mfull, mshort in MODEL_MAP.items():
                r = data.get(topo, {}).get(mfull, {})
                if r:
                    text = fmt(r['iee_mean'], r['iee_ci'])
                    write_cell(n, ts, mshort, text)

        # Full table file
        lines = []
        lines.append(r'\begin{tabular}{lccc}')
        lines.append(r'  \toprule')
        lines.append(r'  Mechanism & ER & BA & WS \\')
        lines.append(r'  \midrule')
        for mfull, mshort in MODEL_MAP.items():
            label = {
                'M0_Baseline': 'M0 (No Deletion)',
                'M1_EGDM':     'M1 (EGDM)',
                'M2_Epidemic': 'M2 (Epidemic-SIR)',
                'M3_Finance':  'M3 (Finance-OU)',
                'M4_Biology':  'M4 (Bio-Hill)',
                'M5_Social':   'M5 (Cascade)',
                'M6_Ensemble': '\\textbf{M6 (ELAPSE)}',
            }[mfull]

            cells = []
            for topo in TOPO_ORDER:
                r = data.get(topo, {}).get(mfull, {})
                if r:
                    cells.append(fmt(r['iee_mean'], r['iee_ci']))
                else:
                    cells.append('---')

            if mfull == 'M6_Ensemble':
                row = f'  \\midrule\n  {label} & ' + ' & '.join(f'\\textbf{{{c}}}' for c in cells) + r' \\'
            else:
                row = f'  {label} & ' + ' & '.join(cells) + r' \\'
            lines.append(row)

        lines.append(r'  \bottomrule')
        lines.append(r'\end{tabular}')

        table_text = '\n'.join(lines)
        fname = os.path.join(TAB_DIR, f'iee_table_n{n}.tex')
        with open(fname, 'w') as f:
            f.write(table_text)
        print(f'  Written: {fname}')


def generate_snap_table(ci_snap):
    """Generate SNAP validation table."""
    if not ci_snap:
        return

    lines = []
    lines.append(r'\begin{tabular}{lccc}')
    lines.append(r'  \toprule')
    lines.append(r'  Mechanism & Gnutella08 & Gnutella31 \\')
    lines.append(r'  \midrule')

    snap_keys = list(ci_snap.keys())

    for mfull, mshort in MODEL_MAP.items():
        label = mfull.replace('_', ' ')
        cells = []
        for sname in snap_keys:
            r = ci_snap[sname].get(mfull, {})
            if r:
                cells.append(fmt(r['iee_mean'], r['iee_ci']))
            else:
                cells.append('---')
        # Pad if fewer than 2 SNAP datasets
        while len(cells) < 2:
            cells.append('---')

        row = f'  {label} & ' + ' & '.join(cells) + r' \\'
        lines.append(row)

    lines.append(r'  \bottomrule')
    lines.append(r'\end{tabular}')
    fname = os.path.join(TAB_DIR, 'snap_table.tex')
    with open(fname, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Written: {fname}')


def generate_summary_csv(ci_results):
    """Generate CSV for external use."""
    import csv
    fname = os.path.join(OUT_DIR, 'iee_summary.csv')
    with open(fname, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['n', 'topology', 'mechanism',
                         'iee_mean', 'iee_ci_lo', 'iee_ci_hi', 'iee_std',
                         'tstar_mean', 'lambda2'])
        for n in sorted(ci_results.keys()):
            for topo, ts in TOPO_SHORT.items():
                for mfull in MODEL_ORDER:
                    r = ci_results[n].get(topo, {}).get(mfull, {})
                    if r:
                        writer.writerow([
                            n, ts, MODEL_MAP[mfull],
                            f'{r["iee_mean"]:.4f}',
                            f'{r["iee_ci"][0]:.4f}',
                            f'{r["iee_ci"][1]:.4f}',
                            f'{r["iee_std"]:.4f}',
                            f'{r["tstar_mean"]:.4f}',
                            f'{r.get("lambda2", 0):.4f}',
                        ])
    print(f'  CSV: {fname}')


def main():
    def load(name):
        p = os.path.join(OUT_DIR, name)
        if not os.path.exists(p):
            print(f'  Not found: {p}')
            return None
        with open(p, 'rb') as f:
            return pickle.load(f)

    ci_results = load('ci_results.pkl')
    ci_snap    = load('ci_snap.pkl')

    if ci_results:
        print('Generating main IEE tables...')
        generate_main_tables(ci_results)
        generate_summary_csv(ci_results)

    if ci_snap:
        print('Generating SNAP table...')
        generate_snap_table(ci_snap)

    print('Done.')


if __name__ == '__main__':
    main()
