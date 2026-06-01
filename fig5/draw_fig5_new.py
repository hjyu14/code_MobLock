import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, 'data')
CSV_PATH    = os.path.join(DATA_DIR, 'realtime_tracking_n1024_lambda20.csv')
TARGET_DIST = 'random'       # 'random' or 'localized'
SAVE_DIR    = SCRIPT_DIR

METRICS = [
    ('isolated_ratio',    'Isolated Target Ratio'),
    ('cost_benefit',      'Cost-Benefit Ratio'),
    ('distances',         'Distance to Nearest Target'),
    ('local_betweenness', 'Local Betweenness Centrality'),
]

NETWORKS = [
    ('WS', 'Homogeneous Networks'),
    ('BA', 'Heterogeneous Networks'),
]

METHOD_ORDER = ['Adaptive TD', 'Adaptive Katz', 'Adaptive TIA', 'GNN-RL']

COLORS = {
    'Adaptive TD':   '#0072B2',
    'Adaptive Katz': '#D55E00',
    'Adaptive TIA':  '#009E73',
    'GNN-RL':        '#CC79A7',
}
MARKERS = {
    'Adaptive TD':   'o',
    'Adaptive Katz': 's',
    'Adaptive TIA':  '^',
    'GNN-RL':        'D',
}

METHOD_DISPLAY = {
    'Adaptive TD': 'Adaptive TD',
    'Adaptive Katz': 'Adaptive T-Katz',
    'Adaptive TIA': 'Adaptive TIA',
    'GNN-RL': 'MobileIsolator',
}

PANEL_LABELS = list('abcdefgh')

INSET_BBOX = [0.32, 0.50, 0.40, 0.40]

T_FINAL_LO, T_FINAL_HI = 0.9, 1.0

C_BORDER = '#5DADE2'

MAIN_YLIM = {
    'isolated_ratio':    (0.0, 1.05),
    'cost_benefit':      None,
    'distances':         None,
    'local_betweenness': (0,0.01),
}

INSET_YLIM = {
    'isolated_ratio':    None,
    'cost_benefit':      None,
    'distances':         (0.9, None),
    'local_betweenness': None,
}

NO_INSET_METRICS = {'isolated_ratio'}


mpl.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica Neue', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'font.size':         7,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   7,
    'axes.linewidth':    0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,
    'xtick.major.size':  2.5,
    'ytick.major.size':  2.5,
    'xtick.minor.size':  1.5,
    'ytick.minor.size':  1.5,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
})


def add_final_inset(ax, fig, sub, metric, t_lo=T_FINAL_LO, t_hi=T_FINAL_HI,
                    inset_ylim=None):
    bar_vals, bar_cols = [], []
    for method in METHOD_ORDER:
        mdf = sub[
            (sub['method'] == method) &
            (sub['time_bin'] >= t_lo) &
            (sub['time_bin'] <= t_hi)
        ]
        val = mdf[f'{metric}_mean'].mean() if not mdf.empty else 0.0
        bar_vals.append(val)
        bar_cols.append(COLORS[method])

    ax.axvspan(t_lo, t_hi, color='#DDDDDD', alpha=0.40, linewidth=0, zorder=0)
    for x in (t_lo, t_hi):
        ax.axvline(x, color='#888888', linewidth=0.55,
                   linestyle=(0, (3, 2)), zorder=1)

    ax_ins = ax.inset_axes(INSET_BBOX)
    x_pos  = np.arange(len(METHOD_ORDER))
    ax_ins.bar(x_pos, bar_vals, color=bar_cols,
               width=0.6, edgecolor='white', linewidth=0.3, zorder=3)

    ax_ins.set_xticks(x_pos)
    ax_ins.set_xticklabels(
        [m.replace('Adaptive ', '') for m in METHOD_ORDER],
        fontsize=4.5, rotation=30, ha='right')
    ax_ins.tick_params(which='both', direction='in',
                       labelsize=4.5, pad=1, length=1.5, width=0.4)
    ax_ins.set_ylabel('Final mean', fontsize=4.5, labelpad=2)
    for sp in ax_ins.spines.values():
        sp.set_linewidth(0.4)
    ax_ins.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)

    if inset_ylim is not None:
        y_lo_ins, y_hi_ins = inset_ylim
        if y_lo_ins is None:
            y_lo_ins = min(0, min(bar_vals) * 0.9) if bar_vals else 0
        kw = {'bottom': y_lo_ins}
        if y_hi_ins is not None:
            kw['top'] = y_hi_ins
        ax_ins.set_ylim(**kw)
    else:
        y_lo_ins = min(0, min(bar_vals) * 0.9) if bar_vals else 0
        ax_ins.set_ylim(bottom=y_lo_ins)

    y_max = ax.get_ylim()[1]
    for (xA, xB_frac) in [(t_lo, 0), (t_hi, 1)]:
        cp = ConnectionPatch(
            xyA=(xA, y_max),   coordsA=ax.transData,
            xyB=(xB_frac, 1),  coordsB=ax_ins.transAxes,
            arrowstyle='-', linestyle='dashed',
            color='#999999', linewidth=0.55, zorder=10,
        )
        fig.add_artist(cp)

    return ax_ins


def main():
    print(f"Loading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"Target distribution : {TARGET_DIST}")

    FIG_W = 7.087
    PAN_H = 1.65
    LEG_H = 0.26

    fig = plt.figure(figsize=(FIG_W, 2 * PAN_H + LEG_H + 0.12))

    gs = fig.add_gridspec(
        3, 4,
        height_ratios=[PAN_H, PAN_H, LEG_H],
        hspace=0.52,
        wspace=0.40,
        left=0.09, right=0.99,
        top=0.92,  bottom=0.06,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_axis_off()

    panel_idx = 0

    for row in range(2):
        for col in range(4):
            net_idx    = col // 2
            metric_idx = (col % 2) + row * 2
            net_type, net_label = NETWORKS[net_idx]
            metric_key, metric_title = METRICS[metric_idx]

            config_key = f"{net_type}-{TARGET_DIST}"
            sub = df[df['config'] == config_key]
            if sub.empty:
                print(f"  Warning: no data for {config_key}")

            ax    = axes[row][col]
            label = PANEL_LABELS[panel_idx]
            panel_idx += 1

            for method in METHOD_ORDER:
                mdf = sub[sub['method'] == method].sort_values('time_bin')
                if mdf.empty:
                    continue
                t    = mdf['time_bin'].values
                mean = mdf[f'{metric_key}_mean'].values
                sem  = mdf[f'{metric_key}_sem'].values
                ax.plot(t, mean,
                        color=COLORS[method], marker=MARKERS[method],
                        markevery=max(1, len(t) // 8),
                        markersize=2.5, linewidth=0.8,
                        markerfacecolor=COLORS[method],
                        markeredgecolor='white', markeredgewidth=0.3,
                        zorder=4)
                ax.fill_between(t, mean - sem, mean + sem,
                                color=COLORS[method], alpha=0.15,
                                linewidth=0, zorder=3)

            ax.set_xlabel('Normalised Episode Time', fontsize=8, labelpad=2)
            ax.set_ylabel(metric_title, fontsize=8, labelpad=3)
            ax.set_xlim(0, 1.02)
            if metric_key not in NO_INSET_METRICS:
                xlo, xhi = ax.get_xlim()
                ticks = sorted(set(
                    t for t in ax.get_xticks().tolist()
                    if xlo <= t <= xhi
                ) | {T_FINAL_LO})
                ax.set_xticks(ticks)

            ax.tick_params(which='both', direction='in',
                           top=True, right=True, labelsize=6, pad=2)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)

            x_off = -0.20 if col in (0, 2) else -0.18
            ax.text(x_off, 1.13, label,
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold', va='top', ha='left')

            if MAIN_YLIM.get(metric_key) is not None:
                ax.set_ylim(*MAIN_YLIM[metric_key])

            if metric_key not in NO_INSET_METRICS:
                fig.canvas.draw()
                add_final_inset(ax, fig, sub, metric_key,
                                inset_ylim=INSET_YLIM.get(metric_key))

    ax_leg.legend(
        handles=[
            mlines.Line2D([], [],
                          color=COLORS[m], marker=MARKERS[m],
                          markersize=4, linewidth=0.8,
                          markerfacecolor=COLORS[m],
                          markeredgecolor='white', markeredgewidth=0.3,
                          label=METHOD_DISPLAY.get(m, m))
            for m in METHOD_ORDER
        ],
        loc='center', ncol=len(METHOD_ORDER),
        frameon=False, fontsize=7,
        handlelength=1.8, handletextpad=0.4,
        columnspacing=1.2, borderpad=0,
    )

    fig.canvas.draw()

    PAD_L = 0.050
    PAD_B = 0.070
    PAD_R = 0.008
    PAD_T = 0.045

    for net_idx, (net_type, net_label) in enumerate(NETWORKS):
        col_lo = net_idx * 2
        col_hi = col_lo + 1

        pos_list = [axes[r][c].get_position()
                    for r in range(2) for c in (col_lo, col_hi)]
        x0 = min(p.x0 for p in pos_list) - PAD_L
        x1 = max(p.x1 for p in pos_list) + PAD_R
        y0 = min(p.y0 for p in pos_list) - PAD_B
        y1 = max(p.y1 for p in pos_list) + PAD_T

        border = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle='square,pad=0',
            linewidth=1.0, edgecolor=C_BORDER, facecolor='none',
            linestyle=(0, (4, 2)),
            transform=fig.transFigure, clip_on=False, zorder=30,
        )
        fig.add_artist(border)

        fig.text(
            (x0 + x1) / 2, y1 + 0.004,
            net_label,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            transform=fig.transFigure,
        )

    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(CSV_PATH))[0]
        stem = os.path.join(SAVE_DIR, f'{base}_{TARGET_DIST}_new')
        fig.savefig(stem + '.pdf', dpi=300)
        fig.savefig(stem + '.png', dpi=300)
        print(f"  Saved: {stem}.pdf / .png")

    plt.show()
    print("Done.")


if __name__ == '__main__':
    main()