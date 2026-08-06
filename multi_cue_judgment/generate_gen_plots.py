"""
Generate RMSE-across-training-blocks plots (replicating the paper figure)
for human baseline, Centaur-70B, and Llama-70B.

Conditions:
  1 = verbal additive      (Format=Verbal,  Task=Additive)
  2 = numeric additive     (Format=Numeric, Task=Additive)
  3 = verbal non-additive  (Format=Verbal,  Task=Non-Additive)
  4 = numeric non-additive (Format=Numeric, Task=Non-Additive)

Blocks 1-10 = training; 11-12 = test.
Outlier exclusion (RMSE > Q3 + 1.5*IQR within each task group) is applied
separately for training and test blocks, for both human and LLM data.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils

VERBAL_TO_NUM = {
    'extremely low': 10, 'very low': 20, 'low': 30,
    'somewhat low': 40, 'normal': 50, 'somewhat high': 60,
    'high': 70, 'very high': 80, 'extremely high': 90,
}

COND_META = {
    1: ('Verbal',   'Additive'),
    2: ('Numeric',  'Additive'),
    3: ('Verbal',   'Non-Additive'),
    4: ('Numeric',  'Non-Additive'),
}

HUMAN_PATH = 'data/in/multi_cue_judgment_exp1.csv'
CENTAUR_DIR = 'data/out/generative/centaur-70B-adapter/singles'
LLAMA_DIR   = 'data/out/generative/llama-70B-adapter/singles'

TRAINING_BLOCKS = list(range(1, 11))
TEST_BLOCKS     = [11, 12]
FIGSIZE=(8,6)
BASE_FONT=24

def v2n(val):
    """Map a verbal string (or numeric string/value) to numeric. Returns NaN if unknown."""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        return VERBAL_TO_NUM.get(s.lower(), np.nan)


# ---------------------------------------------------------------------------
# Load functions
# ---------------------------------------------------------------------------

def _iqr_keep(df, blocks, pid_col):
    """Return the set of participant IDs whose mean RMSE over `blocks` is not
    an outlier (> Q3 + 1.5*IQR), computed separately within each task group."""
    sub = df[df['block'].isin(blocks)].copy()
    if sub.empty:
        return set(df[pid_col].unique())
    sub['_task'] = sub['condition'].map({1: 'Additive', 2: 'Additive',
                                         3: 'Non-Additive', 4: 'Non-Additive'})
    rmse_by_pid = (
        sub.groupby([pid_col, '_task'])['error_sq']
        .mean().apply(np.sqrt)
        .reset_index(name='rmse')
    )
    keep = set()
    for _, grp in rmse_by_pid.groupby('_task'):
        q1, q3 = grp['rmse'].quantile([0.25, 0.75])
        cutoff = q3 + 1.5 * (q3 - q1)
        keep.update(grp.loc[grp['rmse'] <= cutoff, pid_col])
    return keep


def load_human(path=HUMAN_PATH):
    df = pd.read_csv(path)
    df = df.rename(columns={
        'Fp': 'participant_id',
        'Block': 'block',
        'condition': 'condition',
        'Response_as numeric output': 'response_num',
        'Criterium_English': 'criterium_raw',
    })
    df['criterium_num'] = df['criterium_raw'].apply(v2n)
    df['response_num'] = pd.to_numeric(df['response_num'], errors='coerce')
    df['error_sq'] = (df['response_num'] - df['criterium_num']) ** 2

    train_keep = _iqr_keep(df, TRAINING_BLOCKS, 'participant_id')
    test_keep  = _iqr_keep(df, TEST_BLOCKS,     'participant_id')

    train_df = df[df['block'].isin(TRAINING_BLOCKS) & df['participant_id'].isin(train_keep)]
    test_df  = df[df['block'].isin(TEST_BLOCKS)     & df['participant_id'].isin(test_keep)]
    return pd.concat([train_df, test_df], ignore_index=True)


def load_model(folder):
    frames = []
    for fname in os.listdir(folder):
        if not fname.endswith('.csv'):
            continue
        fp = os.path.join(folder, fname)
        d = pd.read_csv(fp)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df['criterium_num'] = df['criterium'].apply(v2n)
    df['response_num']  = df['choice'].apply(v2n)
    df['error_sq'] = (df['response_num'] - df['criterium_num']) ** 2

    train_keep = _iqr_keep(df, TRAINING_BLOCKS, 'participant_id')
    test_keep  = _iqr_keep(df, TEST_BLOCKS,     'participant_id')

    train_df = df[df['block'].isin(TRAINING_BLOCKS) & df['participant_id'].isin(train_keep)]
    test_df  = df[df['block'].isin(TEST_BLOCKS)     & df['participant_id'].isin(test_keep)]
    return pd.concat([train_df, test_df], ignore_index=True)


# ---------------------------------------------------------------------------
# RMSE aggregation
# ---------------------------------------------------------------------------

def rmse_by_block_condition(df, pid_col='participant_id'):
    """
    Returns a DataFrame with columns:
      block, condition, mean_rmse, sem_rmse
    aggregated over participants.
    """
    # Per-participant per-block per-condition RMSE
    per_pid = (
        df.groupby([pid_col, 'block', 'condition'])['error_sq']
        .mean()
        .apply(np.sqrt)
        .reset_index(name='rmse')
    )
    agg = (
        per_pid.groupby(['block', 'condition'])['rmse']
        .agg(['mean', 'sem'])
        .reset_index()
        .rename(columns={'mean': 'mean_rmse', 'sem': 'sem_rmse'})
    )
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

AB_YLIM   = (-0.5, 30)
AB_YTICKS = list(range(0, 31, 5))
C_YLIM    = (0, 30)
C_YTICKS  = list(range(0, 31, 5))

def make_figure(agg, title='', colors=('black', 'grey')):
    """
    agg: DataFrame from rmse_by_block_condition
    colors: (verbal_color, numeric_color)

    Returns matplotlib Figure with 4 panels:
      A = Additive training curves
      B = Non-Additive training curves
      C = Summary mean RMSE over training blocks
      D = Summary mean RMSE over test blocks
    """
    verbal_color, numeric_color = colors
    plotting_utils.set_dynamic_fontsize(fig_width=9, base_font=20)
    fig, axes = plt.subplots(2, 2, figsize=(18, 9))
    fig.suptitle(title, fontweight='bold', y=1.01)

    train = agg[agg['block'].isin(TRAINING_BLOCKS)]
    test  = agg[agg['block'].isin(TEST_BLOCKS)]

    def _plot_training(ax, cond_verbal, cond_numeric, xlabel):
        for cond, fmt, color in [
            (cond_verbal,  'Verbal',  verbal_color),
            (cond_numeric, 'Numeric', numeric_color),
        ]:
            sub = train[train['condition'] == cond].sort_values('block')
            if sub.empty:
                continue
            ax.errorbar(
                sub['block'], sub['mean_rmse'], yerr=sub['sem_rmse'],
                marker='o', markersize=6, linewidth=1.2,
                markerfacecolor=color if fmt == 'Verbal' else 'white',
                markeredgecolor=color, color=color,
                label=fmt, capsize=3,
            )
        ax.set_xlabel(xlabel, labelpad=4)
        ax.set_ylabel('RMSE')
        ax.set_xticks(TRAINING_BLOCKS)
        ax.set_ylim(*AB_YLIM)
        ax.set_yticks(AB_YTICKS)
        ax.legend(title='Format', frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        plotting_utils.style_y_gridlines(ax)

    def _plot_summary(ax, panel_label, src_agg, xlabel):
        summary_data = {}
        for cond, (fmt, task) in COND_META.items():
            sub = src_agg[src_agg['condition'] == cond]
            if sub.empty:
                continue
            summary_data[(task, fmt)] = (sub['mean_rmse'].mean(), sub['sem_rmse'].mean())

        x = np.array([0, 1])
        for fmt, color, filled in [
            ('Numeric', numeric_color, False),
            ('Verbal',  verbal_color,  True),
        ]:
            ys    = [summary_data.get((t, fmt), (np.nan, 0))[0] for t in ['Additive', 'Non-Additive']]
            yerrs = [summary_data.get((t, fmt), (np.nan, 0))[1] for t in ['Additive', 'Non-Additive']]
            ax.errorbar(
                x, ys, yerr=yerrs,
                marker='o', markersize=6, linewidth=1.2,
                markerfacecolor=color if filled else 'white',
                markeredgecolor=color, color=color,
                label=fmt, capsize=3,
            )
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['Additive', 'Non-Additive'])
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
        ax.set_xlabel(xlabel, labelpad=4)
        ax.set_ylabel('RMSE')
        ax.set_title(panel_label, loc='left', fontweight='bold')
        ax.set_ylim(*C_YLIM)
        ax.set_yticks(C_YTICKS)
        ax.legend(title='Format', frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
        plotting_utils.style_y_gridlines(ax)
        ax.annotate('', xy=(1, -0.12), xytext=(0, -0.12),
                    xycoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-', color='black'))

    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    ax_a.set_title('A', loc='left', fontweight='bold')
    _plot_training(ax_a, cond_verbal=1, cond_numeric=2, xlabel='Block\nAdditive')

    ax_b.set_title('B', loc='left', fontweight='bold')
    _plot_training(ax_b, cond_verbal=3, cond_numeric=4, xlabel='Block\nNon-Additive')

    _plot_summary(ax_c, 'C', train, xlabel='Task\nTraining Phase')
    _plot_summary(ax_d, 'D', test,  xlabel='Task\nTest Phase')

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_combined_figure(sources):
    """
    Overlay multiple sources in one 4-panel figure.

    sources: list of dicts with keys:
      - 'label'         : str  (e.g. 'Human', 'Centaur', 'Llama')
      - 'agg'           : DataFrame from rmse_by_block_condition
      - 'verbal_color'  : color for verbal/filled markers
      - 'numeric_color' : color for numeric/open markers
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    linestyles = ['-', '--', ':']

    for ax, task_conds, panel_label, xlabel_suffix in zip(
        axes_flat,
        [(1, 2), (3, 4), None, None],
        ['A', 'B', 'C', 'D'],
        ['Additive', 'Non-Additive', 'Task\nTraining', 'Task\nTest'],
    ):
        ax.set_title(panel_label, loc='left', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        plotting_utils.style_y_gridlines(ax)

        if panel_label in ('A', 'B'):
            verbal_cond, numeric_cond = task_conds
            for src, ls in zip(sources, linestyles):
                agg = src['agg']
                train = agg[agg['block'].isin(TRAINING_BLOCKS)]
                for cond, fmt, color, filled in [
                    (verbal_cond,  'Verbal',  src['verbal_color'],  True),
                    (numeric_cond, 'Numeric', src['numeric_color'], False),
                ]:
                    sub = train[train['condition'] == cond].sort_values('block')
                    if sub.empty:
                        continue
                    lbl = f"{src['label']} – {fmt}"
                    ax.errorbar(
                        sub['block'], sub['mean_rmse'], yerr=sub['sem_rmse'],
                        marker='o', markersize=5, linewidth=1.4, linestyle=ls,
                        markerfacecolor=color if filled else 'white',
                        markeredgecolor=color, color=color,
                        label=lbl, capsize=2,
                    )
            ax.set_xlabel(f'Block\n{xlabel_suffix}', labelpad=4)
            ax.set_ylabel('RMSE')
            ax.set_xticks(TRAINING_BLOCKS)
            ax.set_ylim(*AB_YLIM)
            ax.set_yticks(AB_YTICKS)

        else:
            # Panels C and D: summary dots, x = Additive / Non-Additive
            block_set = TRAINING_BLOCKS if panel_label == 'C' else TEST_BLOCKS
            x = np.array([0, 1])
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(['Additive', 'Non-Additive'])
            ax.tick_params(axis='x', pad=10)
            for src, ls in zip(sources, linestyles):
                agg = src['agg']
                phase = agg[agg['block'].isin(block_set)]
                for fmt, color, filled, add_cond, nonadd_cond in [
                    ('Verbal',  src['verbal_color'],  True,  1, 3),
                    ('Numeric', src['numeric_color'], False, 2, 4),
                ]:
                    ys, errs = [], []
                    for cond in [add_cond, nonadd_cond]:
                        sub = phase[phase['condition'] == cond]
                        ys.append(sub['mean_rmse'].mean() if not sub.empty else np.nan)
                        errs.append(sub['sem_rmse'].mean() if not sub.empty else 0)
                    lbl = f"{src['label']} – {fmt}"
                    ax.errorbar(
                        x, ys, yerr=errs,
                        marker='o', markersize=5, linewidth=1.4, linestyle=ls,
                        markerfacecolor=color if filled else 'white',
                        markeredgecolor=color, color=color,
                        label=lbl, capsize=2,
                    )
            ax.set_xlabel(xlabel_suffix, labelpad=4)
            ax.set_ylabel('RMSE')
            ax.set_ylim(*C_YLIM)
            ax.set_yticks(C_YTICKS)

    # Single shared legend outside panels
    handles, labels = axes_flat[0].get_legend_handles_labels()
    ncol = -(-len(handles) // 2)  # ceil(n/2) → at most 2 rows
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=ncol, frameon=False)
    plt.tight_layout()
    return fig


def make_shaded_training_figure(sources, cond_verbal, cond_numeric, task_label, style='markers'):
    """
    Single-panel figure comparing sources on one task (Additive or Non-Additive)
    across training blocks, with SEM shown as shaded bands instead of error bars.

    sources: same structure as in make_combined_figure.
    style: 'markers' -> circle markers + per-source dashed/dotted linestyles
           'plain'   -> no markers, all lines solid
    No legend is drawn on the axes; use save_legend() with the returned
    handles/labels to render the legend separately.

    Returns (fig, handles, labels).
    """
    plotting_utils.set_dynamic_fontsize(fig_width=FIGSIZE[0], base_font=BASE_FONT)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    linestyles = ['-', '--', ':'] if style == 'markers' else ['-', '-', '-']
    marker = 'o' if style == 'markers' else None

    for src, ls in zip(sources, linestyles):
        agg = src['agg']
        train = agg[agg['block'].isin(TRAINING_BLOCKS)]
        for cond, fmt, color, filled in [
            (cond_verbal,  'Verbal',  src['verbal_color'],  True),
            (cond_numeric, 'Numeric', src['numeric_color'], False),
        ]:
            sub = train[train['condition'] == cond].sort_values('block')
            if sub.empty:
                continue
            lbl = f"{src['label']} – {fmt}"
            ax.plot(
                sub['block'], sub['mean_rmse'],
                marker=marker, markersize=5, linewidth=1.4, linestyle=ls,
                markerfacecolor=(color if filled else 'white') if marker else 'none',
                markeredgecolor=color if marker else 'none', color=color,
                label=lbl,
            )
            ax.fill_between(
                sub['block'],
                sub['mean_rmse'] - sub['sem_rmse'],
                sub['mean_rmse'] + sub['sem_rmse'],
                color=color, alpha=0.15, linewidth=0,
            )

    ax.set_title(task_label, fontweight='bold')
    ax.set_xlabel('Block')
    ax.set_ylabel('RMSE')
    ax.set_xticks(TRAINING_BLOCKS)
    ax.set_ylim(*AB_YLIM)
    ax.set_yticks(AB_YTICKS)
    ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
    ax.spines[['top', 'right']].set_visible(False)
    plotting_utils.style_y_gridlines(ax)

    handles, labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    return fig, handles, labels


def save_legend(handles, labels, path, ncol=1):
    """Render handles/labels as a standalone legend figure and save it."""
    fig_leg = plt.figure(figsize=(3.2, 0.32 * len(labels) + 0.3))
    fig_leg.legend(handles, labels, loc='center', frameon=False, ncol=ncol)
    fig_leg.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig_leg)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs('figures', exist_ok=True)

    print("Loading data...")
    human_df   = load_human()
    centaur_df = load_model(CENTAUR_DIR)
    llama_df   = load_model(LLAMA_DIR)

    human_agg   = rmse_by_block_condition(human_df,   pid_col='participant_id')
    centaur_agg = rmse_by_block_condition(centaur_df, pid_col='participant_id')
    llama_agg   = rmse_by_block_condition(llama_df,   pid_col='participant_id')
    
    # Individual plots
    for agg, title, colors, fname in [
        (human_agg,   'Human Baseline', ('black',    'grey'),      'rmse_training_human.png'),
        (centaur_agg, 'Centaur-70B',    tuple(plotting_utils.CENTAUR_ORANGE),  'rmse_training_centaur.png'),
        (llama_agg,   'Llama-Instruct-3.1-70B', tuple(plotting_utils.LLAMA_COLORS),  'rmse_training_llama.png'),
    ]:
        fig = make_figure(agg, title=title, colors=colors)
        out = f'figures/{fname}'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved {out}")

    # Combined comparison plot
    sources = [
        {'label': 'Human',   'agg': human_agg,   'verbal_color': 'black',   'numeric_color': 'grey'},
        {'label': 'Centaur-70B',             'agg': centaur_agg, 'verbal_color': plotting_utils.CENTAUR_ORANGE[0], 'numeric_color': plotting_utils.CENTAUR_ORANGE[1]},
        {'label': 'Llama-Instruct-3.1-70B', 'agg': llama_agg,   'verbal_color': plotting_utils.LLAMA_COLORS[0], 'numeric_color': plotting_utils.LLAMA_COLORS[1]},
    ]
    fig_combined = make_combined_figure(sources)
    out = 'figures/rmse_training_combined.png'
    fig_combined.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved {out}")

    # Separate additive / non-additive plots with shaded SEM bands.
    # Two style variants each: markers+dashed lines, and plain solid lines.
    # Legend is not drawn on the axes; saved once per style as its own file.
    task_specs = [
        ('Additive',     1, 2, 'additive'),
        ('Non-Additive', 3, 4, 'nonadditive'),
    ]
    legend_saved = {'markers': False, 'plain': False}
    for style, style_suffix in [('markers', ''), ('plain', '_plain')]:
        for task_label, cond_verbal, cond_numeric, task_suffix in task_specs:
            fig, handles, labels = make_shaded_training_figure(
                sources, cond_verbal=cond_verbal, cond_numeric=cond_numeric,
                task_label=task_label, style=style,
            )
            out = f'figures/rmse_training_combined_{task_suffix}_shaded{style_suffix}.png'
            fig.savefig(out, dpi=300, bbox_inches='tight')
            print(f"Saved {out}")

            if not legend_saved[style]:
                legend_out = f'figures/rmse_training_combined_shaded_legend{style_suffix}.png'
                save_legend(handles, labels, legend_out)
                print(f"Saved {legend_out}")
                legend_saved[style] = True

    plt.show()


if __name__ == '__main__':
    main()