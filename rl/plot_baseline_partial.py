import ast
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'spatially_correlated'))
sys.path.insert(0, os.path.dirname(__file__))

from predictive_plots import plot_loglikelihood_bars_dynamic, set_dynamic_fontsize
from generate_gen_plots import bandit_1_prop, plot_bandit_choice_trends_single_axis

LEGEND_LABELS = [
    'Centaur-70B:Baseline',
    'Centaur-70B:Flipped',
    'Llama-Instruct-3.1-70B:Baseline',
    'Llama-Instruct-3.1-70B:Flipped',
]

# colours match define_colors_for_families in generate_plots.py
COLORS = {
    'centaur': ['#D55E00', '#E69F00'],
    'llama':   ['#0072B2', '#56B4E9'],
}


# ── loading helpers ───────────────────────────────────────────────────────────
def _read_singles(folder_path):
    singles = os.path.join(folder_path, 'singles')
    search  = singles if os.path.isdir(singles) else folder_path
    id_re   = re.compile(r'(?:model|participant)_(\d+)')
    frames, auto_id = [], 0
    for fname in os.listdir(search):
        if not fname.lower().endswith('.csv') or 'summary' in fname:
            continue
        try:
            df = pd.read_csv(os.path.join(search, fname))
        except Exception:
            continue
        m = id_re.search(fname.lower())
        df['model_id'] = int(m.group(1)) if m else auto_id
        auto_id += 1
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_condition(base_path):
    """Return {model_name: DataFrame} for every model subfolder."""
    result = {}
    for entry in os.listdir(base_path):
        full = os.path.join(base_path, entry)
        if os.path.isdir(full):
            df = _read_singles(full)
            if not df.empty:
                result[entry] = df
    return result


def _find(d, family):
    for k, v in d.items():
        if family.lower() in k.lower():
            return v
    return None


# ── main figure ───────────────────────────────────────────────────────────────
def plot_reversal_learning(
    predictive,
    predictive_no_rewards,
    generative,
    generative_no_rewards,
    nll_col='nll',
    figsize=(24, 10),
    save_path=None,
):
    centaur_pred   = _find(predictive,            'centaur')
    centaur_pred_p = _find(predictive_no_rewards, 'centaur')
    llama_pred     = _find(predictive,            'llama')
    llama_pred_p   = _find(predictive_no_rewards, 'llama')

    centaur_gen    = _find(generative,            'centaur')
    centaur_gen_p  = _find(generative_no_rewards, 'centaur')
    llama_gen      = _find(generative,            'llama')
    llama_gen_p    = _find(generative_no_rewards, 'llama')

    # ── panel A via plot_loglikelihood_bars_dynamic ───────────────────────────
    # Build family_mapping manually so labels show 'Baseline'/'Partial'
    # and colours cycle correctly (fixed in generate_plots.py).
    family_mapping = {
        'centaur': [
            ('centaur-70B-adapter:Baseline', centaur_pred,   'Baseline'),
            ('centaur-70B-adapter:Flipped',  centaur_pred_p, 'Flipped'),
        ],
        'llama': [
            ('llama-70B-adapter:Baseline', llama_pred,   'Baseline'),
            ('llama-70B-adapter:Flipped',  llama_pred_p, 'Flipped'),
        ],
    }
    w_a = figsize[0] * (1 / 2)
    fig_a = plot_loglikelihood_bars_dynamic(
        family_mapping=family_mapping,
        nll_column=nll_col,
        figsize=(w_a, figsize[1]),
        show_xticklabels=True,
    )
    # Fix chance line: predictive_plots uses log(1/30); RL task is binary → log(2)
    ax_a = fig_a.get_axes()[0]
    for line in ax_a.get_lines():
        if line.get_linestyle() in ('--', 'dashed'):
            line.set_ydata([-np.log(0.5)] * 2)
    for txt in ax_a.texts:
        if 'Random' in txt.get_text():
            txt.set_y(-np.log(0.5))
            # move text up slightly to avoid overlap with chance line
            txt.set_position((txt.get_position()[0]-0.4, txt.get_position()[1] ))

    # ── panel B via plot_bandit_choice_trends_single_axis ────────────────────
    def _prep(df, numeric):
        if 'trial_index' in df.columns and 'trial_num' not in df.columns:
            df = df.rename(columns={'trial_index': 'trial_num'})
        return bandit_1_prop(df, 1 if numeric else 'U')

    centaur_gen   = _prep(centaur_gen,   numeric=False)
    centaur_gen_p = _prep(centaur_gen_p, numeric=True)
    llama_gen     = _prep(llama_gen,     numeric=False)
    llama_gen_p   = _prep(llama_gen_p,   numeric=True)

    w_b = figsize[0] * (1 / 2)
    # colors: [human_placeholder, model1..4, rep_placeholder]
    colors_b = ['#000000',
                COLORS['centaur'][0], COLORS['centaur'][1],
                COLORS['llama'][0],   COLORS['llama'][1],
                '#000000']
    fig_b, _ = plot_bandit_choice_trends_single_axis(
        human_df=None, df_rep=None,
        dfs=[centaur_gen, centaur_gen_p, llama_gen, llama_gen_p],
        labels=LEGEND_LABELS,
        colors=colors_b,
        trial_col='trial_num',
        bandit_avg_col='bandit_1_avg',
        reversal_trials=[50],
        xlim=(1, 100),
        margins=True,
        fig_size=(w_b, figsize[1]),
    )

    # ── combine into single figure ────────────────────────────────────────────
    import io
    import matplotlib.image as mpimg
    set_dynamic_fontsize(10)
    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.02, figure=fig)

    for src_fig, slot, letter in [(fig_a, gs[0], 'A'), (fig_b, gs[1], 'B')]:
        buf = io.BytesIO()
        src_fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        buf.seek(0)
        img = mpimg.imread(buf)
        ax = fig.add_subplot(slot)
        ax.imshow(img)
        ax.axis('off')
        ax.text(-0.04, 1.02, letter, transform=ax.transAxes,
             fontweight='bold', va='top')
        plt.close(src_fig)

    handles = [plt.Line2D([0], [0], color=c, linewidth=2)
               for c in [COLORS['centaur'][0], COLORS['centaur'][1],
                          COLORS['llama'][0],   COLORS['llama'][1]]]
    leg=fig.legend(handles, LEGEND_LABELS,
               loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.05))
    for line in leg.get_lines():
        line.set_linewidth(5)
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
    return fig


# ── bandit 1 distribution figure ─────────────────────────────────────────────
def _parse_top2(top2_str):
    """Parse top2 string → dict of {choice: probability}."""
    try:
        return dict(ast.literal_eval(top2_str))
    except Exception:
        return {}


def _prob_bandit1_per_model(df):
    """
    From a raw predictive CSV DataFrame return a tidy DataFrame with columns
    [model_id, trial, prob_bandit1] where:
      - trial is normalised to 1-indexed
      - bandit1 is identified per participant as the most-chosen key in trials 1-25
        (works for any letter-pair labels, not just U/P)
    """
    df = df.copy()
    df['probs'] = df['top2'].apply(_parse_top2)
    t = df['trial_index']
    df['trial'] = t - t.min() + 1

    # identify bandit_1 key per model from early pre-reversal choices
    early = df[df['trial'] <= 25]
    bandit1_key = (
        early.groupby('model_id')['ground_truth']
        .agg(lambda x: x.value_counts().index[0])
        .rename('bandit1_key')
    )
    df = df.merge(bandit1_key, on='model_id')
    df['prob_bandit1'] = df.apply(
        lambda r: r['probs'].get(r['bandit1_key'], np.nan), axis=1
    )
    return df[['model_id', 'trial', 'prob_bandit1']].dropna()


# keep old name as alias so _epoch_prob_U still works
def _prob_U_per_model(df):
    out = _prob_bandit1_per_model(df)
    return out.rename(columns={'prob_bandit1': 'prob_U'})


# Epochs chosen to match the actual task phases visible in the data:
#   1–20  : early learning   (baseline rises 0.5 → 0.83)
#   21–50 : pre-reversal     (baseline stable ~0.88)
#   51–65 : reversal window  (baseline drops sharply to ~0.10)
#   66–100: post-reversal    (baseline stable ~0.09)
_EPOCHS = [
    (1,  20,  'Early\n(1–20)'),
    (21, 50,  'Pre-reversal\n(21–50)'),
    (51, 65,  'Reversal\n(51–65)'),
    (66, 100, 'Post-reversal\n(66–100)'),
]


def _epoch_prob_U(df):
    """Per-model mean P(U) in each epoch.  Returns DataFrame [model_id, epoch, prob_U]."""
    rows = []
    for lo, hi, label in _EPOCHS:
        sub = df[(df['trial'] >= lo) & (df['trial'] <= hi)]
        per_model = sub.groupby('model_id')['prob_U'].mean()
        for mid, val in per_model.items():
            rows.append({'model_id': mid, 'epoch': label, 'prob_U': val})
    return pd.DataFrame(rows)


def plot_bandit_distribution_comparison(
    centaur_baseline, centaur_partial,
    llama_baseline, llama_partial,
    figsize=(14, 6),
    save_path=None,
):
    """
    Violin plots of P(bandit 1) distribution per task phase,
    derived from model-predicted probabilities in the predictive CSVs.
    Baseline vs Partial compared within each epoch for Centaur and Llama.
    """
    epoch_labels = [label for _, _, label in _EPOCHS]
    # reversal boundary sits between epoch index 1 and 2
    reversal_gap_idx = 1   # draw dashed line after this epoch group

    families = [
        ('Centaur-70B', centaur_baseline, centaur_partial,
         COLORS['centaur'][0], COLORS['centaur'][1]),
        ('Llama-Instruct-3.1-70B', llama_baseline, llama_partial,
         COLORS['llama'][0], COLORS['llama'][1]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, (family_name, df_base, df_part, c_base, c_part) in zip(axes, families):
        epo_base = _epoch_prob_U(_prob_U_per_model(df_base))
        epo_part = _epoch_prob_U(_prob_U_per_model(df_part))

        n = len(epoch_labels)
        pos_base = np.arange(n) * 2.2
        pos_part = pos_base + 0.9

        def _draw_violin(rates_df, positions, color, label):
            data = [
                rates_df.loc[rates_df['epoch'] == ep, 'prob_U'].dropna().values
                for ep in epoch_labels
            ]
            # need ≥2 points for a violin; fall back to a single line
            safe = [d if len(d) > 1 else np.array([d[0], d[0]]) if len(d) == 1 else np.array([0.5, 0.5])
                    for d in data]
            parts = ax.violinplot(safe, positions=positions, widths=0.8,
                                  showmedians=True, showextrema=True)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.55)
            for key in ('cmedians', 'cmins', 'cmaxes', 'cbars'):
                if key in parts:
                    parts[key].set_edgecolor(color)
                    parts[key].set_linewidth(1.8)
            ax.scatter([], [], color=color, alpha=0.8, label=label, s=60)

        _draw_violin(epo_base, pos_base, c_base, 'Baseline')
        _draw_violin(epo_part, pos_part, c_part, 'Partial')

        tick_pos = (pos_base + pos_part) / 2
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(epoch_labels, fontsize=8.5)

        # dashed vertical line between pre-reversal and reversal epochs
        gap_x = (pos_part[reversal_gap_idx] + pos_base[reversal_gap_idx + 1]) / 2
        ax.axvline(x=gap_x, color='black', linestyle='--', linewidth=1.0, alpha=0.4)
        ax.text(gap_x + 0.05, 1.01, 'reversal', fontsize=7.5, color='grey',
                ha='left', va='bottom', transform=ax.get_xaxis_transform())

        ax.set_xlabel('Task phase', fontsize=10)
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(family_name, fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.grid(axis='y', linewidth=0.4, alpha=0.4)
        ax.axhline(0.5, color='grey', linewidth=0.8, linestyle=':', alpha=0.6)

    axes[0].set_ylabel('P(Bandit 1) — model prediction', fontsize=10)
    fig.suptitle('How predicted P(Bandit 1) distribution shifts across task phases',
                 fontsize=11, y=1.02)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
    return fig


# ── post-reversal investigation figure ───────────────────────────────────────
def plot_reversal_investigation(
    centaur_baseline, centaur_partial,
    llama_baseline, llama_partial,
    zoom_start=40,
    figsize=(16, 10),
    save_path=None,
):
    """
    Two-panel figure per model family investigating the reversal window:
      Top row : individual participant trajectories (trials zoom_start–100)
                Baseline (left) vs Partial (right)
      Bottom row: per-trial mean ± SEM ribbon, both conditions overlaid,
                  to show that partial reversal is real but lagged/heterogeneous
    """
    families = [
        ('Centaur-70B',
         centaur_baseline, centaur_partial,
         COLORS['centaur'][0], COLORS['centaur'][1]),
        ('Llama-Instruct-3.1-70B',
         llama_baseline, llama_partial,
         COLORS['llama'][0], COLORS['llama'][1]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for col, (family_name, df_base, df_part, c_base, c_part) in enumerate(families):
        tidy_base = _prob_bandit1_per_model(df_base)
        tidy_part = _prob_bandit1_per_model(df_part)

        for tidy, color, label, ax_top in [
            (tidy_base, c_base, 'Baseline', axes[0, col]),
            (tidy_part, c_part, 'Partial',  axes[0, col]),
        ]:
            sub = tidy[tidy['trial'] >= zoom_start]
            for _, grp in sub.groupby('model_id'):
                ax_top.plot(grp['trial'], grp['prob_bandit1'],
                            color=color, alpha=0.18, linewidth=0.9)
            mean = sub.groupby('trial')['prob_bandit1'].mean()
            ax_top.plot(mean.index, mean.values,
                        color=color, linewidth=2.2, label=label)

        axes[0, col].axvline(50, color='black', linestyle='--',
                             linewidth=1.0, alpha=0.5)
        axes[0, col].axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
        axes[0, col].set_ylim(-0.05, 1.10)
        axes[0, col].set_title(f'{family_name} — individual trajectories', fontsize=10)
        axes[0, col].legend(fontsize=8, frameon=False)
        axes[0, col].set_ylabel('P(Bandit 1)', fontsize=9)

        # bottom row: mean ± SEM ribbon for both conditions
        ax_bot = axes[1, col]
        for tidy, color, label in [
            (tidy_base, c_base, 'Baseline'),
            (tidy_part, c_part, 'Partial'),
        ]:
            sub = tidy[tidy['trial'] >= zoom_start]
            grp = sub.groupby('trial')['prob_bandit1']
            mean = grp.mean()
            sem  = grp.std() / np.sqrt(grp.count())
            ax_bot.plot(mean.index, mean.values, color=color, linewidth=2.2, label=label)
            ax_bot.fill_between(mean.index, mean - sem, mean + sem,
                                color=color, alpha=0.25)

        ax_bot.axvline(50, color='black', linestyle='--', linewidth=1.0, alpha=0.5,
                       label='Reversal (trial 50)')
        ax_bot.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
        ax_bot.set_ylim(-0.05, 1.10)
        ax_bot.set_xlabel('Trial', fontsize=9)
        ax_bot.set_ylabel('P(Bandit 1)', fontsize=9)
        ax_bot.set_title(f'{family_name} — mean ± SEM', fontsize=10)
        ax_bot.legend(fontsize=8, frameon=False)

    fig.suptitle(
        'Post-reversal investigation: Partial condition also reverses, but more slowly\n'
        '(individual traces show heterogeneity in reversal speed)',
        fontsize=11, y=1.01,
    )

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
    return fig


# ── convenience entry point ───────────────────────────────────────────────────
def load_and_plot(base_dir='./data/out', save_path='reversal_learning_figure.png'):
    predictive            = load_condition(os.path.join(base_dir, 'predictive'))
    predictive_no_rewards = load_condition(os.path.join(base_dir, 'predictive_flipped'))
    generative            = load_condition(os.path.join(base_dir, 'generative'))
    generative_no_rewards = load_condition(os.path.join(base_dir, 'generative_no_rewards'))
    return plot_reversal_learning(
        predictive, predictive_no_rewards,
        generative, generative_no_rewards,
        save_path=save_path,
    )


def _load_predictive_pair(base_dir):
    predictive            = load_condition(os.path.join(base_dir, 'predictive'))
    predictive_no_rewards = load_condition(os.path.join(base_dir, 'predictive_no_rewards'))
    return (
        _find(predictive,            'centaur'),
        _find(predictive_no_rewards, 'centaur'),
        _find(predictive,            'llama'),
        _find(predictive_no_rewards, 'llama'),
    )


def load_and_plot_distribution(base_dir='./data/out', save_path='bandit1_distribution.png'):
    return plot_bandit_distribution_comparison(
        *_load_predictive_pair(base_dir), save_path=save_path,
    )


def load_and_plot_investigation(base_dir='./data/out', save_path='reversal_investigation.png'):
    return plot_reversal_investigation(
        *_load_predictive_pair(base_dir), save_path=save_path,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='./data/out')
    parser.add_argument('--out',  default='reversal_learning_figure.png')
    parser.add_argument('--dist-out', default=None,
                        help='If set, also save the distribution figure to this path')
    parser.add_argument('--inv-out', default=None,
                        help='If set, also save the reversal investigation figure to this path')
    args = parser.parse_args()
    load_and_plot(base_dir=args.base, save_path=args.out)
    print(f'Saved to {args.out}')
    if args.dist_out:
        load_and_plot_distribution(base_dir=args.base, save_path=args.dist_out)
        print(f'Distribution figure saved to {args.dist_out}')
    if args.inv_out:
        load_and_plot_investigation(base_dir=args.base, save_path=args.inv_out)
        print(f'Investigation figure saved to {args.inv_out}')
