"""
Empirical p(choice_{t+1} | choice_t, reward_t) from generative RL data.

In generative mode the model builds on its own previous choices (no human
choices in context), so consecutive pairs (c_t, r_t) -> c_{t+1} capture its
true autoregressive sequential dependencies.

Three complementary views:
  1. Win-stay / Lose-shift rates  -- overall WSLS tendency
  2. 2x2 transition matrix        -- p(c_{t+1} | c_t), collapsed over reward
  3. Time-resolved stay rate      -- how sequential dependence shifts across trials
                                     (especially around the reversal at trial 50)

Usage:
    python transition_analysis.py
    python transition_analysis.py --base ./data/out/generative --out ./figures --reversal 50
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REVERSAL_TRIAL = 50  # default; overridable via CLI


# ---------------------------------------------------------------------------
# Data loading (mirrors read_data_from_folder in generate_gen_plots.py)
# ---------------------------------------------------------------------------

def load_generative_data(base_path):
    """
    Returns {model_name: df} where df has at minimum:
        model_id, trial_index, choice (letter), reward
    """
    model_dfs = {}
    id_re = re.compile(r'(?:model|participant)_(\d+)')

    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue

        singles = os.path.join(model_path, 'singles')
        search = singles if os.path.isdir(singles) else model_path

        frames = []
        auto_id = 0
        for fname in sorted(os.listdir(search)):
            if not fname.lower().endswith('.csv'):
                continue
            fpath = os.path.join(search, fname)
            try:
                df = pd.read_csv(fpath)
            except Exception as e:
                print(f"  skip {fpath}: {e}")
                continue

            # Normalise trial column
            if 'trial' in df.columns and 'trial_index' not in df.columns:
                df = df.rename(columns={'trial': 'trial_index'})

            # Assign model_id from filename if absent
            if 'model_id' not in df.columns:
                m = id_re.search(fname.lower())
                df['model_id'] = int(m.group(1)) if m else auto_id
                auto_id += 1

            frames.append(df)

        if not frames:
            continue

        combined = pd.concat(frames, ignore_index=True)
        model_dfs[model_name] = combined
        print(f"  {model_name}: {combined['model_id'].nunique()} runs, "
              f"{len(combined)} total rows")

    return model_dfs


# ---------------------------------------------------------------------------
# Transition computation
# ---------------------------------------------------------------------------

def add_transitions(df):
    """
    Add prev_choice, prev_reward, and stayed (bool) columns to df.
    Operates within each model_id so there is no bleed across runs.
    Returns a copy with consecutive-pair rows (first trial of each run is dropped).
    """
    df = df.sort_values(['model_id', 'trial_index']).copy()
    df['prev_choice'] = df.groupby('model_id')['choice'].shift(1)
    df['prev_reward'] = df.groupby('model_id')['reward'].shift(1)
    df['stayed'] = (df['choice'] == df['prev_choice']).astype(float)
    return df.dropna(subset=['prev_choice', 'prev_reward'])


def _encode_choices(df):
    """
    Within each model_id run, map the two choice letters to 0/1 by alphabetical
    order so matrices are comparable across runs that used different letters.
    Returns df with added columns choice_enc and prev_choice_enc.
    """
    def encode_run(grp):
        opts = sorted(grp['choice'].dropna().unique())
        if len(opts) != 2:
            return grp
        mapping = {opts[0]: 0, opts[1]: 1}
        grp = grp.copy()
        grp['choice_enc']      = grp['choice'].map(mapping)
        grp['prev_choice_enc'] = grp['prev_choice'].map(mapping)
        return grp

    encoded = df.groupby('model_id', group_keys=False).apply(encode_run)
    return encoded.dropna(subset=['choice_enc', 'prev_choice_enc'])


def _crosstab_2x2(encoded):
    mat = pd.crosstab(
        encoded['prev_choice_enc'].astype(int),
        encoded['choice_enc'].astype(int),
        normalize='index'
    )
    mat.index   = [f'option_{i}' for i in mat.index]
    mat.columns = [f'option_{i}' for i in mat.columns]
    mat.index.name   = 'choice_t'
    mat.columns.name = 'choice_t+1'
    return mat


def transition_matrix(df):
    """2x2 p(choice_{t+1} | choice_t), collapsed over reward."""
    return _crosstab_2x2(_encode_choices(df))


def reward_conditioned_transition_matrix(df):
    """
    Returns {1: matrix_after_win, 0: matrix_after_loss}.
    Gives the 2x2x2 breakdown: prev_reward × prev_choice × next_choice.
    """
    encoded = _encode_choices(df)
    return {
        r: _crosstab_2x2(encoded[encoded['prev_reward'] == r])
        for r in [1, 0]
    }


def wsls_rates(df):
    """
    Returns a Series:
        win_stay  = p(stayed | prev_reward == 1)
        lose_stay = p(stayed | prev_reward == 0)   (1 - lose_shift)
    """
    wins  = df[df['prev_reward'] == 1]['stayed'].mean()
    loses = df[df['prev_reward'] == 0]['stayed'].mean()
    return pd.Series({'win_stay': wins, 'lose_stay': loses,
                      'lose_shift': 1 - loses})


def time_resolved_stay(df):
    """
    Returns a Series indexed by trial_index giving the mean stay rate at that
    trial across all runs.  trial_index here is the *current* trial (t+1).
    """
    return df.groupby('trial_index')['stayed'].mean()


def wsls_rates_per_run(df):
    """
    Returns a DataFrame with one row per model_id:
        model_id, win_stay, lose_shift
    Used for the per-run scatter plot.
    """
    rows = []
    for model_id, grp in df.groupby('model_id'):
        win_grp  = grp[grp['prev_reward'] == 1]
        lose_grp = grp[grp['prev_reward'] == 0]
        lose_stay_mean = lose_grp['stayed'].mean() if len(lose_grp) > 0 else np.nan
        rows.append({
            'model_id':   model_id,
            'win_stay':   win_grp['stayed'].mean()  if len(win_grp)  > 0 else np.nan,
            'lose_stay':  lose_stay_mean,
            'lose_shift': 1 - lose_stay_mean if not np.isnan(lose_stay_mean) else np.nan,
        })
    return pd.DataFrame(rows).dropna()


QUADRANT_LABELS = {
    (True,  False): 'explorers',        # high win-stay, low lose-stay (WSLS)
    (True,  True):  'perseverators',    # high win-stay, high lose-stay (always stay)
    (False, False): 'always shift',     # low  win-stay, low lose-stay
    (False, True):  'lose-stay biased', # low  win-stay, high lose-stay
}

def quadrant_label(win_stay, lose_stay, threshold=0.5):
    return QUADRANT_LABELS[(win_stay >= threshold, lose_stay >= threshold)]


def cluster_proportions(df_run, threshold=0.5):
    """
    Returns a dict mapping quadrant label → fraction of runs in that quadrant.
    """
    df_run = df_run.copy()
    df_run['quadrant'] = df_run.apply(
        lambda r: quadrant_label(r['win_stay'], r['lose_stay'], threshold), axis=1
    )
    counts = df_run['quadrant'].value_counts()
    total  = len(df_run)
    return {label: counts.get(label, 0) / total for label in QUADRANT_LABELS.values()}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    'centaur': '#D55E00',
    'llama':   '#0072B2',
    'rw':      '#CC79A7',
}

MARKERS = {
    'centaur': 'o',
    'llama':   '^',
    'rw':      's',
}

def _model_color(name):
    nl = name.lower()
    for key, col in COLORS.items():
        if key in nl:
            return col
    return '#666666'


def _model_marker(name):
    nl = name.lower()
    for key, marker in MARKERS.items():
        if key in nl:
            return marker
    return 'o'


def plot_wsls_scatter(wsls_per_run_dict, save_path=None):
    """
    2D scatter of win-stay vs lose-shift per run, one point per simulation.

    Quadrants:
        top-right    explorers     (win-stay ≥ 0.5, lose-shift ≥ 0.5)
        bottom-right perseverators (win-stay ≥ 0.5, lose-shift < 0.5)
        top-left     adapters      (win-stay < 0.5, lose-shift ≥ 0.5)
        bottom-left  non-adapters  (both < 0.5)

    Below the scatter, a table of quadrant proportions per model is printed
    to stdout and saved alongside the figure as a CSV.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    prop_rows = []
    for model_name, df_run in wsls_per_run_dict.items():
        color  = _model_color(model_name)
        marker = _model_marker(model_name)
        ax.scatter(df_run['win_stay'], df_run['lose_stay'],
                   color=color, marker=marker, s=65, alpha=0.75,
                   label=model_name, zorder=3, linewidths=0.4,
                   edgecolors='white')

        props = cluster_proportions(df_run)
        prop_rows.append({'model': model_name, **props})

    # Quadrant dividers
    ax.axvline(0.5, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.7)
    ax.axhline(0.5, color='#aaaaaa', linestyle='--', linewidth=0.9, alpha=0.7)

    # Quadrant corner labels
    kw = dict(fontsize=9, color='#888888', alpha=0.85)
    ax.text(0.02, 0.98, 'lose-stay biased', transform=ax.transAxes, ha='left',  va='top',    **kw)
    ax.text(0.98, 0.98, 'perseverators',    transform=ax.transAxes, ha='right', va='top',    **kw)
    ax.text(0.02, 0.02, 'always shift',     transform=ax.transAxes, ha='left',  va='bottom', **kw)
    ax.text(0.98, 0.02, 'explorers',        transform=ax.transAxes, ha='right', va='bottom', **kw)

    ax.set_xlabel('win-stay proportion (per run)')
    ax.set_ylabel('lose-stay proportion (per run)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title('Win-stay vs Lose-stay per run\n(generative data, autoregressive)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")

    # Print and return proportion table
    prop_df = pd.DataFrame(prop_rows).set_index('model')
    print("\n=== Quadrant proportions (fraction of runs per model) ===")
    print(prop_df.to_string(float_format='{:.2f}'.format))

    if save_path:
        csv_path = save_path.replace('.png', '_proportions.csv')
        prop_df.to_csv(csv_path)
        print(f"Saved: {csv_path}")

    return fig, prop_df


def plot_wsls(wsls_per_run_dict, save_path=None):
    """Bar chart comparing win-stay and lose-stay rates per model with SEM error bars."""
    models = list(wsls_per_run_dict.keys())
    n = len(models)
    x = np.arange(n)
    w = 0.35

    win_means, win_sems = [], []
    lstay_means, lstay_sems = [], []
    colors = [_model_color(m) for m in models]

    for m in models:
        df_run = wsls_per_run_dict[m]
        ws = df_run['win_stay'].dropna()
        ls = df_run['lose_stay'].dropna()
        win_means.append(ws.mean())
        win_sems.append(ws.sem())
        lstay_means.append(ls.mean())
        lstay_sems.append(ls.sem())

    err_kw = {'elinewidth': 1.2, 'ecolor': 'black', 'capthick': 1.2}
    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))
    ax.bar(x - w/2, win_means,   w, label='Win-Stay',  color=colors, alpha=0.9,
           yerr=win_sems,   capsize=4, error_kw=err_kw)
    ax.bar(x + w/2, lstay_means, w, label='Lose-Stay', color=colors, alpha=0.45,
           hatch='//', yerr=lstay_sems, capsize=4, error_kw=err_kw)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Chance (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Stay Probability')
    ax.set_ylim(0, 1.15)
    ax.set_title('Win-Stay / Lose-Stay rates\n(generative data, autoregressive)')
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")
    return fig


def plot_time_resolved(stay_series_dict, reversal_trial=50, save_path=None):
    """
    Line plot of p(stay at trial t+1) over time.
    stay_series_dict: {model_name: pd.Series indexed by trial_index}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for model_name, series in stay_series_dict.items():
        color = _model_color(model_name)
        ax.plot(series.index, series.values, label=model_name,
                color=color, linewidth=1.8, alpha=0.9)

    ax.axvline(reversal_trial, color='black', linestyle='--', linewidth=1.0,
               alpha=0.4, label=f'Reversal (t={reversal_trial})')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)

    ax.set_xlabel('Trial (t+1)')
    ax.set_ylabel('p(choice_{t+1} == choice_t)')
    ax.set_title('Time-resolved stay rate: p(t+1 | t)\n'
                 '(autoregressive generative data — model conditions on own history)')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")
    return fig


def plot_time_resolved_by_reward(df_dict, reversal_trial=50, save_path=None):
    """
    Stay rate split by previous reward (win-stay vs lose-stay) over time.
    df_dict: {model_name: df_with_transitions}
    """
    n = len(df_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (model_name, df) in zip(axes, df_dict.items()):
        color = _model_color(model_name)
        for reward_val, label, ls in [(1, 'After win (win-stay)', '-'),
                                       (0, 'After loss (lose-stay)', '--')]:
            sub = df[df['prev_reward'] == reward_val]
            series = sub.groupby('trial_index')['stayed'].mean()
            ax.plot(series.index, series.values, label=label,
                    color=color, linestyle=ls, linewidth=1.6)

        ax.axvline(reversal_trial, color='black', linestyle=':', linewidth=1.0,
                   alpha=0.4)
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.7)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel('Trial (t+1)')
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('p(stayed)')

    fig.suptitle('Win-stay / Lose-stay over time\n(generative data)', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")
    return fig


def _draw_heatmap(ax, mat, title, show_ylabel):
    """
    Draw one 2x2 heatmap on ax.
    Cells are colored by deviation from 0.5 (diverging: red=below, blue=above).
    Annotation: probability on top line, signed deviation on bottom line.
    """
    dev = mat.values - 0.5
    im = ax.imshow(dev, vmin=-0.5, vmax=0.5, cmap='RdBu')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(mat.columns, fontsize=10)
    ax.set_yticklabels(mat.index, fontsize=10)
    ax.set_xlabel('choice_{t+1}', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('choice_t', fontsize=9)
    ax.set_title(title, fontsize=9, pad=4)

    for i in range(2):
        for j in range(2):
            p   = mat.values[i, j]
            d   = dev[i, j]
            txt_color = 'white' if abs(d) > 0.3 else 'black'
            sign = '+' if d >= 0 else ''
            ax.text(j, i, f'{p:.2f}\n({sign}{d:.2f})',
                    ha='center', va='center', fontsize=10,
                    color=txt_color, linespacing=1.5)
    return im


def plot_transition_heatmaps(trans_dict, cond_dict, save_path=None):
    """
    Three-column heatmap per model: Overall | After win | After loss.

    trans_dict: {model_name: 2x2 collapsed matrix}
    cond_dict:  {model_name: {1: mat_after_win, 0: mat_after_loss}}

    Color encodes deviation from 0.5 (random baseline); each cell shows
    the raw probability and the signed deviation.
    """
    models = list(cond_dict.keys())
    n = len(models)
    columns = [
        ('overall',  'Overall'),
        ('win',      'After win  (reward=1)'),
        ('loss',     'After loss  (reward=0)'),
    ]

    fig, axes = plt.subplots(n, 3, figsize=(10, 3.8 * n),
                             squeeze=False, constrained_layout=True)

    for row, model_name in enumerate(models):
        mats = {
            'overall': trans_dict[model_name],
            'win':     cond_dict[model_name][1],
            'loss':    cond_dict[model_name][0],
        }
        for col, (key, cond_label) in enumerate(columns):
            ax = axes[row][col]
            mat = mats[key]
            col_title = f'{model_name}\n{cond_label}' if col == 0 else cond_label
            im = _draw_heatmap(ax, mat, title=col_title, show_ylabel=(col == 0))

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label('deviation from chance (0.5)', fontsize=9)
    cbar.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle('p(choice_{t+1} | choice_t, reward_t)  —  generative data\n'
                 'color = deviation from random baseline (0.5)',
                 fontsize=10)

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(base_path, out_dir, reversal_trial):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nLoading generative data from: {base_path}")
    model_dfs = load_generative_data(base_path)
    if not model_dfs:
        print("No data found.")
        return

    wsls_dict         = {}
    stay_series       = {}
    trans_df_dict     = {}
    trans_dict        = {}
    cond_dict         = {}
    wsls_per_run_dict = {}

    for model_name, df in model_dfs.items():
        df_t = add_transitions(df)
        wsls_dict[model_name]         = wsls_rates(df_t)
        stay_series[model_name]       = time_resolved_stay(df_t)
        trans_df_dict[model_name]     = df_t
        trans_dict[model_name]        = transition_matrix(df_t)
        cond_dict[model_name]         = reward_conditioned_transition_matrix(df_t)
        wsls_per_run_dict[model_name] = wsls_rates_per_run(df_t)

    # --- Print summary ---
    print("\n=== Win-Stay / Lose-Shift ===")
    for m, s in wsls_dict.items():
        print(f"  {m}: win-stay={s['win_stay']:.3f}  "
              f"lose-shift={s['lose_shift']:.3f}  "
              f"lose-stay={s['lose_stay']:.3f}")

    print("\n=== Reward-conditioned transition matrices ===")
    for m, mats in cond_dict.items():
        print(f"\n  {m}:")
        for r, label in [(1, 'after win'), (0, 'after loss')]:
            print(f"    {label}:")
            print(mats[r].to_string(float_format='{:.3f}'.format))

    # --- Plots ---
    plot_wsls_scatter(wsls_per_run_dict,
                      save_path=os.path.join(out_dir, 'transition_wsls_scatter.png'))

    plot_wsls(wsls_per_run_dict,
              save_path=os.path.join(out_dir, 'transition_wsls.png'))

    plot_time_resolved(stay_series, reversal_trial=reversal_trial,
                       save_path=os.path.join(out_dir, 'transition_time_resolved.png'))

    plot_time_resolved_by_reward(trans_df_dict, reversal_trial=reversal_trial,
                                  save_path=os.path.join(out_dir,
                                                         'transition_by_reward.png'))

    plot_transition_heatmaps(trans_dict, cond_dict,
                             save_path=os.path.join(out_dir,
                                                    'transition_heatmaps.png'))

    plt.close('all')
    print(f"\nAll plots saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='./data/out/generative',
                        help='Root folder with one subfolder per model')
    parser.add_argument('--out', default='./figures/transitions',
                        help='Output directory for plots')
    parser.add_argument('--reversal', type=int, default=REVERSAL_TRIAL,
                        help='Trial number of the reversal (default: 50)')
    args = parser.parse_args()
    run(args.base, args.out, args.reversal)
