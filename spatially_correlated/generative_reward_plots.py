"""
Plot Average Reward and Maximum Reward over click sequence.

Conditions:
  - Colour:     scenario (0=accumulation, 1=maximization)
  - Line style: horizon (solid=long, dashed=short)
  - Shading:    SEM across participants

Supports human data (experimentData1D.csv) and model data
(generative output CSVs under data/out/generative/).
"""

import os
import json
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── colour / style constants ──────────────────────────────────────────────────
# Per-source scenario colours: (accumulation colour, maximization colour)
SOURCE_COLORS = {
    'Human':               ('#B22222', '#1a1a1a'),   # red / near-black
    'centaur-70B-adapter': ('#E69F00', '#D55E00'),   # orange / dark-red
    'llama-70B-adapter':   ('#56B4E9', '#7B2D8B'),   # sky-blue / violet
}
SCENARIO_LABELS = {0: 'Accumulation', 1: 'Maximization'}
HORIZON_STYLES  = {'long': '-', 'short': '--'}
HORIZON_LABELS  = {'long': 'Long horizon', 'short': 'Short horizon'}


# ── helpers ───────────────────────────────────────────────────────────────────

def _trial_stats(trial_avg_matrix, trial_max_matrix):
    """
    trial_avg_matrix, trial_max_matrix: shape (N_participants, N_trials)
    Returns (mean_avg, sem_avg, mean_max, sem_max) each of shape (N_trials,).
    """
    def _ms(mat):
        n = mat.shape[0]
        m = mat.mean(axis=0)
        s = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(mat.shape[1])
        return m, s
    mean_avg, sem_avg = _ms(trial_avg_matrix)
    mean_max, sem_max = _ms(trial_max_matrix)
    return mean_avg, sem_avg, mean_max, sem_max


# ── human data loader ─────────────────────────────────────────────────────────

def load_human_data(csv_path):
    """
    Returns a list of dicts with keys:
      participant_id, scenario, kernel, horizon ('short'|'long'), rewards (list)
    One entry per (participant × round).
    kernel encoding: 0=smooth, 1=rough
    """
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        sh = json.loads(row['searchHistory'])
        scenario = int(row['scenario'])
        kernel   = 'smooth' if int(row['kernel']) == 0 else 'rough'
        pid      = row['id']
        for round_rewards in sh['ycollect']:
            n = len(round_rewards)
            horizon = 'short' if n <= 6 else 'long'   # 6=5+init, 11=10+init
            records.append({
                'participant_id': pid,
                'scenario': scenario,
                'kernel': kernel,
                'horizon': horizon,
                'rewards': round_rewards,
            })
    return records


# ── model data loader ─────────────────────────────────────────────────────────

def load_model_data(folder_path):
    """
    Load all participant CSVs under <folder_path>/singles/.
    Returns a list of dicts with the same keys as load_human_data.
    """
    singles = os.path.join(folder_path, 'singles')
    records = []
    for fpath in glob.glob(os.path.join(singles, 'participant_*.csv')):
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        pid = int(os.path.basename(fpath).split('_')[1])
        df['model_id'] = pid

        # group by (env_idx, horizon) to reconstruct individual rounds
        for (env_idx, horizon_val), grp in df.groupby(['env_idx', 'horizon']):
            grp = grp.sort_values('trial')
            scenario = int(grp['scenario'].iloc[0])
            kernel   = str(grp['kernel'].iloc[0])
            horizon  = 'short' if int(horizon_val) <= 5 else 'long'
            rewards  = grp['reward'].tolist()
            # prepend init_reward so click 0 is the forced starting choice
            init_r = grp['init_reward'].iloc[0]
            rewards = [init_r] + rewards
            records.append({
                'participant_id': pid,
                'scenario': scenario,
                'kernel': kernel,
                'horizon': horizon,
                'rewards': rewards,
            })
    return records


# ── aggregation ───────────────────────────────────────────────────────────────

def aggregate(records, kernels=None):
    """
    Group records by (scenario, kernel, horizon) and compute per-group stats.
    kernels: list of kernel strings to include, or None for all.

    Returns dict: (scenario, kernel, horizon) -> (x, mean_avg, sem_avg, mean_max, sem_max)
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        if kernels and r['kernel'] not in kernels:
            continue
        key = (r['scenario'], r['kernel'], r['horizon'])
        buckets[key].append(r['rewards'])

    out = {}
    for key, reward_lists in buckets.items():
        # Trim all to shortest round in the group (handles off-by-one)
        min_len = min(len(r) for r in reward_lists)
        trimmed = [r[:min_len] for r in reward_lists]
        mean_avg, sem_avg, mean_max, sem_max = _running_stats(trimmed)
        x = np.arange(min_len)
        out[key] = (x, mean_avg, sem_avg, mean_max, sem_max)
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_rewards(stats, title='', kernels_shown=None):
    """
    stats: dict from aggregate()
    Returns a matplotlib Figure with two subplots (avg reward, max reward).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    ax_avg, ax_max = axes

    plotted_scenarios = set()
    plotted_horizons  = set()

    scenarios = sorted({k[0] for k in stats})
    horizons  = ['long', 'short']

    for scenario in scenarios:
        color = SCENARIO_COLORS[scenario]
        for horizon in horizons:
            ls = HORIZON_STYLES[horizon]

            # If kernels_shown is specified, average over those kernels
            keys = [k for k in stats if k[0] == scenario and k[2] == horizon
                    and (kernels_shown is None or k[1] in kernels_shown)]

            if not keys:
                continue

            # Pool across kernels: concatenate underlying data by re-computing
            # For simplicity, average the pre-computed means (good approximation)
            x_all   = [stats[k][0] for k in keys]
            avg_all = [stats[k][1] for k in keys]
            sem_avg_all = [stats[k][2] for k in keys]
            max_all = [stats[k][3] for k in keys]
            sem_max_all = [stats[k][4] for k in keys]

            min_len = min(len(x) for x in x_all)
            x = x_all[0][:min_len]
            mean_avg = np.mean([a[:min_len] for a in avg_all], axis=0)
            sem_avg  = np.mean([s[:min_len] for s in sem_avg_all], axis=0)
            mean_max = np.mean([a[:min_len] for a in max_all], axis=0)
            sem_max  = np.mean([s[:min_len] for s in sem_max_all], axis=0)

            # click index starting at 1
            x_plot = x + 1

            for ax, mean, sem in [
                (ax_avg, mean_avg, sem_avg),
                (ax_max, mean_max, sem_max),
            ]:
                ax.plot(x_plot, mean, color=color, ls=ls, lw=2)
                ax.fill_between(x_plot, mean - sem, mean + sem,
                                color=color, alpha=0.2)

            plotted_scenarios.add(scenario)
            plotted_horizons.add(horizon)

    # Labels & formatting
    for ax, ylabel in [
        (ax_avg, 'Average Reward'),
        (ax_max, 'Maximum Reward'),
    ]:
        ax.set_xlabel('Click number')
        ax.set_ylabel(ylabel)
        ax.spines[['top', 'right']].set_visible(False)

    # Legend: scenario colours
    scenario_handles = [
        mlines.Line2D([], [], color=SCENARIO_COLORS[s], lw=2,
                      label=SCENARIO_LABELS[s])
        for s in sorted(plotted_scenarios)
    ]
    # Legend: horizon line styles
    horizon_handles = [
        mlines.Line2D([], [], color='black', ls=HORIZON_STYLES[h], lw=2,
                      label=HORIZON_LABELS[h])
        for h in ['long', 'short'] if h in plotted_horizons
    ]

    ax_avg.legend(handles=scenario_handles + horizon_handles,
                  frameon=False, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()
    return fig


# ── drawing helper ────────────────────────────────────────────────────────────

def _draw_source(axes_pair, stats, source_name, kernel_filter,
                 lw, alpha, shade_alpha, zorder, plotted_scenarios, plotted_horizons):
    """Draw one source (human or model) onto (ax_avg, ax_max)."""
    ax_avg, ax_max = axes_pair
    colors = SOURCE_COLORS.get(source_name, ('#333333', '#888888'))
    scenario_colors = {0: colors[0], 1: colors[1]}

    for scenario in sorted({k[0] for k in stats}):
        for horizon in ['long', 'short']:
            ls = HORIZON_STYLES[horizon]
            keys = [k for k in stats
                    if k[0] == scenario and k[2] == horizon
                    and (kernel_filter is None or k[1] in kernel_filter)]
            if not keys:
                continue
            color = scenario_colors[scenario]
            min_len = min(len(stats[k][0]) for k in keys)
            x = stats[keys[0]][0][:min_len] + 1
            mean_avg = np.mean([stats[k][1][:min_len] for k in keys], axis=0)
            sem_avg  = np.mean([stats[k][2][:min_len] for k in keys], axis=0)
            mean_max = np.mean([stats[k][3][:min_len] for k in keys], axis=0)
            sem_max  = np.mean([stats[k][4][:min_len] for k in keys], axis=0)
            for ax, mean, sem in [(ax_avg, mean_avg, sem_avg),
                                  (ax_max, mean_max, sem_max)]:
                ax.plot(x, mean, color=color, ls=ls, lw=lw,
                        alpha=alpha, zorder=zorder)
                ax.fill_between(x, mean - sem, mean + sem,
                                color=color, alpha=shade_alpha,
                                zorder=zorder - 1)
            plotted_scenarios.add(scenario)
            plotted_horizons.add(horizon)


def _build_legend_handles(source_name, model_name, plotted_scenarios, plotted_horizons):
    human_colors  = SOURCE_COLORS.get('Human', ('#B22222', '#1a1a1a'))
    model_colors  = SOURCE_COLORS.get(model_name, ('#333333', '#888888')) if model_name else None

    handles = []
    # scenario colour patches – one row per source
    for s in sorted(plotted_scenarios):
        lbl = SCENARIO_LABELS[s]
        handles.append(mlines.Line2D([], [], color=human_colors[s], lw=2.5,
                                     label=f'Human – {lbl}'))
        if model_colors:
            handles.append(mlines.Line2D([], [], color=model_colors[s], lw=1.5,
                                         alpha=0.6, label=f'{model_name} – {lbl}'))
    handles.append(mlines.Line2D([], [], color='none', label=''))  # spacer
    for h in ['long', 'short']:
        if h in plotted_horizons:
            handles.append(mlines.Line2D([], [], color='grey', ls=HORIZON_STYLES[h],
                                         lw=1.8, label=HORIZON_LABELS[h]))
    return handles


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot average and max reward over click sequence.'
    )
    parser.add_argument('--human', default=None,
                        help='Path to experimentData1D.csv')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Paths to generative model folders')
    parser.add_argument('--out_prefix', default='figures/reward',
                        help='Output PNG prefix; kernel suffix appended automatically')
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    if args.human is None and args.models is None:
        args.human = os.path.join(base, 'data/in/experimentData1D.csv')
        args.models = [
            os.path.join(base, 'data/out/generative/centaur-70B-adapter'),
            os.path.join(base, 'data/out/generative/llama-70B-adapter'),
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)

    # ── load all data ──────────────────────────────────────────────────────────
    sources = {}
    if args.human and os.path.exists(args.human):
        sources['Human'] = load_human_data(args.human)
        print(f"Loaded {len(sources['Human'])} human rounds")
    for model_path in (args.models or []):
        name = os.path.basename(model_path)
        records = load_model_data(model_path)
        if records:
            sources[name] = records
            print(f"Loaded {len(records)} rounds for {name}")

    human_records = sources.get('Human')
    model_names   = [n for n in sources if n != 'Human']

    # ── one figure per kernel ──────────────────────────────────────────────────
    for kernel in ['smooth', 'rough']:
        kf = [kernel]

        # columns: one per model; rows: avg reward, max reward
        n_cols = max(len(model_names), 1)
        fig, axes = plt.subplots(2, n_cols,
                                 figsize=(7 * n_cols, 10),
                                 sharey='row', squeeze=False)
        fig.suptitle(f'Kernel: {kernel}', fontsize=14, y=1.01)

        for col_idx, model_name in enumerate(model_names or [None]):
            ax_avg = axes[0, col_idx]
            ax_max = axes[1, col_idx]
            axes_pair = (ax_avg, ax_max)

            plotted_scenarios = set()
            plotted_horizons  = set()

            # model reference behind
            if model_name and model_name in sources:
                m_stats = aggregate(sources[model_name], kernels=kf)
                _draw_source(axes_pair, m_stats, model_name, kf,
                             lw=1.5, alpha=0.55, shade_alpha=0.08,
                             zorder=1,
                             plotted_scenarios=plotted_scenarios,
                             plotted_horizons=plotted_horizons)

            # human on top
            if human_records:
                h_stats = aggregate(human_records, kernels=kf)
                _draw_source(axes_pair, h_stats, 'Human', kf,
                             lw=2.5, alpha=1.0, shade_alpha=0.20,
                             zorder=3,
                             plotted_scenarios=plotted_scenarios,
                             plotted_horizons=plotted_horizons)

            col_title = f'Human  (ref: {model_name})' if model_name else 'Human'
            for ax, ylabel in [(ax_avg, 'Average Reward'),
                               (ax_max, 'Maximum Reward')]:
                ax.set_xlabel('Click number')
                ax.set_ylabel(ylabel)
                ax.spines[['top', 'right']].set_visible(False)
                ax.set_title(col_title)

            legend_handles = _build_legend_handles(
                'Human', model_name, plotted_scenarios, plotted_horizons)
            ax_avg.legend(handles=legend_handles, frameon=False,
                          fontsize=8.5, handlelength=2.2)

        plt.tight_layout()
        out_path = f'{args.out_prefix}_{kernel}.png'
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved {out_path}")
        plt.close(fig)


if __name__ == '__main__':
    main()
