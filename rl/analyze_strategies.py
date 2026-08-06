import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils


SINGLES_FOLDER = 'data/out/generative/centaur-70B-adapter/singles'


def classify_simulation(choices: list[str]) -> str:
    """Classify a sequence of choices as 'fixed', 'alternating', or 'average'.

    fixed:       every choice is the same letter
    alternating: every consecutive pair switches (U->P or P->U, every trial)
    """
    if len(set(choices)) == 1:
        return 'fixed'
    switches = [choices[i] != choices[i - 1] for i in range(1, len(choices))]
    if all(switches):
        return 'alternating'
    return 'average'


def load_and_classify(folder: str) -> pd.DataFrame:
    """Load all participant CSVs, classify each, return a summary DataFrame.

    Returns a DataFrame with columns:
        participant_id, strategy, choice_sequence (list), df (full trial data)
    """
    id_re = re.compile(r'participant_(\d+)')
    records = []

    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.csv'):
            continue
        m = id_re.search(fname)
        if not m:
            continue
        pid = int(m.group(1))
        df = pd.read_csv(os.path.join(folder, fname))
        choices = df['choice'].tolist()
        strategy = classify_simulation(choices)
        records.append({'participant_id': pid, 'strategy': strategy,
                        'choices': choices, 'df': df})

    return pd.DataFrame(records)


def strategy_percentages(summary: pd.DataFrame) -> dict:
    """Return percentage of simulations per strategy."""
    counts = summary['strategy'].value_counts()
    total = len(summary)
    return {k: 100 * v / total for k, v in counts.items()}


def plot_strategies(summary: pd.DataFrame, out_path: str = 'figures/strategy_analysis.png'):
    """Two-panel figure: (left) strategy breakdown bar chart, (right) cumulative
    reward traces for fixed and alternating participants, coloured by strategy."""
    pct = strategy_percentages(summary)
    strategy_colors = {'fixed': '#D55E00', 'alternating': '#0072B2', 'average': '#999999'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- left panel: bar chart ---
    ax = axes[0]
    labels = list(pct.keys())
    values = [pct[k] for k in labels]
    colors = [strategy_colors.get(k, '#333333') for k in labels]
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('% of simulations')
    ax.set_title('Strategy distribution')
    ax.set_ylim(0, max(values) * 1.2)
    plotting_utils.remove_bar_frame(ax)
    plotting_utils.style_y_gridlines(ax)

    # --- right panel: cumulative reward traces ---
    ax = axes[1]
    plotted = {'fixed': False, 'alternating': False}
    for _, row in summary.iterrows():
        strat = row['strategy']
        if strat not in ('fixed', 'alternating'):
            continue
        color = strategy_colors[strat]
        label = strat if not plotted[strat] else None
        ax.plot(row['df']['trial_index'], row['df']['cumulative_reward'],
                color=color, alpha=0.6, linewidth=5.0, label=label)
        plotted[strat] = True

    ax.set_xlabel('Trial')
    ax.set_ylabel('Cumulative reward')
    ax.set_title('Cumulative reward: fixed vs alternating')
    ax.legend()
    plotting_utils.style_y_gridlines(ax)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'Saved plot to {out_path}')
    return fig


def plot_bandit1_choice_rate(summary: pd.DataFrame, out_path: str = 'figures/bandit1_choice_rate.png',
                             fig_size: tuple = (12, 10)):
    """Plot rolling-5 bandit-1 choice rate as individual traces per participant, grouped by strategy."""
    strategy_colors = {'fixed': 'red', 'alternating': '#9467BD', 'average': 'grey'}

    plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=20)
    fig, ax = plt.subplots(figsize=fig_size)

    for strat, color in strategy_colors.items():
        group = summary[summary['strategy'] == strat]
        #plot only the first 3 of each strategy to avoid overcrowding
        group = group.head(3)
        if group.empty:
            continue
        first = True
        for _, row in group.iterrows():
            is_bandit1 = (row['df']['choice'] == 1).astype(float)
            rate = is_bandit1.rolling(5, min_periods=1).mean().values
            trials = row['df']['trial_index'].values
            ax.plot(trials, rate, color=color, linewidth=3.0, alpha=0.6,
                    label=strat if first else None)
            first = False

    pad_value = fig_size[1] * 3
    ax.set_xlabel('Trial Number', labelpad=pad_value)
    ax.set_ylabel('Choice Rate (Arm 1)', labelpad=20)
    ax.set_ylim(-0.05, 1.05)
    plotting_utils.style_y_gridlines(ax)
    ax.margins(x=0.04)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.22),
                    ncols=3, frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved plot to {out_path}')
    return fig


if __name__ == '__main__':
    summary = load_and_classify(SINGLES_FOLDER)
    pct = strategy_percentages(summary)

    print(f'\nStrategy breakdown across {len(summary)} simulations:')
    for strategy, p in sorted(pct.items(), key=lambda x: -x[1]):
        n = (summary['strategy'] == strategy).sum()
        print(f'  {strategy:15s}: {n:3d} ({p:.1f}%)')

    plot_strategies(summary, out_path='figures/strategy_analysis.png')
    plot_bandit1_choice_rate(summary, out_path='figures/bandit1_choice_rate_new.png')
