"""
Rebuilds the rl_waltmann summary figure comparing each model's Baseline run
against its no-reward-signal Partial run:

    A - predictive performance: NLL bars, Baseline vs Partial per model
        (predictive_plots.py: plot_loglikelihood_bars_dynamic)
        Baseline <- data/out/predictive, Partial <- data/out/predictive_no_rewards
    B - generative choice-rate trends, Baseline vs Partial per model
        (generate_gen_plots.py: plot_bandit_choice_trends_single_axis)
        Baseline <- data/out/generative, Partial <- data/out/generative_no_rewards
    C - win-stay / lose-stay rates, Baseline vs Partial per model
        (transition_analysis.py: plot_wsls)
        Baseline <- data/out/generative, Partial <- data/out/generative_no_rewards

All three panels draw into a shared 1x3 figure via the ax= embedding support
already used by the 2x2 composite in ../plot_panel_a.py. __main__ also runs
transition_analysis.py's full suite (WSLS scatter, time-resolved stay rate,
reward-conditioned heatmaps, ...) separately on both the Baseline and Partial
generative data, not just Baseline.
"""
import argparse
import os
import sys
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../rl_waltmann
PARENT_DIR = os.path.dirname(SCRIPT_DIR)                 # .../llms_unsloth
sys.path.insert(0, PARENT_DIR)

from generate_gen_plots import (
    plot_bandit_choice_trends_single_axis,
    read_data_from_folder as read_gen_folder,
    ensure_reversal_column,
    compute_chose_preferred,
    REVERSAL_POINTS_LLM_GENERATED,
)
from transition_analysis import (
    build_wsls_per_run_dict,
    plot_wsls,
    run as run_transition_analysis,
)
from predictive_plots import plot_loglikelihood_bars_dynamic, build_family_mapping, _rl_waltmann_test_participants
import plotting_utils

RW_DATA_OUT = os.path.join(SCRIPT_DIR, 'data', 'out')

# Folder name (under data/out/<generative|predictive>[_no_rewards]/) -> (display
# label, color pair). Order here fixes the left-to-right / legend order.
MODEL_INFO = [
    ('centaur-70B-adapter', 'Centaur', plotting_utils.CENTAUR_ORANGE),
    ('llama-70B-adapter', 'Llama', plotting_utils.LLAMA_COLORS),
]
# eLife single-column width (inches)
TEXT_WIDTH = 5.6  

# 1. LaTeX Minipage allocation -- must match the \begin{minipage}[t]{X\textwidth}
# used in figures/panel_no_reward.tex (currently 0.49\textwidth per panel, with
# \hfill eating the remaining 0.02\textwidth as the gap between panels).
MINIPAGE_RATIO = 0.49

# 2. Inner padding/border of `panelbox_2x2` (figures/preamble_panels.tex: "panelbox
# base" sets left=6pt right=6pt, and boxrule=1.2pt draws the frame *outside* that
# padding on each side) -- total horizontal space eaten from the minipage width:
# left + right + 2*boxrule = 6 + 6 + 2*1.2 = 14.4pt = 14.4/72.27in.
PANELBOX_PADDING = 14.4 / 72.27

# Target plot height in inches
TARGETED_FIG_HEIGHT = 2.2  

# Calculate exact image width to avoid LaTeX scaling
PLOT_WIDTH = (TEXT_WIDTH * MINIPAGE_RATIO) - PANELBOX_PADDING

FIGSIZE = (PLOT_WIDTH, TARGETED_FIG_HEIGHT)
BASE_FONT = 12.0
PARTIAL="No\nreward"
BASELINE="Full\nprompt"


def build_panel_a_family_mapping():
    """
    Baseline (data/out/predictive) vs Partial (data/out/predictive_no_rewards)
    NLL, per model family. Returns the family_mapping structure
    plot_loglikelihood_bars_dynamic expects: family -> [(model_name, df,
    'Baseline'), (model_name, df, 'Partial')].
    """
    test_participants = _rl_waltmann_test_participants()
    baseline = build_family_mapping(os.path.join(RW_DATA_OUT, 'predictive'), nll_column='nll',
                                     include_participants=test_participants)
    partial = build_family_mapping(os.path.join(RW_DATA_OUT, 'predictive_no_rewards'), nll_column='nll',
                                    include_participants=test_participants)

    combined = {}
    for family, models in baseline.items():
        model_name, df, _size = models[0]
        combined.setdefault(family, []).append((model_name, df, BASELINE))
    for family, models in partial.items():
        model_name, df, _size = models[0]
        combined.setdefault(family, []).append((model_name, df, PARTIAL))
    return combined


def build_panel_b_data(max_trial_to_plot=None):
    """
    Baseline (data/out/generative) vs Partial (data/out/generative_no_rewards)
    choice-rate trends, per model. Returns dfs/labels/colors (colors[0] is an
    unused human placeholder, kept for compatibility with
    plot_bandit_choice_trends_single_axis's colors[1:] indexing) plus the
    reversal points/trial range actually present in the data.
    """
    dfs, labels, colors = [], [], ['#000000']

    for model_name, display_name, (dark, light) in MODEL_INFO:
        for condition, subdir, color in ((BASELINE, 'generative', dark),
                                          (PARTIAL, 'generative_no_rewards', light)):
            folder = os.path.join(RW_DATA_OUT, subdir, model_name)
            df = read_gen_folder(folder)
            if df.empty:
                raise SystemExit(f"No data found under {folder}")
            if 'trial_num' not in df.columns:
                for src in ('trial_index', 'trial'):
                    if src in df.columns:
                        df = df.rename(columns={src: 'trial_num'})
                        break
            df = ensure_reversal_column(df, reversal_points=REVERSAL_POINTS_LLM_GENERATED)
            df = compute_chose_preferred(df)
            if max_trial_to_plot is not None:
                df = df[df['trial_num'] <= max_trial_to_plot].reset_index(drop=True)
            dfs.append(df)
            labels.append(f"{display_name}:{condition}")
            colors.append(color)

    max_trial = int(max(df['trial_num'].max() for df in dfs))
    in_range_reversals = [p for p in REVERSAL_POINTS_LLM_GENERATED if p <= max_trial]
    return dict(dfs=dfs, labels=labels, colors=colors, max_trial=max_trial,
                in_range_reversals=in_range_reversals)


def build_panel_c_wsls_data():
    """
    Win-stay/lose-stay per-run rates, Baseline (data/out/generative) vs
    Partial (data/out/generative_no_rewards) per model. Returns
    (wsls_per_run_dict, colors) -- wsls_per_run_dict is keyed by
    '<display_name>:<condition>' (not the raw folder name) so a model's
    Baseline and Partial entries can coexist without colliding, and colors is
    a matching per-key list (dark=Baseline, light=Partial) since plot_wsls's
    default substring-matched coloring can't tell those two keys apart on its
    own (both still contain e.g. 'centaur').
    """
    baseline = build_wsls_per_run_dict(os.path.join(RW_DATA_OUT, 'generative'))
    partial = build_wsls_per_run_dict(os.path.join(RW_DATA_OUT, 'generative_no_rewards'))

    wsls_per_run_dict, colors = {}, []
    for model_name, display_name, (dark, light) in MODEL_INFO:
        if model_name in baseline:
            wsls_per_run_dict[f"{display_name}:{BASELINE}"] = baseline[model_name]
            colors.append(dark)
        if model_name in partial:
            wsls_per_run_dict[f"{display_name}:{PARTIAL}"] = partial[model_name]
            colors.append(light)
    return wsls_per_run_dict, colors


def build_abc_figure(family_mapping, bandit_data, wsls_data, wsls_colors,
                      fig_size=(FIGSIZE[0] * 3, FIGSIZE[1]), base_font=BASE_FONT,
                      width_ratios=(1, 2, 1)):
    """Combined 1x3 figure with a shared Baseline/Partial legend below panel B."""
    plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=base_font)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=fig_size,
                                            gridspec_kw={'width_ratios': width_ratios})

    plot_loglikelihood_bars_dynamic(ax=ax_a, family_mapping=family_mapping, nll_column='nll',
                                     num_choices=2, show_xticklabels=True)

    plot_bandit_choice_trends_single_axis(
        ax=ax_b, dfs=bandit_data['dfs'], labels=bandit_data['labels'], colors=bandit_data['colors'],
        reversal_trials=bandit_data['in_range_reversals'], xlim=(1, bandit_data['max_trial']))

    plot_wsls(wsls_data, ax=ax_c, colors=wsls_colors, show_xticklabels=False)

    # Reuse the line handles already drawn on ax_b (colors/labels match exactly)
    # rather than re-deriving them, so the combined figure's legend and the
    # separately-saved legend (from build_panel_b_figure) can never drift apart.
    handles, legend_labels = ax_b.get_legend_handles_labels()
    fig.legend(handles=handles, labels=legend_labels, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05), frameon=True)

    letter_fontsize = plotting_utils.get_dynamic_fontsize(
        multiplier=1.6, fig_width=fig_size[0], base_font=base_font)
    for ax, letter in ((ax_a, 'A'), (ax_b, 'B'), (ax_c, 'C')):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
                fontsize=letter_fontsize, fontweight='bold', va='bottom', ha='left')

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def build_panel_a_figure(family_mapping, figsize=FIGSIZE, num_choices=2):
    """Standalone panel A (NLL bars), same content as the embedded version."""
    return plot_loglikelihood_bars_dynamic(family_mapping=family_mapping, nll_column='nll',
                                            num_choices=num_choices, figsize=figsize,
                                            show_xticklabels=True)


def build_panel_b_figure(bandit_data, fig_size=FIGSIZE):
    """
    Standalone panel B (choice-rate trends). No legend is drawn on the axes
    itself -- plot_bandit_choice_trends_single_axis's standalone path (ax=None)
    already builds and returns a matching legend figure from the same line
    handles/colors, so that's reused as the separately-saved legend rather
    than re-deriving it a second way.
    """
    return plot_bandit_choice_trends_single_axis(
        dfs=bandit_data['dfs'], labels=bandit_data['labels'], colors=bandit_data['colors'],
        reversal_trials=bandit_data['in_range_reversals'], xlim=(1, bandit_data['max_trial']),
        fig_size=fig_size)


def build_panel_c_figure(wsls_data, wsls_colors, figsize=FIGSIZE, base_font=BASE_FONT):
    """Standalone panel C (win-stay/lose-stay bars), same content as the embedded version."""
    return plot_wsls(wsls_data, colors=wsls_colors, show_xticklabels=False,
                      figsize=figsize, base_font=base_font)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=os.path.join(SCRIPT_DIR, 'figures', 'partial'),
                         help="Output folder for the figures (default: figures/partial)")
    args = parser.parse_args()

    FIGURES_DIR = args.out_dir
    os.makedirs(FIGURES_DIR, exist_ok=True)

    family_mapping = build_panel_a_family_mapping()
    bandit_data = build_panel_b_data(max_trial_to_plot=50)
    wsls_data, wsls_colors = build_panel_c_wsls_data()

    fig_abc = build_abc_figure(family_mapping, bandit_data, wsls_data, wsls_colors)
    fig_abc.savefig(os.path.join(FIGURES_DIR, 'panel_abc_baseline_partial_rl_waltmann.png'),
           bbox_inches='tight',
           pad_inches=0.02,
           dpi=300,
           transparent=True)

    fig_a = build_panel_a_figure(family_mapping)
    plotting_utils.save_panel(fig_a, os.path.join(FIGURES_DIR, 'panel_a_baseline_partial_rl_waltmann.png'),
               figsize=FIGSIZE)

    fig_b, fig_legend = build_panel_b_figure(bandit_data)
    plotting_utils.save_panel(fig_b, os.path.join(FIGURES_DIR, 'panel_b_baseline_partial_rl_waltmann.png'),
               figsize=FIGSIZE)
    fig_legend.savefig(os.path.join(FIGURES_DIR, 'legend_baseline_partial_rl_waltmann.png'),
              bbox_inches='tight',
              pad_inches=0.02,
              dpi=300,
              transparent=True)

    fig_c = build_panel_c_figure(wsls_data, wsls_colors)
    plotting_utils.save_panel(fig_c, os.path.join(FIGURES_DIR, 'panel_c_baseline_partial_rl_waltmann.png'),
               figsize=FIGSIZE)

    print(f"Saved combined (A/B/C), panel A, panel B, panel C, and legend PNGs to {FIGURES_DIR}")

    # Full transition-analysis suite (WSLS scatter, time-resolved stay rate,
    # reward-conditioned transition heatmaps, ...) -- previously only ever run
    # on the Baseline (generative) data; now also run on Partial
    # (generative_no_rewards) so both conditions get the complete set of plots,
    # not just the WSLS bars above. No human overlay (rw_path=None), matching
    # the Baseline-vs-Partial-only scope of the rest of this script.
    first_reversal = REVERSAL_POINTS_LLM_GENERATED[0]
    print(f"\nRunning full transition-analysis suite on {BASELINE} (generative) data...")
    run_transition_analysis(
        os.path.join(RW_DATA_OUT, 'generative'),
        os.path.join(FIGURES_DIR, 'transitions_baseline'),
        reversal_trial=first_reversal, rw_path=None)

    print(f"\nRunning full transition-analysis suite on {PARTIAL} (generative_no_rewards) data...")
    run_transition_analysis(
        os.path.join(RW_DATA_OUT, 'generative_no_rewards'),
        os.path.join(FIGURES_DIR, 'transitions_partial'),
        reversal_trial=first_reversal, rw_path=None)
