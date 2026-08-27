"""
Follow-up analyses on the flip-vs-full-prompt divergence computed by
kl_flip_analysis.py. Reuses that script's saved per-trial CSVs
(figures/kl_flip/kl_flip_full_trials_<model>.csv: participant_id, trial_index,
p_arm1_full, p_arm2_full, p_arm1_flip, p_arm2_flip, kl_flip_full, kl_full_flip)
rather than reloading raw predictive data.

1. Time-resolved KL: mean divergence per trial across participants, across the
   session -- does it track the bandit reversals (trials 36/56/71/86/106,
   REVERSAL_POINTS from generate_gen_plots.py, confirmed against the raw
   input CSV's own 'reversal' column)?
2. KL vs. confidence: bin trials by how confident the full-prompt prediction
   already was (|p_arm1_full - 0.5|) and see whether divergence concentrates
   in already-uncertain trials or also hits confident ones.

Usage:
    python kl_flip_followup.py
    python kl_flip_followup.py --in_dir ./figures/kl_flip --out ./figures/kl_flip
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils
from generate_gen_plots import REVERSAL_POINTS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, 'figures', 'kl_flip')

MODELS = [
    ('llama-70B-adapter', 'Llama', plotting_utils.LLAMA_COLORS),
    ('centaur-70B-adapter', 'Centaur', plotting_utils.CENTAUR_ORANGE),
]

FIGSIZE = (14, 6)
BASE_FONT = 22
CONF_BIN_EDGES = np.linspace(0, 0.5, 9)  # |p_arm1_full - 0.5|, 8 bins


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trial_data(in_dir, model_dir):
    path = os.path.join(in_dir, f'kl_flip_full_trials_{model_dir}.csv')
    df = pd.read_csv(path)
    df['confidence_full'] = (df['p_arm1_full'] - 0.5).abs()
    return df


# ---------------------------------------------------------------------------
# 1. Time-resolved KL
# ---------------------------------------------------------------------------

def time_resolved_kl(df):
    """Mean +/- SEM of both KL directions per trial_index, across participants."""
    g = df.groupby('trial_index')[['kl_flip_full', 'kl_full_flip']]
    return g.mean().rename(columns=lambda c: f'{c}_mean').join(
        g.sem().rename(columns=lambda c: f'{c}_sem'))


def draw_time_resolved_panel(ax, display_name, colors, df, fig_width, base_font=BASE_FONT):
    dark, light = colors
    ts = time_resolved_kl(df)

    for col, color, ls, label in [
        ('kl_flip_full', dark, '-', 'KL(flip‖full)'),
        ('kl_full_flip', light, '--', 'KL(full‖flip)'),
    ]:
        mean = ts[f'{col}_mean']
        sem = ts[f'{col}_sem']
        ax.plot(ts.index, mean, color=color, linestyle=ls, linewidth=2.0, label=label, zorder=3)
        ax.fill_between(ts.index, mean - sem, mean + sem, color=color, alpha=0.2, linewidth=0, zorder=2)

    for i, rp in enumerate(REVERSAL_POINTS):
        ax.axvline(rp, color='black', linestyle=':', linewidth=1.0, alpha=0.4,
                   label='Reversal' if i == 0 else None)

    ax.set_xlabel('Trial')
    ax.set_ylabel('KL divergence (nats)')
    ax.set_title(display_name)
    ax.legend(frameon=False, fontsize=plotting_utils.get_dynamic_fontsize(
        multiplier=0.7, fig_width=fig_width, base_font=base_font))
    plotting_utils.remove_bar_frame(ax)


# ---------------------------------------------------------------------------
# 2. KL vs. confidence
# ---------------------------------------------------------------------------

def kl_vs_confidence(df, bin_edges=CONF_BIN_EDGES):
    """Mean +/- SEM of both KL directions, binned by |p_arm1_full - 0.5|.

    Bins via np.digitize on the interior edges (rather than pd.cut) so bin
    index -> center is a plain array lookup, with no float-precision matching
    against pd.cut's epsilon-adjusted first-bin boundary (include_lowest=True
    nudges bin_edges[0] down internally, which breaks exact-value lookups).
    """
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.clip(np.digitize(df['confidence_full'], bin_edges[1:-1]), 0, len(centers) - 1)
    g = df.assign(_bin=bin_idx).groupby('_bin')[['kl_flip_full', 'kl_full_flip']]
    out = g.mean().rename(columns=lambda c: f'{c}_mean').join(
        g.sem().rename(columns=lambda c: f'{c}_sem'))
    out['confidence_center'] = centers[out.index]
    return out.sort_values('confidence_center')


def draw_confidence_panel(ax, display_name, colors, df, fig_width, base_font=BASE_FONT):
    dark, light = colors
    binned = kl_vs_confidence(df)

    for col, color, ls, marker, label in [
        ('kl_flip_full', dark, '-', 'o', 'KL(flip‖full)'),
        ('kl_full_flip', light, '--', 's', 'KL(full‖flip)'),
    ]:
        ax.errorbar(binned['confidence_center'], binned[f'{col}_mean'],
                    yerr=binned[f'{col}_sem'], color=color, linestyle=ls, marker=marker,
                    markersize=5, linewidth=2.0, capsize=3, label=label, zorder=3)

    ax.set_xlabel('Full-prompt confidence  |p(arm 1) − 0.5|')
    ax.set_ylabel('KL divergence (nats)')
    ax.set_title(display_name)
    ax.legend(frameon=False, fontsize=plotting_utils.get_dynamic_fontsize(
        multiplier=0.7, fig_width=fig_width, base_font=base_font))
    plotting_utils.remove_bar_frame(ax)


# ---------------------------------------------------------------------------
# Figure assembly (combined 1x2 + standalone per model, matching
# kl_flip_analysis.py's convention)
# ---------------------------------------------------------------------------

def _combined_figure(draw_panel_fn, data, title, save_path=None):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE[0] / 2 * n, FIGSIZE[1]))
    if n == 1:
        axes = [axes]
    fig_width = fig.get_size_inches()[0]
    plotting_utils.set_dynamic_fontsize(fig_width=fig_width, base_font=BASE_FONT)

    for ax, (display_name, colors, df) in zip(axes, data):
        draw_panel_fn(ax, display_name, colors, df, fig_width)

    fig.suptitle(title, y=1.03)
    plt.tight_layout()
    if save_path:
        # suptitle sits at y=1.03, above the canvas -- needs bbox_inches='tight'
        # to grow the saved canvas to include it, not save_panel's fixed letterbox.
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        print(f"Saved: {save_path}")
    return fig


def _single_figure(draw_panel_fn, display_name, colors, df, save_path=None, figsize=(7.5, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    plotting_utils.set_dynamic_fontsize(fig_width=figsize[0], base_font=BASE_FONT)
    draw_panel_fn(ax, display_name, colors, df, figsize[0])
    plt.tight_layout()
    if save_path:
        plotting_utils.save_panel(fig, save_path, figsize=figsize)
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    data = []
    for model_dir, display_name, colors in MODELS:
        df = load_trial_data(in_dir, model_dir)
        data.append((display_name, colors, df))
        print(f"{display_name}: {len(df)} trials loaded from "
              f"kl_flip_full_trials_{model_dir}.csv")

    # --- 1. Time-resolved KL ---
    _combined_figure(draw_time_resolved_panel, data,
                     'Flip vs. full-prompt divergence across the session',
                     save_path=os.path.join(out_dir, 'kl_flip_full_time_resolved.png'))
    for (model_dir, _, _), (display_name, colors, df) in zip(MODELS, data):
        _single_figure(draw_time_resolved_panel, display_name, colors, df,
                       save_path=os.path.join(out_dir, f'kl_flip_full_time_resolved_{model_dir}.png'))

    # --- 2. KL vs. confidence ---
    _combined_figure(draw_confidence_panel, data,
                     'Flip vs. full-prompt divergence by prediction confidence',
                     save_path=os.path.join(out_dir, 'kl_flip_full_vs_confidence.png'))
    for (model_dir, _, _), (display_name, colors, df) in zip(MODELS, data):
        _single_figure(draw_confidence_panel, display_name, colors, df,
                       save_path=os.path.join(out_dir, f'kl_flip_full_vs_confidence_{model_dir}.png'))

    plt.close('all')
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default=DEFAULT_DIR,
                        help='Directory containing kl_flip_analysis.py\'s per-trial CSVs')
    parser.add_argument('--out', default=DEFAULT_DIR,
                        help='Output directory for the follow-up plots')
    args = parser.parse_args()
    run(args.in_dir, args.out)
