"""
KL-divergence analysis: flip vs. full-prompt predictive distributions.

'Flip' (data/out/predictive_flipped) mirrors 'full prompt' (data/out/predictive)
except every trial's reward bit is inverted before the prompt is built (see
data/in/test_waltmann_data_flipped.csv vs test_waltmann_data_cleaned.csv --
reward is exactly `1 - reward`, choice and choice_mapped-per-arm are otherwise
identical). The two conditions use different surface letters per participant
(e.g. O/M vs X/A) for the same two physical arms, so probabilities are matched
by *arm identity* (via each participant's choice<->choice_mapped mapping in
the corresponding input CSV), not by literal token.

For each matched trial this gives two Bernoulli-over-2-arms distributions,
p_full and p_flip. We compute both directions of KL divergence per trial:

    KL(flip || full) = sum_a p_flip(a) * log(p_flip(a) / p_full(a))
    KL(full || flip) = sum_a p_full(a) * log(p_full(a) / p_flip(a))

then average per participant and report mean +/- SEM across participants.

Usage:
    python kl_flip_analysis.py
    python kl_flip_analysis.py --out ./figures/kl_flip
"""

import os
import re
import sys
import ast
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_OUT = os.path.join(SCRIPT_DIR, 'data', 'out')
DATA_IN = os.path.join(SCRIPT_DIR, 'data', 'in')

MODELS = [
    ('llama-70B-adapter', 'Llama', plotting_utils.LLAMA_COLORS),
    ('centaur-70B-adapter', 'Centaur', plotting_utils.CENTAUR_ORANGE),
]

FULL_INPUT_CSV = os.path.join(DATA_IN, 'test_waltmann_data_cleaned.csv')
FLIP_INPUT_CSV = os.path.join(DATA_IN, 'test_waltmann_data_flipped.csv')
FULL_PRED_DIR = os.path.join(DATA_OUT, 'predictive')
FLIP_PRED_DIR = os.path.join(DATA_OUT, 'predictive_flipped')

EPS = 1e-12
FIGSIZE = (14, 6)
BASE_FONT = 22


# ---------------------------------------------------------------------------
# Letter <-> arm mapping (per participant), from the raw input CSVs
# ---------------------------------------------------------------------------

def build_letter_to_arm_map(input_csv):
    """
    {participant_id: {letter: arm}} from the raw trial-level CSV's
    (choice, choice_mapped) pairs -- 'choice' is the physical arm (1 or 2),
    'choice_mapped' is the letter shown in that condition's prompt.
    """
    df = pd.read_csv(input_csv)
    mapping = {}
    for subject, grp in df.groupby('subject'):
        pairs = grp[['choice_mapped', 'choice']].drop_duplicates()
        mapping[int(subject)] = dict(zip(pairs['choice_mapped'], pairs['choice']))
    return mapping


# ---------------------------------------------------------------------------
# Predictive-data loading
# ---------------------------------------------------------------------------

def load_predictive_arm_probs(pred_dir, letter_to_arm):
    """
    Read every participant CSV under pred_dir/singles, parse the 'top2'
    column, and remap letters -> physical arm (1/2) via that participant's
    letter_to_arm mapping. Returns a long DataFrame:
        participant_id, trial_index, p_arm1, p_arm2
    """
    singles = os.path.join(pred_dir, 'singles')
    id_re = re.compile(r'model_(\d+)')
    rows = []

    for fname in sorted(os.listdir(singles)):
        m = id_re.search(fname)
        if not m or not fname.endswith('.csv'):
            continue
        participant_id = int(m.group(1))
        arm_map = letter_to_arm.get(participant_id)
        if not arm_map:
            continue

        df = pd.read_csv(os.path.join(singles, fname))
        for _, row in df.iterrows():
            top2 = ast.literal_eval(row['top2'])
            probs_by_arm = {}
            for letter, prob in top2:
                arm = arm_map.get(letter)
                if arm is not None:
                    probs_by_arm[arm] = prob
            if len(probs_by_arm) != 2:
                continue  # couldn't resolve both arms for this trial, skip it
            rows.append({
                'participant_id': participant_id,
                'trial_index': row['trial_index'],
                'p_arm1': probs_by_arm[1],
                'p_arm2': probs_by_arm[2],
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KL divergence
# ---------------------------------------------------------------------------

def _kl(p1, p2, q1, q2):
    p1, p2 = np.clip([p1, p2], EPS, 1.0)
    q1, q2 = np.clip([q1, q2], EPS, 1.0)
    return p1 * np.log(p1 / q1) + p2 * np.log(p2 / q2)


def compute_kl_table(full_df, flip_df):
    """
    Inner-join full/flip predictive arm-probabilities on
    (participant_id, trial_index) and compute per-trial KL divergence in
    both directions. Returns the merged DataFrame with added columns
    kl_flip_full = KL(flip || full), kl_full_flip = KL(full || flip).
    """
    merged = full_df.merge(flip_df, on=['participant_id', 'trial_index'],
                            suffixes=('_full', '_flip'))
    merged['kl_flip_full'] = _kl(merged['p_arm1_flip'], merged['p_arm2_flip'],
                                  merged['p_arm1_full'], merged['p_arm2_full'])
    merged['kl_full_flip'] = _kl(merged['p_arm1_full'], merged['p_arm2_full'],
                                  merged['p_arm1_flip'], merged['p_arm2_flip'])
    return merged


def per_participant_summary(merged):
    """Per-participant mean KL (both directions), then mean +/- SEM across participants."""
    per_participant = merged.groupby('participant_id')[['kl_flip_full', 'kl_full_flip']].mean()
    return {
        'kl_flip_full_mean': per_participant['kl_flip_full'].mean(),
        'kl_flip_full_sem': per_participant['kl_flip_full'].sem(),
        'kl_full_flip_mean': per_participant['kl_full_flip'].mean(),
        'kl_full_flip_sem': per_participant['kl_full_flip'].sem(),
        'n_participants': len(per_participant),
        'n_trials': len(merged),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

BINS = np.linspace(0, 1, 31)


def _step_xy(values, bins):
    """Histogram `values` into `bins` (density-normalized) and return
    (x, y) polyline coordinates tracing the bar outline -- used instead of
    plt.hist bars so two heavily-overlapping distributions (both piled up
    near 0/1 here) can be told apart as crossing lines plus a shaded gap,
    rather than indistinguishable stacked/translucent bars."""
    counts, edges = np.histogram(values, bins=bins, density=True)
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(counts, 2)
    return x, y


def draw_kl_panel(ax, display_name, colors, merged, stats, fig_width, base_font=BASE_FONT):
    """Draw one model's Full-prompt-vs-Flip panel into `ax`: step-outline
    densities (solid = full, dashed = flip), the gap between them shaded to
    make the divergence visible even where both curves are near-identical,
    and the two KL directions annotated above the axes."""
    dark, light = colors
    x_full, y_full = _step_xy(merged['p_arm1_full'], BINS)
    x_flip, y_flip = _step_xy(merged['p_arm1_flip'], BINS)

    ax.fill_between(x_full, y_full, y_flip, color='#666666', alpha=0.25, linewidth=0,
                     zorder=1, label='|Full − Flip|')
    ax.plot(x_full, y_full, color=dark, linewidth=2.6, label='Full prompt', zorder=3)
    ax.plot(x_flip, y_flip, color=light, linewidth=2.6, linestyle='--', label='Flip', zorder=3)

    ax.set_xlabel('p(arm 1)')
    ax.set_ylabel('Density')
    ax.set_title(display_name)
    ax.legend(frameon=False, loc='upper center')
    plotting_utils.remove_bar_frame(ax)

    kl_text = (f"KL(flip‖full) = {stats['kl_flip_full_mean']:.3f} "
               f"± {stats['kl_flip_full_sem']:.3f}\n"
               f"KL(full‖flip) = {stats['kl_full_flip_mean']:.3f} "
               f"± {stats['kl_full_flip_sem']:.3f}")
    ax.text(0.5, 1.14, kl_text, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=plotting_utils.get_dynamic_fontsize(
                multiplier=0.75, fig_width=fig_width, base_font=base_font),
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#888888', alpha=0.9))


def plot_distributions(results, save_path=None):
    """Combined figure: one panel per model, side by side."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(FIGSIZE[0] / 2 * n, FIGSIZE[1]))
    if n == 1:
        axes = [axes]
    fig_width = fig.get_size_inches()[0]
    plotting_utils.set_dynamic_fontsize(fig_width=fig_width, base_font=BASE_FONT)

    for ax, (display_name, colors, merged, stats) in zip(axes, results):
        draw_kl_panel(ax, display_name, colors, merged, stats, fig_width)

    fig.suptitle('Predictive distribution: Full prompt vs. Flip (reward-inverted context)',
                 y=1.05)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


def plot_single_distribution(display_name, colors, merged, stats, save_path=None,
                              figsize=(7.5, 6.5)):
    """Standalone single-model figure, same panel content as plot_distributions."""
    fig, ax = plt.subplots(figsize=figsize)
    plotting_utils.set_dynamic_fontsize(fig_width=figsize[0], base_font=BASE_FONT)
    draw_kl_panel(ax, display_name, colors, merged, stats, figsize[0])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading letter<->arm maps from:\n  {FULL_INPUT_CSV}\n  {FLIP_INPUT_CSV}")
    full_letter_map = build_letter_to_arm_map(FULL_INPUT_CSV)
    flip_letter_map = build_letter_to_arm_map(FLIP_INPUT_CSV)

    results = []
    summary_rows = []
    for model_dir, display_name, colors in MODELS:
        print(f"\n=== {display_name} ===")
        full_df = load_predictive_arm_probs(os.path.join(FULL_PRED_DIR, model_dir), full_letter_map)
        flip_df = load_predictive_arm_probs(os.path.join(FLIP_PRED_DIR, model_dir), flip_letter_map)
        print(f"  full: {len(full_df)} rows, flip: {len(flip_df)} rows")

        merged = compute_kl_table(full_df, flip_df)
        stats = per_participant_summary(merged)
        print(f"  matched trials: {stats['n_trials']} across {stats['n_participants']} participants")
        print(f"  KL(flip||full) = {stats['kl_flip_full_mean']:.4f} +/- {stats['kl_flip_full_sem']:.4f}")
        print(f"  KL(full||flip) = {stats['kl_full_flip_mean']:.4f} +/- {stats['kl_full_flip_sem']:.4f}")

        results.append((display_name, colors, merged, stats))
        summary_rows.append({'model': display_name, **stats})

        merged.to_csv(os.path.join(out_dir, f'kl_flip_full_trials_{model_dir}.csv'), index=False)

    summary_df = pd.DataFrame(summary_rows).set_index('model')
    summary_path = os.path.join(out_dir, 'kl_flip_full_summary.csv')
    summary_df.to_csv(summary_path)
    print(f"\nSaved: {summary_path}")

    plot_distributions(results, save_path=os.path.join(out_dir, 'kl_flip_full_distributions.png'))

    for (display_name, colors, merged, stats), (model_dir, _, _) in zip(results, MODELS):
        plot_single_distribution(
            display_name, colors, merged, stats,
            save_path=os.path.join(out_dir, f'kl_flip_full_distribution_{model_dir}.png'))

    plt.close('all')
    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(SCRIPT_DIR, 'figures', 'kl_flip'),
                        help='Output directory for plots and CSVs')
    args = parser.parse_args()
    run(args.out)
