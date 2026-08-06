import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils

# Fixed reversal trials for the Waltmann PRLT design (ground truth, from the human
# data's own 'reversal' column, and the value generate_rl.py's generate_timeline()
# intends via reversal_points=(36, 56, 71, 86, 106)).
REVERSAL_POINTS = (36, 56, 71, 86, 106)

# generate_timeline() has a confirmed off-by-one: it uses reversal_points as slice
# boundaries without subtracting 1, so each block actually flips one trial later than
# REVERSAL_POINTS states (verified by re-running its block logic: P(bandit_1 rewarded)
# stays ~0.79 through trial 36 and only drops at trial 37). The already-generated LLM
# CSVs on disk reflect that shifted schedule, so their true first reversal is trial 37.
REVERSAL_POINTS_LLM_GENERATED = tuple(p + 1 for p in REVERSAL_POINTS)
FIGSIZE_STRIP = (10, 6)
#FIGSIZE_WSLS = (8, 6)
FIGSIZE_BARE_SPAGHETTI = (3, 6)
BASE_FONT = 24
def read_data_from_folder(folder_path):
    dfs = pd.DataFrame()

    # Allow pointing directly at a single CSV file (e.g. the combined human/Waltmann
    # CSV, where the containing folder also holds a sibling file like the flipped
    # variant that must NOT be pulled in alongside it).
    if os.path.isfile(folder_path) and folder_path.lower().endswith('.csv'):
        search_path = os.path.dirname(folder_path)
        filenames = [os.path.basename(folder_path)]
    else:
        # Prefer a 'singles' subfolder but fall back to the folder itself (useful for rw CSVs)
        singles_path = os.path.join(folder_path, 'singles')
        if os.path.isdir(singles_path):
            search_path = singles_path
        elif os.path.isdir(folder_path):
            search_path = folder_path
        else:
            print(f"Warning: folder not found at: {folder_path}")
            return dfs
        filenames = os.listdir(search_path)

    file_count = 0  # counter for loaded files
    auto_id = 0
    # Regex to extract the number after "participant_" or "model_".
    participant_id_regex = re.compile(r'(?:model|participant)_(\d+)')

    for filename in filenames:
        if not filename.lower().endswith('.csv'):
            continue
        file_path = os.path.join(search_path, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: failed to read CSV {file_path}: {e}")
            continue

        # Normalize trial column names commonly used across datasets
        if 'trial' in df.columns and 'trial_num' not in df.columns:
            df = df.rename(columns={'trial': 'trial_num'})

        # Human/Waltmann CSVs identify participants via a 'subject' column rather than
        # a 'model_id' column or a per-participant filename. Without this rename, every
        # row in the file would fall back to a single auto-incremented id below, merging
        # all subjects into one bogus rolling-average sequence.
        if 'subject' in df.columns and 'model_id' not in df.columns:
            df = df.rename(columns={'subject': 'model_id'})

        # If DataFrame already contains a model_id column, keep it.
        if 'model_id' in df.columns:
            pass
        else:
            # Try to extract model_id from filename, otherwise assign an auto-incremented id
            match = participant_id_regex.search(filename.lower())
            if match:
                model_id = int(match.group(1))
            else:
                model_id = auto_id
                auto_id += 1
            df['model_id'] = model_id

        dfs = pd.concat([dfs, df], ignore_index=True)
        file_count += 1

    print(f"{file_count} CSV file(s) loaded from {search_path}.")
    return dfs

def load_models(base_path="predictive"):
    """
    Load one CSV per model folder and create separate DataFrame variables
    named <model_name>_df in the global namespace.

    Parameters
    ----------
    base_path : str
        Path to the 'predictive' folder.
    """
    for model_name in os.listdir(base_path):
        model_path = os.path.join(base_path, model_name)

        if os.path.isdir(model_path):
            df = read_data_from_folder(model_path)
            model_name = model_name.replace("-", "_")
            globals()[f"{model_name}_df"] = df
            print(f"Loaded {model_name}_df with shape {df.shape}")



def identify_model_families(model_dfs):
    """
    Identify model families based on model names and return a mapping of family to
    model tuples (model_name, DataFrame, size_label).

    Size detection recognizes patterns like '-70B' or '-8B' (case-insensitive)
    and returns size_label as '70B' or '8B', otherwise None.
    """
    family_mapping = {}
    size_regex = re.compile(r"-(\d{1,3})\s*[bB]\b")
    for model_name, df in model_dfs.items():
        fam = model_name.split('-')[0].lower().replace('-', '_')
        # initialize per-iteration size_label to avoid carry-over between loop iterations
        size_label = None
        # Human data gets its own family bucket rather than a size-based one.
        if fam == 'human':
            size_label = 'Human'
        m = size_regex.search(model_name)
        # Only set size_label from regex if not already assigned (e.g., Human)
        if size_label is None:
            size_label = f"{m.group(1)}B" if m else None
        family_mapping.setdefault(fam, []).append((model_name, df, size_label))
    return family_mapping


def ensure_reversal_column(df, trial_col='trial_num', reversal_points=REVERSAL_POINTS):
    """
    Add a 'reversal' flag (1 on the first trial of a new block, else 0) if the
    dataframe doesn't already have one. Human CSVs carry a real 'reversal' column
    (kept as-is, since it's ground truth); LLM-generated CSVs don't log one, since
    their schedule is fixed rather than recorded per trial, so it's reconstructed
    from `reversal_points` here.
    """
    if 'reversal' in df.columns:
        return df
    df = df.copy()
    df['reversal'] = df[trial_col].isin(reversal_points).astype(int)
    return df


def compute_chose_preferred(df, action_col='choice', trial_col='trial_num', id_col='model_id'):
    """
    For each run (subject/simulation), find their first reversal trial, take the
    action they chose most often strictly before it as their 'preferred' arm, then
    flag every trial where they chose that same arm again. This tracks a subject's
    own baseline preference rather than an arbitrary fixed "bandit 1" label, so it
    stays meaningful across datasets where arm identity/labeling differs.

    Requires a 'reversal' column (see `ensure_reversal_column`).
    """
    df = df.copy()

    first_reversal = (
        df[df['reversal'] == 1]
        .groupby(id_col)[trial_col]
        .min()
        .rename('first_reversal_trial')
    )
    df = df.merge(first_reversal, on=id_col, how='left')

    pre_reversal = df[df[trial_col] < df['first_reversal_trial']]
    preferred_action = (
        pre_reversal
        .groupby(id_col)[action_col]
        .agg(lambda x: x.mode().iloc[0])
        .rename('preferred_action')
    )
    df = df.merge(preferred_action, on=id_col, how='left')

    df['chose_preferred'] = (df[action_col] == df['preferred_action']).astype(int)
    return df


def subsample_to_n_participants(df, n, seed=42, id_col='model_id'):
    """
    Restrict df to a random, unbiased subset of `n` unique `id_col` values.

    Used to match the number of LLM simulated runs to the number of human
    participants they're compared against, so per-model averages aren't
    computed over an unequal (and therefore not directly comparable) sample
    size. If df already has <= n unique ids, it is returned unchanged.
    """
    ids = df[id_col].unique()
    if n >= len(ids):
        return df
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=n, replace=False)
    return df[df[id_col].isin(chosen)].reset_index(drop=True)


def plot_bandit_choice_trends_single_axis(human_df=None, df_rep=None, dfs=None, labels=None, colors=None,
                                          trial_col="trial_num", bandit_avg_col="chose_preferred",
                                          reversal_trials=list(REVERSAL_POINTS), timeline=None,
                                          xlim=(0, 100), margins=True,
                                          shade_post_reversal=True, shade_color='#888888', shade_alpha=0.10,
                                          fig_size=(14, 9), ax=None, ymin=-0.05):
    """
    Plots "chose preferred arm" trends (no reward probability subplot). Overlays human and model predictions.

    Args:
        human_df (pd.DataFrame): Human data with trial-wise bandit choice.
        df_rep (pd.DataFrame): Repetitive baseline model.
        dfs (list of pd.DataFrame): Model prediction DataFrames.
        labels (list of str): Labels for models.
        colors (list of str): Colors for plotting (first is for human).
        trial_col (str): Trial number column.
        bandit_avg_col (str): Column for P(choose bandit 1).
        reversal_trials (list of int): List of reversal trial indices.
        timeline (list of dict): Optional reward schedule for reference markers.
        xlim (tuple): Trial range to plot.
        margins (bool): Whether to plot confidence intervals.
        shade_post_reversal (bool): Shade the area after each reversal in grey,
            alternating on/off block by block (so with a single reversal, just the
            area after it is shaded; with several, blocks alternate grey/white).
        shade_color (str): Color used for the post-reversal shading.
        shade_alpha (float): Alpha for the post-reversal shading.
        ax: existing Axes to draw into (e.g. one panel of a larger composite
            figure). When None (default), a standalone figure is created and a
            separate legend figure is returned, as before. When provided, the
            global set_dynamic_fontsize call, standalone pad/legend, and
            tight_layout are skipped since the composite is expected to
            size/legend itself.
        ymin (float): Lower bound of the y-axis (e.g. 0.2 to start the axis
            partway up instead of at 0). Upper bound stays fixed at 1.05.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size)
        plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=BASE_FONT)
    else:
        fig = ax.figure

    # --- Post-reversal shading (drawn first, behind everything else) ---
    if shade_post_reversal and reversal_trials:
        boundaries = sorted(reversal_trials) + [xlim[1]]
        shade = True  # shade the block right after the first reversal
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if shade:
                ax.axvspan(start, end, color=shade_color, alpha=shade_alpha, zorder=0)
            shade = not shade

    # --- Human Data ---
    if human_df is not None:
        stats = plotting_utils.aggregate_trend(human_df, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        human_color = colors[0]

        ax.plot(mean.index, mean.values, color=human_color, linewidth=0.5, label='Human', alpha=1)
        if margins:
            ax.fill_between(mean.index, mean - sem, mean + sem, color=human_color, alpha=0.2)

    # --- Repetitive Baseline ---
    if df_rep is not None:
        stats = plotting_utils.aggregate_trend(df_rep, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        rep_color = colors[-1]

        ax.plot(mean.index, mean.values, color=rep_color, linewidth=1.5, label=labels[-1], alpha=1)
        if margins:
            ax.fill_between(mean.index, mean - sem, mean + sem, color=rep_color, alpha=0.3, hatch='/')

    # --- Model Data ---
    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]  # skip human
        for i, (df, label) in enumerate(zip(dfs, labels)):
            stats = plotting_utils.aggregate_trend(df, trial_col, bandit_avg_col)
            mean, sem = stats.mean, stats.error
            model_color = model_colors[i % len(model_colors)]

            ax.plot(mean.index, mean.values, label=label, color=model_color, linewidth=1.5, alpha=1)
            if margins:
                ax.fill_between(mean.index, mean - sem, mean + sem, color=model_color, alpha=0.4)

    # --- Reversal Markers ---
    for reversal in reversal_trials:
        #add text label for reversal, centered on the line; line drawn on top so it
        #visibly crosses over the text
        ax.text(reversal, 0.15, 'reversal', rotation=0, va='center', ha='center', alpha=0.7,
                 transform=ax.get_xaxis_transform(), zorder=4)
        ax.axvline(x=reversal, color='black', linestyle='--', linewidth=1.0, alpha=0.3, zorder=5)

    # --- Optional Ground Truth Rewards ---
    if timeline is not None:
        bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
        ax.plot(range(1, len(bandit_1_rewards)+1), bandit_1_rewards,
                color='gray', marker='|', linestyle='', markersize=10, alpha=0.6,
                label='Ground Truth Rewards')
    # --- Formatting ---
    # Scale labelpad off the actual figure's width (not fig_size, which is ignored
    # once ax is passed in by a caller) so labels tuned for a large standalone
    # figure don't look oversized on a small embedded composite panel.
    fig_width = fig.get_size_inches()[0]
    ax.set_xlabel("Trial", labelpad=plotting_utils.get_dynamic_labelpad(fig_width, base_pad=10))
    ax.set_ylabel("Choice rate\n(original high-value arm)",
                  labelpad=plotting_utils.get_dynamic_labelpad(fig_width, base_pad=20))
    ax.set_xlim(xlim)
    ax.set_ylim(ymin, 1.05)
    plotting_utils.style_y_gridlines(ax)
    ax.margins(x=0.04)

    if standalone:
        # --- Legend saved separately ---
        handles, labels_legend = ax.get_legend_handles_labels()
        legend_fig = plt.figure(figsize=(6, 2))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        legend_ax.legend(handles, labels_legend, loc='center', frameon=False)
        plt.tight_layout()
    else:
        legend_fig = None

    return fig, legend_fig


def plot_bandit_choice_trends_with_inset(human_df=None, df_rep=None, dfs=None, labels=None, colors=None,
                                          trial_col="trial_num", bandit_avg_col="chose_preferred",
                                          reversal_trials=list(REVERSAL_POINTS), timeline=None,
                                          xlim=(0, 100), margins=True,
                                          zoom_window=(55, 100),
                                          fig_size=(16, 7)):
    """
    Same as plot_bandit_choice_trends_single_axis but adds a compact floating inset panel
    to the right showing individual runs (spaghetti) for the post-reversal zoom_window.
    A dashed rectangle on the main plot marks the zoomed region; dashed lines link it to the inset.
    """
    fig = plt.figure(figsize=fig_size)
    # Main axes occupies the left ~62% of the figure; inset is placed manually via add_axes
    ax_main = fig.add_axes([0.07, 0.14, 0.58, 0.78])

    plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=BASE_FONT)

    y_lo, y_hi = -0.05, 1.05
    z_lo, z_hi = zoom_window

    # --- Main axis ---
    if human_df is not None:
        stats = plotting_utils.aggregate_trend(human_df, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        human_color = colors[0]
        ax_main.plot(mean.index, mean.values, color=human_color, linewidth=0.5, label='Human', alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=human_color, alpha=0.2)

    if df_rep is not None:
        stats = plotting_utils.aggregate_trend(df_rep, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        rep_color = colors[-1]
        ax_main.plot(mean.index, mean.values, color=rep_color, linewidth=1.5, label=labels[-1], alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=rep_color, alpha=0.3, hatch='/')

    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            stats = plotting_utils.aggregate_trend(df, trial_col, bandit_avg_col)
            mean, sem = stats.mean, stats.error
            model_color = model_colors[i % len(model_colors)]
            ax_main.plot(mean.index, mean.values, label=label, color=model_color, linewidth=1.5, alpha=1)
            if margins:
                ax_main.fill_between(mean.index, mean - sem, mean + sem, color=model_color, alpha=0.4)

    for reversal in reversal_trials:
        ax_main.axvline(x=reversal, color='black', linestyle='--', linewidth=1.0, alpha=0.3, zorder=7)
        ax_main.text(reversal, 0.02, 'reversal', rotation=90, va='bottom', ha='left', alpha=0.7, zorder=6)

    if timeline is not None:
        bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
        ax_main.plot(range(1, len(bandit_1_rewards) + 1), bandit_1_rewards,
                     color='gray', marker='|', linestyle='', markersize=10, alpha=0.6,
                     label='Ground Truth Rewards')

    # Zoom-region rectangle: colored + dot-dash so it reads differently from the black reversal lines
    rect = mpatches.FancyBboxPatch(
        (z_lo, y_lo), z_hi - z_lo, y_hi - y_lo,
        boxstyle="square,pad=0", linewidth=1.4,
        edgecolor='#4a7aa7', facecolor='#4a7aa7', alpha=0.06,
        linestyle=(0, (6, 2, 1, 2)), zorder=5
    )
    rect.set_edgecolor('#4a7aa7')
    rect.set_linestyle((0, (6, 2, 1, 2)))
    ax_main.add_patch(rect)

    ax_main.set_xlabel("Trial", labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=10))
    ax_main.set_ylabel("Choice rate (original high-value arm)",
                        labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=20))
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(y_lo, y_hi)
    plotting_utils.style_y_gridlines(ax_main)

    # --- Compact floating inset: [left, bottom, width, height] in figure coordinates ---
    ax_inset = fig.add_axes([0.71, 0.20, 0.26, 0.55])
    ax_inset.set_facecolor('#f5f5f5')
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.8)

    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            model_color = model_colors[i % len(model_colors)]
            df_zoom = df[(df[trial_col] >= z_lo) & (df[trial_col] <= z_hi)]
            for _, run_df in df_zoom.groupby('model_id'):
                ax_inset.plot(run_df[trial_col], run_df[bandit_avg_col],
                              color=model_color, linewidth=0.6, alpha=0.35)
            mean_z = plotting_utils.aggregate_trend(df_zoom, trial_col, bandit_avg_col, compute_error=False).mean
            ax_inset.plot(mean_z.index, mean_z.values, color=model_color, linewidth=2.0, alpha=1)

    ax_inset.set_xlim(z_lo, z_hi)
    ax_inset.set_ylim(y_lo, y_hi)
    ax_inset.set_title(f"trials {z_lo}–{z_hi} (individual runs)", color='#444444', pad=4)
    ax_inset.tick_params(axis='both', length=3, color='#888888')
    plotting_utils.style_y_gridlines(ax_inset)

    # Connection lines match the rectangle color so they read as one visual unit
    for y_anchor in (y_lo, y_hi):
        con = ConnectionPatch(
            xyA=(z_hi, y_anchor), coordsA=ax_main.transData,
            xyB=(z_lo, y_anchor), coordsB=ax_inset.transData,
            arrowstyle="-", color='#4a7aa7', linewidth=0.9, linestyle=(0, (6, 2, 1, 2)),
            zorder=0
        )
        fig.add_artist(con)

    handles, labels_legend = ax_main.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(6, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels_legend, loc='center', frameon=False)

    return fig, legend_fig


def _short_model_label(model_name):
    ll = model_name.lower()
    if ll.startswith('centaur'):
        return 'C'
    if ll.startswith('llama'):
        return 'L'
    if ll.startswith('human'):
        return 'H'
    return model_name[:2].upper()


def plot_bandit_choice_trends_with_endpoint_strip(
        human_df=None, df_rep=None, dfs=None, labels=None, colors=None,
        trial_col="trial_num", bandit_avg_col="chose_preferred",
        reversal_trials=list(REVERSAL_POINTS), timeline=None,
        xlim=(0, 100), margins=True,
        endpoint_trial=None,
        fig_size=FIGSIZE_STRIP):
    """
    Main plot + compact right-margin dot strip showing the per-run endpoint distribution
    at `endpoint_trial` (defaults to the last trial). Each model gets a jittered dot column
    with a mean line. Arrow connects main plot to strip.
    """
    fig = plt.figure(figsize=fig_size)
    left_main, bottom, width_main, height = 0.07, 0.14, 0.61, 0.78
    gap = 0.01  # Adjust your desired gap here (3% of figure width)
    width_strip = 0.10

    ax_main = fig.add_axes([left_main, bottom, width_main, height])
    ax_strip = fig.add_axes([left_main + width_main + gap, bottom, width_strip, height])

    plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=BASE_FONT)

    y_lo, y_hi = -0.05, 1.05

    # --- Main axis ---
    if human_df is not None:
        stats = plotting_utils.aggregate_trend(human_df, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        human_color = colors[0]
        ax_main.plot(mean.index, mean.values, color=human_color, linewidth=0.5, label='Human', alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=human_color, alpha=0.2)

    if df_rep is not None:
        stats = plotting_utils.aggregate_trend(df_rep, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        rep_color = colors[-1]
        ax_main.plot(mean.index, mean.values, color=rep_color, linewidth=1.5, label=labels[-1], alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=rep_color, alpha=0.3, hatch='/')

    # Collect endpoint info: (trial_x, mean_y, strip_col_index, color)
    endpoint_marks = []

    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            stats = plotting_utils.aggregate_trend(df, trial_col, bandit_avg_col)
            mean, sem = stats.mean, stats.error
            model_color = model_colors[i % len(model_colors)]
            ax_main.plot(mean.index, mean.values, label=label, color=model_color, linewidth=1.5, alpha=1)
            if margins:
                ax_main.fill_between(mean.index, mean - sem, mean + sem, color=model_color, alpha=0.4)
            t = endpoint_trial if endpoint_trial is not None else int(df[trial_col].max())
            if t in mean.index:
                endpoint_marks.append((t, mean.loc[t], i, model_color))

    # Endpoint circles: white-edged filled dot at trial t on each mean line
    for ex, ey, _, ec in endpoint_marks:
        ax_main.plot(ex, ey, 'o', color=ec, markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=6)

    for reversal in reversal_trials:
        ax_main.axvline(x=reversal, color='black', linestyle='--', linewidth=1.0, alpha=0.3, zorder=8)
        ax_main.text(reversal, 0.02, 'reversal', rotation=0, va='bottom', ha='left', alpha=0.7, zorder=7)

    if timeline is not None:
        bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
        ax_main.plot(range(1, len(bandit_1_rewards) + 1), bandit_1_rewards,
                     color='gray', marker='|', linestyle='', markersize=10, alpha=0.6,
                     label='Ground Truth Rewards')

    ax_main.set_xlabel("Trial", labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=10))
    ax_main.set_ylabel("Choice rate (original high-value arm)",
                        labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=20))
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(y_lo, y_hi)
    plotting_utils.style_y_gridlines(ax_main)

    # --- Dot strip panel ---
    ax_strip.set_facecolor('#f5f5f5')
    for spine in ax_strip.spines.values():
        spine.set_edgecolor('#aaaaaa')
        spine.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    strip_colors = []
    t_display = endpoint_trial

    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            model_color = model_colors[i % len(model_colors)]
            t = endpoint_trial if endpoint_trial is not None else int(df[trial_col].max())
            if t_display is None:
                t_display = t
            run_vals = df[df[trial_col] == t].groupby('model_id')[bandit_avg_col].mean()
            if run_vals.empty:
                strip_colors.append(model_color)
                continue
            jitter = rng.uniform(-0.13, 0.13, len(run_vals))
            ax_strip.scatter(i + jitter, run_vals.values,
                             color=model_color, s=22, alpha=0.75, zorder=3, linewidths=0)
            mean_val = run_vals.mean()
            ax_strip.hlines(mean_val, i - 0.24, i + 0.24,
                            colors=model_color, linewidth=2.2, zorder=4)
            strip_colors.append(model_color)

    short_labels = [_short_model_label(l) for l in labels]
    ax_strip.set_xticks(range(len(labels)))
    #ax_strip.set_xticklabels(short_labels, fontsize=8)
    for tick, col in zip(ax_strip.get_xticklabels(), strip_colors):
        tick.set_color(col)
        tick.set_fontweight('bold')

    ax_strip.set_title(f"Distribution of Choice \n rates at trial {t_display}", color='#444444', pad=4)
    ax_strip.set_xlim(-0.5, len(labels) - 0.5)
    ax_strip.set_ylim(y_lo, y_hi)
    ax_strip.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_strip.tick_params(axis='y', length=3, color='#888888')
    ax_strip.tick_params(axis='x', length=0)
    plotting_utils.style_y_gridlines(ax_strip)

    # Colored connection lines: endpoint circle on main -> mean line in strip, same y-value
    for (ex, ey, strip_col, ec) in endpoint_marks:
        con = ConnectionPatch(
            xyA=(ex, ey), coordsA=ax_main.transData,
            xyB=(strip_col, ey), coordsB=ax_strip.transData,
            arrowstyle="-", color=ec, linewidth=1.0,
            linestyle=(0, (4, 3)), alpha=0.6, zorder=0
        )
        fig.add_artist(con)

    handles, labels_legend = ax_main.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(6, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels_legend, loc='center', frameon=False)

    return fig, legend_fig


def plot_bandit_choice_trends_with_colored_strip(
        human_df=None, df_rep=None, dfs=None, labels=None, colors=None,
        trial_col="trial_num", bandit_avg_col="chose_preferred",
        reversal_trials=list(REVERSAL_POINTS), timeline=None,
        xlim=(0, 100), margins=True,
        endpoint_trial=None,
        fig_size=FIGSIZE_STRIP,
        ax_main=None, ax_strip=None):
    """
    Same as plot_bandit_choice_trends_with_endpoint_strip but each model's column in the
    dot strip gets its own colored frame (light tinted fill + solid colored border) instead
    of a shared grey background. Per-model colored ConnectionPatch lines link the endpoint
    circle on the main plot to the matching strip column.

    ax_main/ax_strip: existing Axes pair to draw into (e.g. two panels of a larger
        composite figure, typically a subgridspec). When both are None (default), a
        standalone figure is created with its own [main, strip] axes as before, and a
        separate legend figure is returned. When provided, the global
        set_dynamic_fontsize call (sized for a standalone figure) is skipped, and no
        legend figure is created since the composite is expected to draw its own
        shared legend.
    """
    standalone = ax_main is None
    if standalone:
        fig = plt.figure(figsize=fig_size)
        left_main, bottom, width_main, height = 0.07, 0.14, 0.61, 0.78
        gap = 0.03  # Adjust your desired gap here (3% of figure width)
        width_strip = 0.2

        ax_main = fig.add_axes([left_main, bottom, width_main, height])
        ax_strip = fig.add_axes([left_main + width_main + gap, bottom, width_strip, height])
        plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=BASE_FONT)
    else:
        if ax_strip is None:
            raise ValueError("ax_strip must be provided alongside ax_main")
        fig = ax_main.figure

    y_lo, y_hi = -0.05, 1.05

    # --- Main axis ---
    if human_df is not None:
        stats = plotting_utils.aggregate_trend(human_df, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        human_color = colors[0]
        ax_main.plot(mean.index, mean.values, color=human_color, linewidth=0.5, label='Human', alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=human_color, alpha=0.2)

    if df_rep is not None:
        stats = plotting_utils.aggregate_trend(df_rep, trial_col, bandit_avg_col)
        mean, sem = stats.mean, stats.error
        rep_color = colors[-1]
        ax_main.plot(mean.index, mean.values, color=rep_color, linewidth=1.5, label=labels[-1], alpha=1)
        if margins:
            ax_main.fill_between(mean.index, mean - sem, mean + sem, color=rep_color, alpha=0.3, hatch='/')

    endpoint_marks = []  # (trial_x, mean_y, strip_col_index, color)
    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            stats = plotting_utils.aggregate_trend(df, trial_col, bandit_avg_col)
            mean, sem = stats.mean, stats.error
            model_color = model_colors[i % len(model_colors)]
            ax_main.plot(mean.index, mean.values, label=label, color=model_color, linewidth=1.5, alpha=1)
            if margins:
                ax_main.fill_between(mean.index, mean - sem, mean + sem, color=model_color, alpha=0.4)
            t = endpoint_trial if endpoint_trial is not None else int(df[trial_col].max())
            if t in mean.index:
                endpoint_marks.append((t, mean.loc[t], i, model_color))

    for ex, ey, _, ec in endpoint_marks:
        ax_main.plot(ex, ey, 'o', color=ec, markersize=8,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=6)

    for reversal in reversal_trials:
        ax_main.axvline(x=reversal, color='black', linestyle='--', linewidth=1.0, alpha=0.45, zorder=7)
        ax_main.text(reversal, 0.01, 'reversal', rotation=0, va='bottom', ha='center', alpha=0.45, zorder=8)

    # Small grey band marking the endpoint trial the strip is drawn from, so the
    # main plot visually flags which trial's distribution the strip shows.
    #if endpoint_trial is not None:
        #ax_main.axvspan(endpoint_trial - 1.5, endpoint_trial + 1.5,color='#888888', alpha=0.15, zorder=0)

    if timeline is not None:
        bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
        ax_main.plot(range(1, len(bandit_1_rewards) + 1), bandit_1_rewards,
                     color='gray', marker='|', linestyle='', markersize=10, alpha=0.6,
                     label='Ground Truth Rewards')

    fig_width = fig.get_size_inches()[0]
    ax_main.set_xlabel("Trial", labelpad=plotting_utils.get_dynamic_labelpad(fig_width, base_pad=10))
    ax_main.set_xlim(xlim)
    ax_main.set_ylim(y_lo, y_hi)
    plotting_utils.style_y_gridlines(ax_main)

    # Same y-axis treatment as the strip boxes (tick values, grey tick/spine
    # styling, boxed lowercase label) so the two panels read as one visual system.
    ax_main.spines['left'].set_color('#aaaaaa')
    ax_main.spines['left'].set_linewidth(0.8)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_main.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
    ax_main.set_ylabel("Choice rate\n(original high-value arm)",
                        labelpad=plotting_utils.get_dynamic_labelpad(fig_width, base_pad=10))

    # --- Dot strip with per-model colored frames ---
    ax_strip.set_facecolor('none')
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    # Left spine stays visible (light grey) so the strip visibly shares the main
    # plot's y-axis scale rather than reading as an unrelated panel.
    #ax_strip.spines['left'].set_visible(Tr)
    #ax_strip.spines['left'].set_color('#aaaaaa')
    #ax_strip.spines['left'].set_linewidth(0.8)

    rng = np.random.default_rng(42)
    strip_colors = []
    t_display = endpoint_trial
    col_h = y_hi - y_lo - 0.02

    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]
        for i, (df, label) in enumerate(zip(dfs, labels)):
            model_color = model_colors[i % len(model_colors)]
            t = endpoint_trial if endpoint_trial is not None else int(df[trial_col].max())
            if t_display is None:
                t_display = t

            # Colored frame: light fill + solid border in model color
            ax_strip.add_patch(mpatches.Rectangle(
                (i - 0.44, y_lo + 0.01), 0.88, col_h,
                facecolor=model_color, alpha=0.07, linewidth=0, zorder=1
            ))
            ax_strip.add_patch(mpatches.Rectangle(
                (i - 0.44, y_lo + 0.01), 0.88, col_h,
                facecolor='none', edgecolor=model_color, linewidth=1.4, zorder=2
            ))

            run_vals = df[df[trial_col] == t].groupby('model_id')[bandit_avg_col].mean()
            if run_vals.empty:
                strip_colors.append(model_color)
                continue
            jitter = rng.uniform(-0.13, 0.13, len(run_vals))
            ax_strip.scatter(i + jitter, run_vals.values,
                             color=model_color, s=22, alpha=0.75, zorder=3, linewidths=0)
            mean_val = run_vals.mean()
            ax_strip.hlines(mean_val, i - 0.24, i + 0.24,
                            colors=model_color, linewidth=2.2, zorder=4)
            strip_colors.append(model_color)

    short_labels = [_short_model_label(l) for l in labels]
    ax_strip.set_xticks(range(len(labels)))
    ax_strip.get_xaxis().set_visible(False)

    #ax_strip.set_xticklabels(short_labels, fontsize=8)
    for tick, col in zip(ax_strip.get_xticklabels(), strip_colors):
        tick.set_color(col)
        tick.set_fontweight('bold')

    # (0.5, 1.02) places text centered horizontally, slightly above the top edge
    ax_strip.text(
        0.5, 1.02, 
        f"Distribution of choice\nrates at trial {t_display}", 
        transform=ax_strip.transAxes,
        ha='center', va='bottom',
        fontweight='normal', color='#444444'
    )
    ax_strip.set_xlim(-0.5, len(labels) - 0.5)
    # Same y-axis scale as the main plot, shown explicitly (rather than hidden)
    # so the strip reads as directly comparable rather than a separately-scaled panel.
    #ax_strip.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    #ax_strip.tick_params(axis='y', labelsize=7, length=3, color='#888888', labelcolor='#666666')
    #ax_strip.set_ylabel("choice rate\n(original high-value arm)", fontsize=8, labelpad=6,bbox=dict(boxstyle='round,pad=0.3', facecolor='white',edgecolor='#aaaaaa', linewidth=0.6))
    ax_strip.get_yaxis().set_visible(False)
    ax_strip.tick_params(axis='x', length=0)
    plotting_utils.style_y_gridlines(ax_strip)

    # Colored connection lines: endpoint circle -> strip column mean level
    for (ex, ey, strip_col, ec) in endpoint_marks:
        con = ConnectionPatch(
            xyA=(ex, ey), coordsA=ax_main.transData,
            xyB=(strip_col, ey), coordsB=ax_strip.transData,
            arrowstyle="-", color=ec, linewidth=1.0,
            linestyle=(0, (4, 3)), alpha=0.6, zorder=0
        )
        fig.add_artist(con)

    if standalone:
        handles, labels_legend = ax_main.get_legend_handles_labels()
        legend_fig = plt.figure(figsize=(6, 2))
        legend_ax = legend_fig.add_subplot(111)
        legend_ax.axis("off")
        legend_ax.legend(handles, labels_legend, loc='center', frameon=False)
    else:
        legend_fig = None

    return fig, legend_fig


def plot_main_with_per_model_spaghetti(
        dfs=None, labels=None, colors=None,
        trial_col="trial_num", bandit_avg_col="chose_preferred",
        reversal_trial=REVERSAL_POINTS[0],
        xlim=(1, 100), zoom_window=(50, 100),
        ci_multiplier=1.96,
        fig_size=(16, 10)):
    """
    Two-row figure:
    - Top: full-task mean +/- 95% CI per model, vertical reversal line, legend embedded below
    - Bottom: one spaghetti panel per model (zoom_window only), individual runs + mean,
              colored border per model, shared x- and y-axes
    Human data is drawn dashed in both rows.
    """
    if not dfs:
        return None

    n = len(dfs)
    model_colors = (colors[1:] if colors else ['#333333'] * n)


    fig = plt.figure(figsize=fig_size)
    plotting_utils.set_dynamic_fontsize(fig_width=fig_size[0], base_font=BASE_FONT)
    gs = fig.add_gridspec(
        2, n,
        height_ratios=[1.1, 1],
        hspace=0.55, wspace=0.10,
        left=0.10, right=0.97, top=0.95, bottom=0.10
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_bots = []
    for j in range(n):
        shared = ax_bots[0] if j > 0 else None
        ax_bots.append(fig.add_subplot(gs[1, j], sharex=shared, sharey=shared))

    # --- Top panel ---
    for i, (df, label) in enumerate(zip(dfs, labels)):
        mc = model_colors[i % len(model_colors)]
        ls = '--' if label.lower().startswith('human') else '-'
        stats = plotting_utils.aggregate_trend(df, trial_col, bandit_avg_col, ci_multiplier=ci_multiplier)
        mean, ci = stats.mean, stats.error
        ax_top.plot(mean.index, mean.values, color=mc, linewidth=1.8, linestyle=ls,
                    label=label, alpha=1)
        ax_top.fill_between(mean.index, mean - ci, mean + ci, color=mc, alpha=0.18)

    annotation_fontsize = plotting_utils.get_dynamic_fontsize(
        multiplier=1 / 3, fig_width=fig_size[0], base_font=BASE_FONT)
    ax_top.axvline(x=reversal_trial, color='#999999', linestyle='--', linewidth=1.0, alpha=0.8, zorder=5)
    ax_top.text(reversal_trial + 0.5, 1.0, 'reversal',
                fontsize=annotation_fontsize, color='#888888', va='top',
                transform=ax_top.get_xaxis_transform(), zorder=4)
    ax_top.annotate(
        'post-reversal zoom ↓',
        xy=(reversal_trial, 0), xycoords=('data', 'axes fraction'),
        xytext=(reversal_trial, -0.20), textcoords=('data', 'axes fraction'),
        ha='center', fontsize=annotation_fontsize, color='#888888',
        arrowprops=dict(arrowstyle='->', color='#bbbbbb', lw=0.8)
    )

    ax_top.set_xlim(xlim)
    ax_top.set_ylim(-0.02, 1.05)
    ax_top.set_ylabel("Choice rate (original high-value arm)",
                       labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=8))
    ax_top.tick_params(labelbottom=False)
    plotting_utils.style_y_gridlines(ax_top)
    ax_top.legend(loc='lower center', bbox_to_anchor=(0.5, -0.18),
                  ncol=n, frameon=False)

    # --- Bottom panels ---
    z_lo, z_hi = zoom_window

    for j, (df, label, ax_bot) in enumerate(zip(dfs, labels, ax_bots)):
        mc = model_colors[j % len(model_colors)]
        ls = '--' if label.lower().startswith('human') else '-'

        df_zoom = df[(df[trial_col] >= z_lo) & (df[trial_col] <= z_hi)]

        for _, run_df in df_zoom.groupby('model_id'):
            ax_bot.plot(run_df[trial_col], run_df[bandit_avg_col],
                        color=mc, linewidth=0.6, alpha=0.35, linestyle=ls)

        stats_z = plotting_utils.aggregate_trend(df_zoom, trial_col, bandit_avg_col, ci_multiplier=ci_multiplier)
        mean_z, ci_z = stats_z.mean, stats_z.error
        ax_bot.fill_between(mean_z.index, mean_z - ci_z, mean_z + ci_z,
                            color=mc, alpha=0.18)
        ax_bot.plot(mean_z.index, mean_z.values, color=mc, linewidth=2.0,
                    linestyle=ls, alpha=1)

        for spine in ax_bot.spines.values():
            spine.set_edgecolor(mc)
            spine.set_linewidth(2.0)

        ax_bot.set_title(label, color=mc,
                         fontsize=plotting_utils.get_dynamic_fontsize(
                             multiplier=0.375, fig_width=fig_size[0], base_font=BASE_FONT),
                         pad=5)
        ax_bot.set_xlim(z_lo, z_hi)
        ax_bot.set_ylim(-0.02, 1.05)
        plotting_utils.style_y_gridlines(ax_bot)

        if j == 0:
            ax_bot.set_ylabel("Choice rate (original high-value arm)",
                               labelpad=plotting_utils.get_dynamic_labelpad(fig_size[0], base_pad=8))
        else:
            ax_bot.tick_params(labelleft=False)

        ax_bot.set_xlabel("Trial Number" if j == n // 2 else "")

    return fig


def plot_bare_spaghetti_per_model(
        dfs=None, labels=None, colors=None,
        trial_col="trial_num", bandit_avg_col="chose_preferred",
        zoom_window=(50, 100),
        reversal_trial=None,
        fig_size=None):
    """
    One bare panel per model: individual runs (dashed) + bold mean (solid), in model
    color. No axis labels, no ticks, no spines, no legend. Each model saved as a
    separate figure; returns a list of (fig, label) pairs.

    reversal_trial: if given, draws a vertical dashed reversal line at that trial
        (only when it falls within zoom_window), so panels whose zoom_window starts
        a few trials before the reversal show where the flip happens.
    """
    if not dfs:
        return []

    n = len(dfs)
    model_colors = (colors[1:] if colors else ['#333333'] * n)
    z_lo, z_hi = zoom_window
    panel_w = fig_size[0] if fig_size else 4
    panel_h = fig_size[1] if fig_size else 4
    plotting_utils.set_dynamic_fontsize(fig_width=panel_w, base_font=BASE_FONT)

    figs = []
    for j, (df, label) in enumerate(zip(dfs, labels)):
        mc = model_colors[j % len(model_colors)]
        fig, ax = plt.subplots(figsize=(panel_w, panel_h))

        df_zoom = df[(df[trial_col] >= z_lo) & (df[trial_col] <= z_hi)]
        for _, run_df in df_zoom.groupby('model_id'):
            ax.plot(run_df[trial_col], run_df[bandit_avg_col],
                    color=mc, linewidth=0.6, alpha=0.35, linestyle='--')

        mean_z = plotting_utils.aggregate_trend(df_zoom, trial_col, bandit_avg_col, compute_error=False).mean
        ax.plot(mean_z.index, mean_z.values, color=mc, linewidth=2.0, alpha=1, linestyle='-')

        # zorder=5 puts the reversal line/label above the spaghetti and mean
        # lines (both default zorder=2), which previously sat on top of the
        # line at zorder=0 and hid it.
        if reversal_trial is not None and z_lo <= reversal_trial <= z_hi:
            ax.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=1.0, alpha=0.3, zorder=5)
            ax.text(reversal_trial, 0.15, 'reversal', rotation=0, va='center', ha='center',
                    alpha=0.7, transform=ax.get_xaxis_transform(), zorder=5)

        ax.set_xlim(z_lo, z_hi)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        figs.append((fig, label))

    return figs


def plot_bare_spaghetti_per_model_ranked(
        dfs=None, labels=None, colors=None,
        trial_col="trial_num", bandit_avg_col="chose_preferred",
        zoom_window=(50, 100),
        reversal_trial=None,
        spaghetti_alpha=0.12,
        n_highlight=4,
        highlight_seed=42,
        fig_size=None,
        ax=None, model_index=None):
    """
    Same panel layout as plot_bare_spaghetti_per_model, but with a draw order
    meant to make individual-run structure more visible:

    1. Every run is drawn first, thin and low-alpha (spaghetti_alpha, ~0.08-0.15),
    in the model's color.
    2. A random subset of `n_highlight` runs is drawn on top of that, each with
       its own linestyle (from a fixed cycle) so the highlighted trajectories
       stay distinguishable from one another.
    3. The group mean is drawn last, bold and solid, so it sits on top of everything.

    No axis labels, no ticks, no spines, no legend.

    reversal_trial: if given, draws a vertical dashed reversal line at that trial
        (only when it falls within zoom_window), so panels whose zoom_window starts
        a few trials before the reversal show where the flip happens.

    ax/model_index: draw a single model's panel into an existing Axes (e.g. one
        small-multiple slot of a composite figure) instead of the default batch
        mode. When ax is None (default), every model in dfs/labels gets its own
        standalone figure and the function returns a list of (fig, label) pairs.
        When ax is given, model_index selects which entry of dfs/labels to draw
        into it, and the function returns that ax's figure instead of a list.
    """
    if not dfs:
        return [] if ax is None else None

    n = len(dfs)
    model_colors = (colors[1:] if colors else ['#333333'] * n)
    z_lo, z_hi = zoom_window
    panel_w = fig_size[0] if fig_size else 4
    panel_h = fig_size[1] if fig_size else 4

    def _draw_one(ax, j, df, label):
        mc = model_colors[j % len(model_colors)]

        df_zoom = df[(df[trial_col] >= z_lo) & (df[trial_col] <= z_hi)]
        run_groups = list(df_zoom.groupby('model_id'))
        run_ids = [run_id for run_id, _ in run_groups]

        # 1) Spaghetti: every run, thin, dashed and low-alpha, in the model
        # color. Drawn first so later layers sit on top of it.
        for run_id, run_df in run_groups:
            ax.plot(run_df[trial_col], run_df[bandit_avg_col],
                    color=mc, linewidth=0.6, linestyle='--',
                    alpha=spaghetti_alpha, zorder=1)

        # 2) Highlighted subset: a few runs replotted on top, dashed, so they
        # stay readable against the low-alpha spaghetti underneath.
        rng = np.random.default_rng(highlight_seed)
        highlight_ids = rng.choice(run_ids, size=min(n_highlight, len(run_ids)), replace=False)
        for run_id in highlight_ids:
            run_df = df_zoom[df_zoom['model_id'] == run_id]
            ax.plot(run_df[trial_col], run_df[bandit_avg_col],
                    color=mc, linestyle='--',
                    linewidth=1.4, alpha=0.9, zorder=2)

        # 3) Mean line: drawn last so it sits on top of everything else.
        mean_z = plotting_utils.aggregate_trend(df_zoom, trial_col, bandit_avg_col, compute_error=False).mean
        ax.plot(mean_z.index, mean_z.values, color=mc, linewidth=2.2, alpha=1,
                linestyle='-', zorder=3)

        # zorder=5/4 puts the reversal line/label above the mean line
        # (zorder=3), which previously sat on top of the line at zorder=0 and
        # hid it.
        if reversal_trial is not None and z_lo <= reversal_trial <= z_hi:
            ax.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=1.0, alpha=0.3, zorder=5)
            ax.text(reversal_trial, 0.15, 'reversal', rotation=0, va='center', ha='center',
                    alpha=0.7, transform=ax.get_xaxis_transform(), zorder=4)

        ax.set_xlim(z_lo, z_hi)
        ax.set_ylim(-0.02, 1.05)

        # Only right (y) and bottom (x) spines are shown; left/top are hidden.
        # Ticks are kept sparse: y at 0/0.5/1, x at the zoom window's start,
        # midpoint, and end.
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        ax.yaxis.tick_right()
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
        plotting_utils.style_y_gridlines(ax)

        x_mid = round((z_lo + z_hi) / 2)
        ax.set_xticks([z_lo, x_mid, z_hi])

    if ax is not None:
        if model_index is None:
            raise ValueError("model_index is required when ax is provided")
        # rcParams are left as the composite caller set them (sized for the
        # composite's own fig_size), rather than being reset to panel_w here.
        _draw_one(ax, model_index, dfs[model_index], labels[model_index])
        return ax.figure

    plotting_utils.set_dynamic_fontsize(fig_width=panel_w, base_font=BASE_FONT)
    figs = []
    for j, (df, label) in enumerate(zip(dfs, labels)):
        fig, ax_j = plt.subplots(figsize=(panel_w, panel_h))
        _draw_one(ax_j, j, df, label)
        # Small margin on all sides so the (now-visible) right/bottom spines
        # and their tick labels sit clear of the figure edge.
        plt.subplots_adjust(left=0.05, right=0.90, top=0.95, bottom=0.12)
        figs.append((fig, label))

    return figs


def prepare_bandit_choice_data(base_path='./data/out/generative',
                               rw_path='./data/in/test_waltmann_data_cleaned.csv',
                               subsample_seed=42, max_trial_to_plot=50):
    """
    Load human + all per-model generative CSVs, compute each run's 'chose_preferred'
    indicator, subsample LLM participants down to the human participant count, and
    assign per-model colors. Returns a dict:
        dfs, labels, colors     -- matching-order lists ready to pass straight into
                                   plot_bandit_choice_trends_* and
                                   plot_bare_spaghetti_per_model_ranked
        max_trial               -- last trial actually present in the (possibly
                                   truncated) data
        in_range_reversals      -- REVERSAL_POINTS that fall within max_trial
        zoom_lo                 -- start of the "post-last-reversal" zoom window
    Returns None if no model data was found under base_path.

    Factored out of bandit_choice_trends() so other callers (e.g. a composite
    figure assembling several panels) can get the same plotting-ready data
    without also saving PNGs.

    See bandit_choice_trends() for parameter docs (base_path/rw_path/
    subsample_seed/max_trial_to_plot mean the same thing here).
    """
    base_path = os.path.abspath(base_path)

    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    # Load human data first (if any) so we know how many participants to subsample
    # each LLM model down to.
    rw_df = None
    n_target = None
    if rw_path is not None:
        rw_path = os.path.abspath(rw_path)
        rw_df = read_data_from_folder(rw_path)
        if not rw_df.empty:
            if 'trial_num' not in rw_df.columns and 'trial' in rw_df.columns:
                rw_df = rw_df.rename(columns={'trial': 'trial_num'})
            # Human CSVs already carry a real 'reversal' column; ensure_reversal_column
            # leaves it untouched.
            rw_df = ensure_reversal_column(rw_df, reversal_points=REVERSAL_POINTS)
            rw_df = compute_chose_preferred(rw_df)
            n_target = rw_df['model_id'].nunique()
            print(f"Loaded human data from {rw_path}: {len(rw_df)} rows, "
                  f"{n_target} participants")
        else:
            print(f"Warning: no human data found at {rw_path}")
            rw_df = None

    # Load all model folders
    model_dfs = {}
    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue
        df = read_data_from_folder(model_path)
        if df.empty:
            continue
        # Normalise trial column: LLM CSVs use 'trial_index', RW CSVs use 'trial'
        if 'trial_num' not in df.columns:
            for src in ('trial_index', 'trial'):
                if src in df.columns:
                    df = df.rename(columns={src: 'trial_num'})
                    break
        if n_target is not None:
            n_before = df['model_id'].nunique()
            df = subsample_to_n_participants(df, n_target, seed=subsample_seed)
            print(f"Subsampled {model_name}: {n_before} -> {df['model_id'].nunique()} "
                  f"participants (seed={subsample_seed})")
        # LLM CSVs don't log a per-trial 'reversal' column, so reconstruct it from the
        # schedule actually baked into these generated runs (REVERSAL_POINTS_LLM_GENERATED,
        # not REVERSAL_POINTS — see the off-by-one note above the constant).
        df = ensure_reversal_column(df, reversal_points=REVERSAL_POINTS_LLM_GENERATED)
        df = compute_chose_preferred(df)
        model_dfs[model_name] = df

    if rw_df is not None:
        model_dfs['human'] = rw_df

    if not model_dfs:
        print(f"No model data loaded from {base_path}")
        return None

    # Restrict every series to the first `max_trial_to_plot` trials. Doing this after
    # compute_chose_preferred is safe: 'preferred_action' only depends on trials strictly
    # before each run's own first reversal (well within any reasonable max_trial_to_plot),
    # so values for trials <= max_trial_to_plot are unaffected by later trials.
    if max_trial_to_plot is not None:
        for model_name, df in model_dfs.items():
            model_dfs[model_name] = df[df['trial_num'] <= max_trial_to_plot].reset_index(drop=True)

    # Identify families and assign colors
    family_mapping = identify_model_families(model_dfs)
    family_colors = plotting_utils.define_colors_for_families(family_mapping)

    model_color_map = {}
    for fam, models in family_mapping.items():
        fam_colors = family_colors.get(fam, ['#000000'])
        for idx, (model_name, df, _) in enumerate(models):
            model_color_map[model_name] = fam_colors[idx % len(fam_colors)]

    # Build plotting lists
    labels = []
    dfs = []
    colors = []

    human_color = '#000000'
    colors.append(human_color)  # placeholder so indexing in plot function stays consistent

    for model_name, df in model_dfs.items():
        labels.append(model_name)
        dfs.append(df)
        colors.append(model_color_map.get(model_name, '#000000'))

    colors.append('#999999')  # placeholder rep_color at tail

    max_trial = int(next(iter(dfs))['trial_num'].max()) if 'trial_num' in next(iter(dfs)).columns else 100

    # Only reversal points that actually fall within the plotted trial range.
    in_range_reversals = [r for r in REVERSAL_POINTS if r <= max_trial]
    # Zoom on the trials following the last reversal inside the plotted range (falls
    # back to a fixed-width window near the end if no reversal is in range).
    zoom_lo = in_range_reversals[-1] if in_range_reversals else max(1, max_trial - 20)

    return dict(dfs=dfs, labels=labels, colors=colors, max_trial=max_trial,
               in_range_reversals=in_range_reversals, zoom_lo=zoom_lo)


def bandit_choice_trends(base_path='./data/out/generative', save_dir='./figures',
                         rw_path='./data/in/test_waltmann_data_cleaned.csv',
                         subsample_seed=42, max_trial_to_plot=50):
    """
    Load all model single-run CSVs from `base_path`, compute each run's "chose preferred
    arm" indicator, group models into families, pick colors, and plot combined trends.

    The plotted metric is per-run: whichever action a subject/simulation chose most often
    before their own first reversal trial is their "preferred" arm; `chose_preferred` flags
    every trial where they choose that same arm again. This tracks each run's own baseline
    preference rather than an arbitrary fixed "bandit 1" label, so results stay meaningful
    even though arm identity/letter-mapping differs across datasets. See
    `compute_chose_preferred`.

    Parameters
    - base_path: str path to the `generative` output folder containing model subfolders.
    - save_dir: directory to save plots.
    - rw_path: path to the human data to overlay: a folder (e.g. '/path/to/rw') or a
               single CSV (default: the Waltmann human test set, data/in/test_waltmann_data_
               cleaned.csv). That data is overlaid as the 'human' family in black/grey,
               and each LLM model is randomly subsampled down to the same number
               of participants as the human data so the comparison isn't skewed by unequal
               sample sizes. Pass None to skip the overlay and subsampling entirely.
    - subsample_seed: RNG seed used when subsampling LLM participants to match the RW/human
                       participant count (default 42, for reproducibility).
    - max_trial_to_plot: only trials up to and including this trial number are plotted
                          (default 50). The Waltmann PRLT design has reversals at
                          REVERSAL_POINTS = (36, 56, 71, 86, 106); with the default of 50
                          trials, only the first reversal (36) falls inside the plotted range.
    """
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    prepared = prepare_bandit_choice_data(base_path=base_path, rw_path=rw_path,
                                          subsample_seed=subsample_seed,
                                          max_trial_to_plot=max_trial_to_plot)
    if prepared is None:
        return None

    dfs = prepared['dfs']
    labels = prepared['labels']
    colors = prepared['colors']
    max_trial = prepared['max_trial']
    in_range_reversals = prepared['in_range_reversals']
    zoom_lo = prepared['zoom_lo']

    # Standard plot
    fig, legend_fig = plot_bandit_choice_trends_single_axis(human_df=None, df_rep=None,
                                                           dfs=dfs, labels=labels, colors=colors,
                                                           trial_col='trial_num', bandit_avg_col='chose_preferred',
                                                           reversal_trials=in_range_reversals,
                                                           xlim=(1, max_trial), margins=True,
                                                           fig_size=(12, 10))
    out_fig = os.path.join(save_dir, f'chose_preferred_arm.png')
    out_legend = os.path.join(save_dir, f'chose_preferred_arm_legend.png')
    fig.savefig(out_fig, dpi=200, bbox_inches='tight')
    legend_fig.savefig(out_legend, dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close(legend_fig)

    # Inset version
    fig_inset, legend_inset = plot_bandit_choice_trends_with_inset(
        human_df=None, df_rep=None,
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        reversal_trials=in_range_reversals,
        xlim=(1, max_trial), margins=True,
        zoom_window=(zoom_lo, max_trial),
        fig_size=(16, 7)
    )
    out_inset = os.path.join(save_dir, f'chose_preferred_arm_inset.png')
    out_inset_legend = os.path.join(save_dir, f'chose_preferred_arm_inset_legend.png')
    fig_inset.savefig(out_inset, dpi=200, bbox_inches='tight')
    legend_inset.savefig(out_inset_legend, dpi=200, bbox_inches='tight')
    plt.close(fig_inset)
    plt.close(legend_inset)

    # Endpoint dot strip version
    fig_strip, legend_strip = plot_bandit_choice_trends_with_endpoint_strip(
        human_df=None, df_rep=None,
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        reversal_trials=in_range_reversals,
        xlim=(1, max_trial), margins=True,
        endpoint_trial=max_trial,
        fig_size=FIGSIZE_STRIP
    )
    out_strip = os.path.join(save_dir, f'chose_preferred_arm_strip.png')
    out_strip_legend = os.path.join(save_dir, f'chose_preferred_arm_strip_legend.png')
    fig_strip.savefig(out_strip, dpi=200, bbox_inches='tight')
    legend_strip.savefig(out_strip_legend, dpi=200, bbox_inches='tight')
    plt.close(fig_strip)
    plt.close(legend_strip)

    # Colored-frame strip version
    fig_cstrip, legend_cstrip = plot_bandit_choice_trends_with_colored_strip(
        human_df=None, df_rep=None,
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        reversal_trials=in_range_reversals,
        xlim=(1, max_trial), margins=True,
        endpoint_trial=max_trial,
        fig_size=(16, 7)
    )
    out_cstrip = os.path.join(save_dir, f'chose_preferred_arm_colored_strip.png')
    out_cstrip_legend = os.path.join(save_dir, f'chose_preferred_arm_colored_strip_legend.png')
    fig_cstrip.savefig(out_cstrip, dpi=200, bbox_inches='tight')
    legend_cstrip.savefig(out_cstrip_legend, dpi=200, bbox_inches='tight')
    plt.close(fig_cstrip)
    plt.close(legend_cstrip)

    print(f"Saved plots: {out_fig}, {out_legend}")
    print(f"Saved inset plots: {out_inset}, {out_inset_legend}")
    print(f"Saved strip plots: {out_strip}, {out_strip_legend}")
    print(f"Saved colored-strip plots: {out_cstrip}, {out_cstrip_legend}")

    # Two-row spaghetti figure
    fig_spaghetti = plot_main_with_per_model_spaghetti(
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        reversal_trial=zoom_lo, xlim=(1, max_trial),
        zoom_window=(zoom_lo, max_trial),
        ci_multiplier=1.96,
        fig_size=(16, 10)
    )
    out_spaghetti = os.path.join(save_dir, f'chose_preferred_arm_spaghetti.png')
    if fig_spaghetti is not None:
        fig_spaghetti.savefig(out_spaghetti, dpi=200, bbox_inches='tight')
        plt.close(fig_spaghetti)
        print(f"Saved spaghetti plot: {out_spaghetti}")

    # Bare per-model spaghetti (no labels, no ticks, no spines). Zoom window starts
    # 5 trials before the reversal (rather than exactly at it) so the reversal line
    # is visible inside the panel instead of sitting on the left edge.
    bare_zoom_lo = max(1, zoom_lo - 5)
    bare_figs = plot_bare_spaghetti_per_model(
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        zoom_window=(bare_zoom_lo, max_trial),
        reversal_trial=zoom_lo,
        fig_size=FIGSIZE_BARE_SPAGHETTI
    )
    out_bare = []
    for fig_bare, model_label in bare_figs:
        safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', model_label)
        path = os.path.join(save_dir, f'chose_preferred_arm_bare_{safe_label}.png')
        fig_bare.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig_bare)
        out_bare.append(path)
        print(f"Saved bare spaghetti: {path}")

    # Bare per-model spaghetti, ranked/layered variant: spaghetti colored by
    # rank drawn first, a highlighted subset with distinct linestyles drawn on
    # top, mean drawn last. Unlike the plain bare version above, this zoom
    # window starts exactly at the reversal trial rather than 5 trials early;
    # the reversal line/label still draw (z_lo == reversal_trial), just at the
    # panel's left edge instead of inset from it.
    bare_ranked_figs = plot_bare_spaghetti_per_model_ranked(
        dfs=dfs, labels=labels, colors=colors,
        trial_col='trial_num', bandit_avg_col='chose_preferred',
        zoom_window=(zoom_lo, max_trial),
        reversal_trial=zoom_lo,
        fig_size=FIGSIZE_BARE_SPAGHETTI
    )
    out_bare_ranked = []
    for fig_bare, model_label in bare_ranked_figs:
        safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', model_label)
        path = os.path.join(save_dir, f'chose_preferred_arm_bare_ranked_{safe_label}.png')
        fig_bare.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig_bare)
        out_bare_ranked.append(path)
        print(f"Saved bare ranked spaghetti: {path}")

    return (out_fig, out_legend, out_inset, out_inset_legend, out_strip, out_strip_legend,
            out_cstrip, out_cstrip_legend, out_spaghetti, out_bare, out_bare_ranked)


def _discover_bare_pngs(save_dir, prefix='chose_preferred_arm_bare_'):
    """
    Find the per-model "bare" spaghetti PNGs saved by plot_bare_spaghetti_per_model
    (excludes the '..._bare_ranked_...' variant). Returns (paths, labels) sorted so
    'human' comes first, then alphabetically.
    """
    pattern = os.path.join(save_dir, f'{prefix}*.png')
    paths = [p for p in glob.glob(pattern) if '_bare_ranked_' not in os.path.basename(p)]

    def sort_key(p):
        name = os.path.basename(p)
        return (0, name) if 'human' in name.lower() else (1, name)

    paths.sort(key=sort_key)
    labels = [os.path.basename(p)[len(prefix):-len('.png')] for p in paths]
    return paths, labels


def _place_image_panel(ax, png_path, panel_label=None, title=None):
    ax.imshow(plt.imread(png_path))
    ax.axis('off')
    fig_width = ax.figure.get_size_inches()[0]
    if title:
        ax.set_title(title, fontsize=plotting_utils.get_dynamic_fontsize(
            multiplier=0.5, fig_width=fig_width, base_font=BASE_FONT), pad=4)
    if panel_label:
        ax.text(-0.02, 1.03, panel_label, transform=ax.transAxes,
                fontsize=plotting_utils.get_dynamic_fontsize(
                    multiplier=1.0, fig_width=fig_width, base_font=BASE_FONT),
                fontweight='bold', va='bottom', ha='right')


def build_summary_figure(save_dir='./figures',
                          predictive_png=None,
                          generative_png=None,
                          wsls_png=None,
                          bare_pngs=None,
                          bare_labels=None,
                          out_path=None,
                          fig_size=(20, 11)):
    """
    Assemble the four-panel rl_waltmann summary figure from PNGs already rendered
    by three separate scripts (does not re-run any analysis itself):

        A - predictive performance: NLL bars (predictive_plots.py, repo root)
        B - generative choice-rate trends w/ colored endpoint strip
            (plot_bandit_choice_trends_with_colored_strip, this script)
        C - win-stay / lose-stay bar plot (transition_analysis.py: plot_wsls)
        D - bare per-model spaghetti panels, one per model
            (plot_bare_spaghetti_per_model, this script)

    Parameters
    ----------
    save_dir : str
        This script's own figures directory (holds the generative + bare PNGs).
    predictive_png, generative_png, wsls_png : str or None
        Explicit paths for panels A/B/C. Defaults:
          A: '<repo_root>/nll_bars_rl_waltmann.png'
          B: '<save_dir>/chose_preferred_arm_colored_strip.png'
          C: '<save_dir>/transitions/transition_wsls.png'
    bare_pngs, bare_labels : list or None
        Explicit paths/labels for panel D. Default: auto-discover every
        'chose_preferred_arm_bare_*.png' in save_dir (human first).
    out_path : str or None
        Where to save the composite. Default: '<save_dir>/summary_figure.png'.
    """
    save_dir = os.path.abspath(save_dir)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    predictive_png = predictive_png or os.path.join(root_dir, 'nll_bars_rl_waltmann.png')
    generative_png = generative_png or os.path.join(save_dir, 'chose_preferred_arm_colored_strip.png')
    wsls_png = wsls_png or os.path.join(save_dir, 'transitions', 'transition_wsls.png')

    if bare_pngs is None:
        bare_pngs, bare_labels = _discover_bare_pngs(save_dir)
    if not bare_pngs:
        raise FileNotFoundError(
            f"No bare per-model PNGs found in {save_dir} "
            f"(expected chose_preferred_arm_bare_*.png from plot_bare_spaghetti_per_model)")

    for p in [predictive_png, generative_png, wsls_png, *bare_pngs]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Composite figure panel image not found: {p}")

    # Bottom row is C (1 column) + one column per bare-model panel in D; top row
    # (A, B) splits the same total column count roughly in half so the two rows
    # line up as one grid rather than two independently-sized rows.
    n_cols = len(bare_pngs) + 1
    half = max(n_cols // 2, 1)

    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(2, n_cols, height_ratios=[1.1, 1], hspace=0.28, wspace=0.06,
                          left=0.02, right=0.99, top=0.95, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, :half])
    ax_b = fig.add_subplot(gs[0, half:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = [fig.add_subplot(gs[1, i + 1]) for i in range(len(bare_pngs))]

    _place_image_panel(ax_a, predictive_png, panel_label='A')
    _place_image_panel(ax_b, generative_png, panel_label='B')
    _place_image_panel(ax_c, wsls_png, panel_label='C')
    for i, (ax, png) in enumerate(zip(ax_d, bare_pngs)):
        title = bare_labels[i] if bare_labels else None
        _place_image_panel(ax, png, panel_label='D' if i == 0 else None, title=title)

    out_path = out_path or os.path.join(save_dir, 'summary_figure.png')
    fig.savefig(out_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved composite summary figure: {out_path}")
    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate "chose preferred arm" trend plots for generative outputs')
    parser.add_argument('--base', default='./data/out/generative', help='Base generative output folder')
    parser.add_argument('--out', default='./figures', help='Output directory for saved figures')
    parser.add_argument('--rw_path', default='./data/in/test_waltmann_data_cleaned.csv',
                        help='Path to human/RW data: either a folder (e.g. ./rw) or a single CSV '
                             '(default: the Waltmann human test set, ./data/in/test_waltmann_data_'
                             'cleaned.csv). This data is overlaid and each LLM model is randomly '
                             'subsampled to match its participant count. Pass an empty string to '
                             'disable the overlay.')
    parser.add_argument('--subsample_seed', type=int, default=42,
                        help='RNG seed for subsampling LLM participants down to the RW/human '
                             'participant count (default: 42)')
    parser.add_argument('--max_trial', type=int, default=50,
                        help='Only plot trials up to and including this trial number (default: 50). '
                             'The Waltmann reversal schedule is REVERSAL_POINTS = (36, 56, 71, 86, 106); '
                             'only reversals within this range are drawn.')
    parser.add_argument('--composite', action='store_true',
                        help='Instead of regenerating the trend plots, assemble the 4-panel summary '
                             'figure (A: predictive NLL, B: generative colored-strip, C: WSLS bars, '
                             'D: per-model bare spaghetti) from PNGs already saved under --out and '
                             'the repo root.')
    parser.add_argument('--predictive_png', default=None,
                        help='Override path to the predictive-performance PNG for panel A '
                             '(default: <repo_root>/nll_bars_rl_waltmann.png)')
    args = parser.parse_args()

    if args.composite:
        build_summary_figure(save_dir=args.out, predictive_png=args.predictive_png)
    else:
        rw_path = args.rw_path if args.rw_path else None
        bandit_choice_trends(base_path=args.base, save_dir=args.out,
                             rw_path=rw_path,
                             subsample_seed=args.subsample_seed, max_trial_to_plot=args.max_trial)
