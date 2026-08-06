import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

sys.path.insert(0, os.path.dirname(__file__))
import plotting_utils

PARTICIPANTS_TO_PLOT = None  # None means plot all; set to a list of IDs to restrict, e.g. [1, 2, 3]

# Number of discrete choices per task, used for the "random guessing" reference
# line (chance_nll = -log(1/num_choices)). Was previously hardcoded separately
# in each task's own copy of this script.
TASK_NUM_CHOICES = {
    'rl': 2,
    'horizon': 2,
    'spatially_correlated': 30,
    'multi_cue_judgment': 9,
    'rl_waltmann': 2,
}

FIGSIZE=(8,6)
BASE_FONT=24
def read_data_from_folder(folder_path):
    dfs = pd.DataFrame()
    # join without a leading slash — a leading '/' makes os.path.join return '/singles'
    folder_path = os.path.join(folder_path, 'singles')
    if not os.path.isdir(folder_path):
        print(f"Warning: singles folder not found at: {folder_path}")
        return dfs
    file_count = 0  # counter for loaded files
    # Regex to extract the number after "participant_" or "model_".
    # Be flexible about the file extension/casing and match the digits part only.
    participant_id_regex = re.compile(r'(?:model|participant)_(\d+)')

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.csv') and participant_id_regex.search(filename.lower()):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Extract model_id from filename using regex (match against lowercased name)
            match = participant_id_regex.search(filename.lower())
            if match:
                model_id = int(match.group(1)) # Convert the captured digits to an integer
                df['model_id'] = model_id # Add the model_id column
            else:
                # Handle cases where the filename doesn't match the expected format
                print(f"Warning: Could not extract model_id from filename: {repr(filename)}")
                df['model_id'] = None # Or some other indicator of missing ID

            dfs = pd.concat([dfs, df], ignore_index=True)
            file_count += 1  # increment counter

    print(f"{file_count} CSV file(s) loaded.")
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
    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)

        if os.path.isdir(model_path):
            # Read data from the model folder
            df = read_data_from_folder(model_path)
            #if model name has - in its name make it underscore
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
        # Treat RW as part of the domain-specific family and append model to that family
        if fam == 'rw':
            fam = 'domain-specific'
            size_label = 'RW'
        m = size_regex.search(model_name)
        # Only set size_label from regex if not already assigned (e.g., RW)
        if size_label is None:
            size_label = f"{m.group(1)}B" if m else None
        family_mapping.setdefault(fam, []).append((model_name, df, size_label))
    return family_mapping


def calculate_negative_log_likelihood_stats(df, column_name='nll'):
    """
    Calculates the mean and standard error of the mean (SEM)
    for the specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a
                           'log_likelihood' column.

    Returns:
        tuple: A tuple containing the mean negative log-likelihood and the SEM.
    """
    # ensure column exists
    if column_name not in df.columns:
        return None, None

    vals = df[column_name].dropna()
    if vals.empty:
        return None, None

    # If an id column exists, compute per-id means first, then return mean and SEM of those per-id means
    id_col = None
    for c in ('model_id', 'participant_id'):
        if c in df.columns:
            id_col = c
            break

    if id_col is not None:
        grouped = df.groupby(id_col)[column_name].mean().dropna()
        if grouped.empty:
            return None, None
        mean = grouped.mean()
        sem = grouped.std(ddof=1) / np.sqrt(len(grouped)) if len(grouped) > 1 else grouped.sem()
        return float(mean), float(sem)
    else:
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float(vals.sem())
        return mean, sem


def plot_loglikelihood_bars_dynamic(family_mapping=None, figsize=FIGSIZE, nll_column='nll', num_choices=2,
                                     ax=None, show_xticklabels=False, show_bars=True):
    """Plot mean NLL by family.

    family_mapping: dict family -> list of (model_name, df) OR list of (mean, sem)
    num_choices: number of discrete choices in the task, used for the
        "random guessing" reference line (chance_nll = -log(1/num_choices)).
    ax: existing Axes to draw into (e.g. one panel of a larger composite figure).
        When None (default), a standalone figure is created as before. When
        provided, global font-size/layout calls that assume this is the whole
        figure (set_dynamic_fontsize, tight_layout) are skipped so embedding
        this panel doesn't disturb the rest of the composite.
    show_xticklabels: when True, label each bar with its size/condition tag
        (e.g. 'Baseline'/'Partial') and each family group with its name (e.g.
        'Centaur'/'Llama') instead of leaving the x-axis unlabeled. Off by
        default to preserve the existing look of the 4-panel composite and
        other standalone callers.
    show_bars: when False, skip drawing the bars (and their value annotations)
        while keeping the chance-guessing line, axes, and labels. On by default.
    """
    standalone = ax is None
    if standalone:
        plotting_utils.set_dynamic_fontsize(fig_width=figsize[0], base_font=BASE_FONT)
    # If caller didn't provide mapping, build from globals
    if family_mapping is None:
        family_mapping = identify_model_families({k: v for k, v in globals().items() if k.endswith('_df')})

    # Normalize to numeric triplets (mean, sem, size)
    numeric_family = {}
    for family, models in family_mapping.items():
        numeric_models = []
        for item in models:
            # support (model_name, df) or (model_name, df, size)
            if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], pd.DataFrame):
                if len(item) >= 3:
                    model_name, df, size = item[0], item[1], item[2]
                else:
                    model_name, df, size = item[0], item[1], None
                mean, sem = calculate_negative_log_likelihood_stats(df, column_name=nll_column)
                if mean is None:
                    print(f"Warning: No valid NLL values for model '{model_name}'. It will be skipped in the plot.")
                    continue
                numeric_models.append((mean, sem, size))
            else:
                # support numeric triplet (mean, sem, size)
                if isinstance(item, tuple) and len(item) >= 3:
                    try:
                        mean = float(item[0])
                        sem = float(item[1])
                        size = item[2]
                        numeric_models.append((mean, sem, size))
                        continue
                    except Exception:
                        pass
                try:
                    mean, sem = float(item[0]), float(item[1])
                    numeric_models.append((mean, sem, None))
                except Exception:
                    print(f"Warning: Unrecognized model entry for family '{family}': {item}")
        if numeric_models:
            numeric_family[family] = numeric_models

    if not numeric_family:
        print("No valid family statistics found to plot.")
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Adopt the requested plot formatting (grouped variants with family labels)
    # Build flattened lists of means, sems, labels and family slices
    means, errs, used_labels = [], [], []
    family_slices = []
    current_idx = 0
    for family, models in numeric_family.items():
        # models are list of (mean, sem, size) tuples
        if len(models) == 2:
            m1, s1, size1 = models[0][0], models[0][1], models[0][2]
            m2, s2, size2 = models[1][0], models[1][1], models[1][2]
            means += [m1, m2]
            errs += [s1, s2]
            # Primary tick: show size (e.g., '70B' or '8B') or a fallback short label
            lbl1 = size1 if size1 else f"{family.capitalize()} A"
            lbl2 = size2 if size2 else f"{family.capitalize()} B"
            used_labels += [lbl1, lbl2]
            family_slices.append((current_idx, current_idx + 1))
            current_idx += 2
        else:
            # single-model family
            m, s, size = models[0][0], models[0][1], models[0][2]
            print(f"Adding single-model family '{family}' with mean={m}, sem={s}, size_label={size}")
            means.append(m)
            errs.append(s)
            used_labels.append(size if size else family.capitalize())
            family_slices.append(current_idx)
            current_idx += 1

    # If nothing collected, return empty figure
    if not means:
        if standalone:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Keep the same positioning/formatting as provided
    w = 0.5
    gap_in = 0
    gap_out = 0.1

    xpos = [0]
    for i in range(1, len(means)):
        new_family = False
        for fslice in family_slices:
            if isinstance(fslice, tuple):
                if i == fslice[0]:
                    new_family = True
                    break
            elif i == fslice:
                new_family = True
                break
        if new_family:
            xpos.append(xpos[-1] + w + gap_out)
        else:
            xpos.append(xpos[-1] + w + gap_in)
    xpos = np.array(xpos)

    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # Use color scheme from plotting_utils.define_colors_for_families
    family_colors = plotting_utils.define_colors_for_families(numeric_family)
    # build color list matching means length. Each family's color list is
    # dark/light shades meant to distinguish its models by position (e.g. two
    # model sizes, or a baseline/partial condition pair) -- cycle through it by
    # index rather than repeating the same (dark) shade for every model in the
    # family.
    colors = []
    for family, models in numeric_family.items():
        c = family_colors.get(family)
        if isinstance(c, (list, tuple)):
            colors += [c[i % len(c)] for i in range(len(models))]
        else:
            colors += [c] * len(models)
    if len(colors) < len(means):
        colors += ['#CC79A7'] * (len(means) - len(colors))

    if show_bars:
        bars = ax.bar(xpos, means, w, yerr=errs,
                      color=colors[:len(means)])

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    chance_nll = -np.log(1 / num_choices)
    ax.axhline(chance_nll, ls='--', c='grey', lw=1.2, alpha=0.45)
    # Placed just outside the axes' right spine (x=1.02 in axes-fraction, blended
    # with data-coordinate y) so it never overlaps the bars, instead of a
    # magic-number position inside the plotting grid. clip_on=False is required
    # here -- the label is deliberately outside the axes' data extent, so
    # clip_on=True would clip it to invisibility. That does mean it can bleed
    # into a neighboring panel if this axes is narrow/embedded in a composite
    # figure; revisit with a fixed offset in points (constrained by the
    # figure's own bbox) if that happens.
    # va='center' straddles the chance line itself. (Previously va='top' which
    # sat entirely below the line; va='bottom' would grow upward and had
    # nowhere to go when the line sat near the autoscaled top of the axis --
    # not a concern now that the label lives outside the axes' data extent.)
    #rotate text to 90 degrees to avoid overlapping with the bars
    ax.text(1.01, chance_nll, 'Random\nguessing',
            va='center', ha='left', alpha=0.45, rotation=90, clip_on=False,
            transform=ax.get_yaxis_transform())

    ax.set_xticks(xpos)
    if show_xticklabels:
        ax.set_xticklabels(used_labels, ha='center')
    else:
        ax.set_xticklabels([])
    # family centers for secondary axis
    family_centers = [
        (xpos[fslice[0]] + xpos[fslice[1]]) / 2 if isinstance(fslice, tuple) else xpos[fslice]
        for fslice in family_slices
    ]

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(family_centers)
    if show_xticklabels:
        ax2.set_xticklabels([f.capitalize() for f in numeric_family.keys()])
    else:
        ax2.set_xticklabels([])
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.set_label_position('bottom')
    # figsize is a plain default parameter, never overridden by embedded callers, so this
    # would otherwise always evaluate to the standalone-sized pad (24pt) even when ax_a is
    # a small embedded subplot -- inflating this axes' tight bbox disproportionately.
    # The embedded case must still clear the *actual* rendered height of the primary
    # tick labels (used_labels, e.g. '70B') below it -- a flat 8pt was smaller than
    # that text's own line height, so the two label rows overlapped into each other
    # (e.g. 'Centaur' + '70B' rendering as 'CentBur'). Scale the pad off the real
    # tick fontsize instead of a constant.
    primary_tick_fontsize = ax.xaxis.get_majorticklabels()[0].get_fontsize() if ax.xaxis.get_majorticklabels() else plt.rcParams['xtick.labelsize']
    # used_labels can be multi-line (e.g. 'Full\nprompt'); the base pad above was
    # only ever tuned for a single line, so a 2-line label's second line pushed
    # up into the family label below it (e.g. 'Full prompt' colliding with
    # 'Centaur'). Add extra pad per extra line, scaled off the tick fontsize.
    max_label_lines = max((lbl.count('\n') + 1.5 for lbl in used_labels), default=1)
    base_pad = figsize[1] * 4 if standalone else primary_tick_fontsize * 1.6
    pad_value = base_pad + (max_label_lines - 1) * primary_tick_fontsize * 1.4
    ax2.tick_params(axis='x', pad=pad_value, length=0)

    for i in range(len(family_slices) - 1):
        fslice = family_slices[i]
        if isinstance(fslice, tuple):
            _, i1 = fslice
            divider_pos = xpos[i1] + w / 2 + gap_out / 2
        else:
            divider_pos = xpos[fslice] + w / 2 + gap_out / 2

    ax.set_ylabel('Negative log-likelihood (NLL)', labelpad=(20 if standalone else 6),
               fontsize=plotting_utils.get_dynamic_fontsize(multiplier=1.3, fig_width=figsize[0], base_font=BASE_FONT))
    #ax.yaxis.set_label_coords(-0.12, 0.4)
    ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
    plotting_utils.remove_bar_frame(ax)
    #plotting_utils.style_y_gridlines(ax)
    ax.margins(x=0.04)
    #ax.set_ylim(0, 0.75)
    if standalone:
        plt.tight_layout()
    return fig


def build_family_mapping(base_path='predictive', nll_column='log_likelihood', include_participants=None):
    """
    Load every model folder under base_path, keep only models that have
    nll_column, fold in RW's separate JSON metrics (if present), optionally
    restrict to a fixed set of participant/model ids, and return the resulting
    family_mapping -- the same structure plot_loglikelihood_bars_dynamic expects.

    Factored out of load_and_plot so other callers (e.g. a composite figure
    assembling multiple panels) can get the data without also triggering a
    standalone save-to-disk plot.
    """
    model_dfs = {}
    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue
        df = read_data_from_folder(model_path)
        # check whether nll_column exists
        if nll_column not in df.columns:
            print(f"Warning: Column '{nll_column}' not found in {model_name}. Skipping this model.")
            continue
        model_dfs[model_name] = df

    family_mapping = identify_model_families(model_dfs)
    # If RW has separate JSON metrics, load average_log_loss_per_trial and include it
    rw_json_path = os.path.join(base_path, 'rw', 'rw_model_metrics.json')
    if os.path.exists(rw_json_path):
        try:
            with open(rw_json_path, 'r') as fh:
                j = json.load(fh)
            mean = j.get('average_log_loss_per_trial')
            if mean is not None:
                # We have mean per trial and n=6 participants; no per-participant SD available,
                # so set SEM to 0.0 (or provide an estimate externally if preferred).
                sem = 0.0
                # Append as a numeric triplet (mean, sem, size_label) so the plotting
                # normalization can preserve the 'RW' size label.
                family_mapping.setdefault('domain-specific', []).append((float(mean), float(sem), 'RW'))
                print(f"Loaded RW metrics from {rw_json_path}: mean={mean}, sem={sem}, size_label=RW")
        except Exception as e:
            print(f"Warning: failed to load RW metrics from {rw_json_path}: {e}")
    # Keep only specified participants from all models before plotting
    if include_participants is not None:
        include_set = set(include_participants)
        for model_name in list(model_dfs.keys()):
            df = model_dfs[model_name]
            id_col = next((c for c in ('model_id', 'participant_id') if c in df.columns), None)
            if id_col is not None:
                before = df[id_col].nunique()
                model_dfs[model_name] = df[df[id_col].isin(include_set)]
                after = model_dfs[model_name][id_col].nunique()
                if before != after:
                    print(f"Kept participants {include_set & set(df[id_col].unique())} from {model_name} ({before} -> {after})")
        # Rebuild family_mapping after filtering
        family_mapping = identify_model_families(model_dfs)

    return family_mapping


def load_and_plot(base_path='predictive', out_png='loglikelihood_bars.png', nll_column='log_likelihood',
                   include_participants=None, num_choices=2, show_bars=True):
    family_mapping = build_family_mapping(base_path, nll_column=nll_column,
                                          include_participants=include_participants)

    fig = plot_loglikelihood_bars_dynamic(family_mapping=family_mapping, nll_column=nll_column,
                                           figsize=FIGSIZE, num_choices=num_choices, show_bars=show_bars)
    print({k: len(v) for k, v in family_mapping.items()})
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")


def _rl_waltmann_test_participants():
    """rl_waltmann's RW model (rescorla_wagner.py) is fit on 50 train
    participants and evaluated on 6 held-out test participants; read that
    split so this task's bars can be restricted to the same 6, matching what
    RW was actually scored on. Task-specific (not a generic mechanism), since
    no other task currently has this train/test split."""
    split_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'rl_waltmann', 'data', 'out', 'rw', 'train_test_split.json')
    with open(split_path) as f:
        return json.load(f)['test']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--include_participants', type=int, nargs='+', default=PARTICIPANTS_TO_PLOT,
                        help='List of participant IDs to plot (default: None = plot all, '
                             'except rl_waltmann which defaults to its 6 RW test participants)')
    parser.add_argument('--task', choices=sorted(TASK_NUM_CHOICES), required=True,
                        help="Task name; base_path defaults to '<task>/data/out/predictive' "
                             "(resolved relative to this script) and sets --num_choices from a known task")
    parser.add_argument('--num_choices', type=int, default=None,
                        help='Number of discrete choices for the "random guessing" reference line '
                             '(overrides --task if both are given; default 2 if neither is given)')
    parser.add_argument('--out', default=None,
                        help='Output PNG path (default: loglikelihood_bars_<task>.png)')
    parser.add_argument('--no_bars', dest='show_bars', action='store_false',
                        help='Skip drawing the log-likelihood bars (keeps the chance-guessing '
                             'line, axes, and labels)')
    args = parser.parse_args()

    include_participants = args.include_participants
    if args.task == 'rl_waltmann' and include_participants is None:
        include_participants = _rl_waltmann_test_participants()
        print(f"rl_waltmann: restricting to RW's {len(include_participants)} test participants: {include_participants}")

    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.task, 'data', 'out', 'predictive')
    out_png = args.out or f'nll_bars_{args.task}.png'
    num_choices = args.num_choices or TASK_NUM_CHOICES.get(args.task, 2)
    load_and_plot(base_path=base_path, out_png=out_png, nll_column='nll',
                  include_participants=include_participants, num_choices=num_choices,
                  show_bars=args.show_bars)
