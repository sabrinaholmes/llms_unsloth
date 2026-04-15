import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

PARTICIPANTS_TO_PLOT = None  # None means plot all; set to a list of IDs to restrict, e.g. [1, 2, 3]
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
    
    for filename in os.listdir(folder_path):
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
    for model_name in os.listdir(base_path):
        model_path = os.path.join(base_path, model_name)
        #print(f"Loading model: {model_name} from {model_path}")
        
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


def define_colors_for_families(family_mapping):
    """
    Define a color mapping for each model family.
    
    Parameters
    ----------
    family_mapping : dict
        A dictionary mapping family names to lists of (model_name, DataFrame) tuples.
    
    Returns
    -------
    dict
        A dictionary mapping family names to colors.
    """
    color_map = {
        'centaur': ['#D55E00','#E69F00'],  # Two shades for Centaur variants
        'llama':  ['#0072B2','#56B4E9'],  # Two shades for LLaMA variants
        'domain-specific': ['#CC79A7','#999999'],  # Purple for domain-specific (RW), gray for any other domain-specific models
    }
    
    family_colors = {family: color_map.get(family, ['#000000']) for family in family_mapping.keys()}
    return family_colors


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
    

def set_dynamic_fontsize(fig_width=12, base_font=20):
    scale = fig_width / 6  # 6 is your baseline width, adjust as needed
    plt.rcParams.update({
        'font.size': base_font * scale * 0.65,
        'axes.titlesize': base_font * scale * 1.2,
        'axes.labelsize': base_font * scale * 0.9,
        'xtick.labelsize': base_font * scale * 0.9,
        'ytick.labelsize': base_font * scale * 0.9,
        'legend.fontsize': base_font * scale,
    })

def plot_loglikelihood_bars_dynamic(family_mapping=None, figsize=(12, 6), nll_column='nll'):
    """Plot mean NLL by family.

    family_mapping: dict family -> list of (model_name, df) OR list of (mean, sem)
    """
    set_dynamic_fontsize(fig_width=figsize[0], base_font=20)
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
        fig, ax = plt.subplots(figsize=figsize)
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
        fig, ax = plt.subplots(figsize=figsize)
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

    fig, ax = plt.subplots(figsize=figsize)
    # Use color scheme from define_colors_for_families if possible
    family_colors = define_colors_for_families(numeric_family)
    # build color list matching means length
    colors = []
    for family, models in numeric_family.items():
        c = family_colors.get(family)
        if isinstance(c, (list, tuple)):
            colors += [c[0]] * len(models)
        else:
            colors += [c] * len(models)
    if len(colors) < len(means):
        colors += ['#CC79A7'] * (len(means) - len(colors))

    bars = ax.bar(xpos, means, w, yerr=errs,
                  color=colors[:len(means)], edgecolor='black', linewidth=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    chance_nll = -np.log(1/9)
    ax.axhline(chance_nll, ls='--', c='grey', lw=1.2)
    adjustment_term = figsize[0] * 0.04
    ax.text(xpos[-1] - adjustment_term, chance_nll, 'Random guessing',
            va='bottom', ha='left')

    ax.set_xticks(xpos)
    ax.set_xticklabels(used_labels, ha='center')

    # family centers for secondary axis
    family_centers = [
        (xpos[fslice[0]] + xpos[fslice[1]]) / 2 if isinstance(fslice, tuple) else xpos[fslice]
        for fslice in family_slices
    ]

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(family_centers)
    ax2.set_xticklabels([f.capitalize() for f in numeric_family.keys()])
    ax2.spines['bottom'].set_visible(False)
    ax2.xaxis.set_label_position('bottom')
    pad_value = figsize[1] * 4
    ax2.tick_params(axis='x', pad=pad_value, length=0)

    for i in range(len(family_slices) - 1):
        fslice = family_slices[i]
        if isinstance(fslice, tuple):
            _, i1 = fslice
            divider_pos = xpos[i1] + w / 2 + gap_out / 2
        else:
            divider_pos = xpos[fslice] + w / 2 + gap_out / 2
        ax.axvline(divider_pos, color='grey', lw=1)

    ax.set_ylabel('Negative Log-Likelihood (NLL)', labelpad=20)
    ax.yaxis.set_label_coords(-0.12, 0.4)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(True)
    ax.grid(False)
    ax.margins(x=0.04)
    #ax.set_ylim(0, 0.75)
    plt.tight_layout()
    return fig


def load_and_plot(base_path='predictive', out_png='loglikelihood_bars.png', nll_column='log_likelihood', include_participants=None):
    # Load all models in base_path
    model_dfs = {}
    for model_name in os.listdir(base_path):
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

    fig = plot_loglikelihood_bars_dynamic(family_mapping=family_mapping, nll_column=nll_column, figsize=(12,10))
    print({k: len(v) for k, v in family_mapping.items()})
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")


def plot_nll_by_participant(
    underscore_path,
    lowercase_path,
    participant_ids,
    nll_column='nll',
    figsize=(10, 5),
    out_png=None,
):
    """
    One figure per model family (centaur, llama).
    X-axis: participants. Each participant has 2 grouped bars:
      - lowercase condition: solid family color
      - underscore condition: same family color with /// hatch

    Parameters
    ----------
    underscore_path : str
        Base folder for the underscore condition (one subfolder per model).
    lowercase_path : str
        Base folder for the lowercase condition (one subfolder per model).
    participant_ids : sequence of int
        Participant IDs to include (matched against 'model_id' or 'participant_id').
    nll_column : str
        NLL column name in the CSVs.
    figsize : tuple
        Size of each per-family figure.
    out_png : str or None
        If given, used as a path prefix: saves '<stem>_<family><ext>' per family.
        E.g. 'figures/by_participant.png' → 'figures/by_participant_centaur.png'.

    Returns
    -------
    dict
        Mapping of family name -> matplotlib Figure.
    """
    from matplotlib.patches import Patch

    participant_ids = sorted(participant_ids)
    chance_nll = -np.log(1 / 9)
    w = 0.35  # bar half-width

    # ── load both conditions using the same approach as load_models ───────────
    def _load_models_from_path(base_path):
        model_dfs = {}
        for model_name in sorted(os.listdir(base_path)):
            model_path = os.path.join(base_path, model_name)
            if not os.path.isdir(model_path):
                continue
            df = read_data_from_folder(model_path)
            if df.empty or nll_column not in df.columns:
                continue
            model_dfs[model_name.replace('-', '_')] = df
        return model_dfs

    underscore_dfs = _load_models_from_path(underscore_path)
    lowercase_dfs  = _load_models_from_path(lowercase_path)

    ref_dfs = underscore_dfs if underscore_dfs else lowercase_dfs
    if not ref_dfs:
        print("No valid data loaded.")
        return {}

    # ── identify families and colors ──────────────────────────────────────────
    family_mapping = identify_model_families(ref_dfs)
    family_colors  = define_colors_for_families(family_mapping)
    # if family colors are missing any families, assign them a default color
    for family in family_mapping.keys():
        if family not in family_colors:
            family_colors[family] = ['#1f77b4']  # default blue
    print(f"these are families{family_mapping}")
    print(f"these are family colors{family_colors}")

    # ── helper: mean NLL for one participant across all models in a family ────
    def _mean_nll_for_pid(models_in_family, dfs_dict, pid):
        vals = []
        for model_name, _, _ in models_in_family:
            df = dfs_dict.get(model_name, pd.DataFrame())
            if df.empty:
                continue
            id_col = next((c for c in ('model_id', 'participant_id') if c in df.columns), None)
            pdf = df[df[id_col] == pid] if id_col else df
            if not pdf.empty and nll_column in pdf.columns:
                vals.extend(pdf[nll_column].dropna().tolist())
        return float(np.mean(vals)) if vals else None

    # ── one figure per family ─────────────────────────────────────────────────
    figs = {}
    x = np.arange(len(participant_ids))

    for family, models_in_family in family_mapping.items():
        fam_color = family_colors.get(family, ['#888888'])[0]
        set_dynamic_fontsize(fig_width=figsize[0], base_font=20)
        fig, ax = plt.subplots(figsize=figsize)

        for i, pid in enumerate(participant_ids):
            nll_lower = _mean_nll_for_pid(models_in_family, lowercase_dfs, pid)
            nll_under = _mean_nll_for_pid(models_in_family, underscore_dfs, pid)

            if nll_lower is not None:
                b = ax.bar(x[i] - w / 2, nll_lower, w,
                           color=fam_color, edgecolor='black', linewidth=0.5)
                ax.text(x[i] - w / 2, nll_lower + 0.003, f'{nll_lower:.3f}',
                        ha='center', va='bottom', fontsize=7)

            if nll_under is not None:
                ax.bar(x[i] + w / 2, nll_under, w,
                       color=fam_color, hatch='///', edgecolor='black', linewidth=0.5)
                ax.text(x[i] + w / 2, nll_under + 0.003, f'{nll_under:.3f}',
                        ha='center', va='bottom', fontsize=7)

        ax.axhline(chance_nll, ls='--', c='grey', lw=1.2)
        ax.text(x[-1] + w, chance_nll, ' Random\nguessing',
                va='bottom', ha='left', fontsize=8, color='grey')

        ax.set_xticks(x)
        ax.set_xticklabels([f'P{pid}' for pid in participant_ids], fontsize=10)
        ax.set_xlabel('Participant', labelpad=6)
        ax.set_ylabel('Negative Log-Likelihood (NLL)', labelpad=10)
        ax.set_title(family.capitalize(), fontsize=13, fontweight='bold', pad=10)
        ax.margins(x=0.1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)

        legend_handles = [
            Patch(facecolor=fam_color, edgecolor='black', label='lowercase'),
            Patch(facecolor=fam_color, edgecolor='black', hatch='///', label='underscore'),
        ]
        ax.legend(handles=legend_handles, fontsize=9, loc='upper right')

        plt.tight_layout()

        if out_png:
            stem, ext = os.path.splitext(out_png)
            out_family = f'{stem}_{family}{ext}'
            fig.savefig(out_family, dpi=300, bbox_inches='tight')
            print(f"Saved to {out_family}")

        figs[family] = fig

    return figs


def plot_nll_per_participant(
    base_paths,
    participant_ids,
    labels=None,
    nll_column='nll',
    figsize=None,
    out_png=None,
):
    """
    One subplot per participant; within each subplot, show mean NLL per model
    for both datasets side-by-side as grouped bars.

    Parameters
    ----------
    base_paths : sequence of 2 str
        Two dataset folders (e.g. ['predictive_underscore', 'predictive_lowercase']).
    participant_ids : sequence of int
        Participant IDs to plot — each gets its own subplot.
    labels : sequence of 2 str or None
        Legend labels for the two datasets; defaults to folder names.
    nll_column : str
        NLL column name in the CSVs.
    figsize : tuple or None
        Figure size; defaults to (5 * n_participants, 5).
    out_png : str or None
        If given, saves the figure here.
    """
    if len(base_paths) != 2:
        raise ValueError("base_paths must contain exactly 2 paths.")
    if labels is None:
        labels = [os.path.basename(p.rstrip('/')) for p in base_paths]

    n = len(participant_ids)
    if figsize is None:
        figsize = (5 * n, 6)

    # ── load both datasets once ──────────────────────────────────────────────
    def _load_dataset(base_path):
        model_dfs = {}
        for model_name in sorted(os.listdir(base_path)):
            model_path = os.path.join(base_path, model_name)
            if not os.path.isdir(model_path):
                continue
            df = read_data_from_folder(model_path)
            if df.empty or nll_column not in df.columns:
                continue
            model_dfs[model_name.replace('-', '_')] = df
        return model_dfs

    datasets = [_load_dataset(p) for p in base_paths]
    all_model_names = sorted(set().union(*[d.keys() for d in datasets]))

    # ── build figure ─────────────────────────────────────────────────────────
    set_dynamic_fontsize(fig_width=figsize[0] / n, base_font=20)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    w = 0.35
    chance_nll = -np.log(1 / 9)

    for ax, pid in zip(axes, participant_ids):
        means_per_dataset = []
        for ds in datasets:
            row_means = []
            for model_name in all_model_names:
                df = ds.get(model_name, pd.DataFrame())
                if df.empty:
                    row_means.append(None)
                    continue
                id_col = next((c for c in ('model_id', 'participant_id') if c in df.columns), None)
                if id_col:
                    df = df[df[id_col] == pid]
                if df.empty or nll_column not in df.columns:
                    row_means.append(None)
                    continue
                mean = float(df[nll_column].dropna().mean())
                row_means.append(mean)
            means_per_dataset.append(row_means)

        x = np.arange(len(all_model_names))
        for di, (ds_means, color, lbl) in enumerate(zip(means_per_dataset, colors, labels)):
            heights = [m if m is not None else 0 for m in ds_means]
            offset = (di - 0.5) * w
            bars = ax.bar(x + offset, heights, w, label=lbl, color=color,
                          edgecolor='black', linewidth=0.3)
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                            f'{h:.3f}', ha='center', va='bottom', fontsize=6.5)

        ax.axhline(chance_nll, ls='--', c='grey', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace('_', '\n') for m in all_model_names],
            fontsize=7, ha='center'
        )
        ax.set_title(f'Participant {pid}', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        ax.margins(x=0.08)

    axes[0].set_ylabel('Negative Log-Likelihood (NLL)', labelpad=10)
    axes[-1].legend(title='Dataset', fontsize=9, title_fontsize=9,
                    loc='upper right')
    ax.text(x[-1] + w, chance_nll, 'Random\nguessing',
            va='bottom', ha='left', fontsize=7, color='grey')

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"Saved to {out_png}")
    return fig


def plot_token_cascade(df, trial_indices=None, participant_id=None, out_png=None):
    """
    For each specified trial, plot the model's top-5 token distribution at every
    token position in the English phrase (from top_per_position).

    Highlights the actual chosen token in each panel and annotates its NLL
    contribution at the top. Useful for showing e.g. "model put 97% on 'somewhat'
    at position 0 but 'very' only got 0.2%, yet after 'very' in context 'high'
    was likely at 53%".

    Parameters
    ----------
    df : pd.DataFrame
        Output from predict_multi_cue — must have columns trial_index, ground_truth,
        nll, top_per_position.
    trial_indices : list[int] or None
        1-based trial indices to plot. If None, plots all trials with top_per_position.
    participant_id : int or None
        Used only in the figure title.
    out_png : str or None
        If given, saves the figure to this path.
    """
    import ast

    rows = df[df['top_per_position'].notna()].copy()
    if trial_indices is not None:
        rows = rows[rows['trial_index'].isin(trial_indices)]
    if rows.empty:
        print("No rows with top_per_position found.")
        return

    n_trials = len(rows)
    # Find max number of positions across selected trials
    def parse_top(val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val

    rows['_parsed'] = rows['top_per_position'].apply(parse_top)
    max_positions = rows['_parsed'].apply(len).max()

    fig, axes = plt.subplots(
        n_trials, max_positions,
        figsize=(max_positions * 3.5, n_trials * 3.2),
        squeeze=False
    )

    for row_i, (_, row) in enumerate(rows.iterrows()):
        positions = row['_parsed']
        ground_truth = str(row['ground_truth'])
        nll = row['nll']

        for pos_i in range(max_positions):
            ax = axes[row_i][pos_i]
            if pos_i >= len(positions):
                ax.axis('off')
                continue

            top5 = positions[pos_i]  # list of (token, prob)
            tokens = [t.replace('Ġ', ' ').replace('Ċ', '↵').strip() for t, _ in top5]
            probs = [p for _, p in top5]

            # Determine which bar corresponds to the actual chosen token
            actual_tokens_in_phrase = [
                t.replace('Ġ', ' ').replace('Ċ', '↵').strip()
                for t, _ in positions[pos_i]
            ]
            # The actual token at this position — try to match from ground_truth
            colors = ['#0072B2' if t == actual_tokens_in_phrase[j] else '#AAAAAA'
                      for j, t in enumerate(tokens)]
            # Highlight whichever bar matches the actual chosen token at this position
            # by checking ground_truth phrase tokens
            from_gt = ground_truth.replace('.', '').strip().split()
            actual_at_pos = from_gt[pos_i] if pos_i < len(from_gt) else None

            bar_colors = []
            for t in tokens:
                if actual_at_pos and t.lower() == actual_at_pos.lower():
                    bar_colors.append('#D55E00')  # highlight actual
                else:
                    bar_colors.append('#AAAAAA')

            y = range(len(tokens))
            bars = ax.barh(list(y), probs, color=bar_colors, edgecolor='black', linewidth=0.4)
            ax.set_yticks(list(y))
            ax.set_yticklabels(tokens, fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability', fontsize=8)
            ax.invert_yaxis()

            # Annotate probability values on bars
            for bar, prob in zip(bars, probs):
                ax.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                        f'{prob:.3f}', va='center', fontsize=7.5)

            pos_label = f'Position {pos_i}'
            if pos_i == 0:
                pos_label += ' (masked)'
            ax.set_title(pos_label, fontsize=9)

        # Row label: trial info + NLL
        axes[row_i][0].set_ylabel(
            f'Trial {int(row["trial_index"])}\nGT: "{ground_truth}"\nNLL: {nll:.3f}',
            fontsize=8, labelpad=8
        )

    pid_str = f'Participant {participant_id} — ' if participant_id is not None else ''
    fig.suptitle(f'{pid_str}Token-position distributions (orange = actual choice)', fontsize=11)
    plt.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        print(f"Saved to {out_png}")
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='data/out/predictive', help='Base predictive folder containing model subfolders')
    parser.add_argument('--out', default='figures/loglikelihood_bars.png', help='Output PNG filename')
    parser.add_argument('--include_participants', type=int, nargs='+', default=PARTICIPANTS_TO_PLOT,
                        help='List of participant IDs to plot (default: None = plot all)')
    parser.add_argument('--cascade', default=None, help='Path to a participant CSV to plot token cascade')
    parser.add_argument('--cascade_trials', type=int, nargs='+', default=None,
                        help='Trial indices to include in cascade plot (default: all)')
    parser.add_argument('--cascade_out', default='figures/token_cascade.png', help='Output PNG for cascade plot')
    # condition comparison subcommand flags
    parser.add_argument('--compare', nargs=2, metavar=('PATH1', 'PATH2'),
                        help='Two dataset folders to compare side-by-side (uses plot_nll_by_condition)')
    parser.add_argument('--compare_labels', nargs=2, metavar=('LABEL1', 'LABEL2'), default=None,
                        help='Subplot titles for --compare (defaults to folder names)')
    parser.add_argument('--conditions', type=int, nargs='+', default=None,
                        help='Condition IDs to include in --compare plot (e.g. 1 3); omit to skip condition filtering')
    parser.add_argument('--compare_out', default='figures/nll_conditions.png',
                        help='Output PNG for --compare plot')
    # per-participant comparison flags
    parser.add_argument('--per_participant', nargs=2, metavar=('PATH1', 'PATH2'),
                        help='Two dataset folders; plot one subplot per participant (uses plot_nll_per_participant)')
    parser.add_argument('--participants', type=int, nargs='+', default=None,
                        help='Participant IDs (used by --per_participant and --by_participant)')
    parser.add_argument('--per_participant_labels', nargs=2, metavar=('LABEL1', 'LABEL2'), default=None,
                        help='Legend labels for --per_participant (defaults to folder names)')
    parser.add_argument('--per_participant_out', default='figures/nll_per_participant.png',
                        help='Output PNG for --per_participant plot')
    # by-participant grouped bar flags (plot_nll_by_participant)
    parser.add_argument('--by_participant', action='store_true',
                        help='Plot NLL grouped bars per participant (uses plot_nll_by_participant)')
    parser.add_argument('--underscore', default=None, metavar='PATH',
                        help='Base folder for underscore condition (for --by_participant)')
    parser.add_argument('--lowercase', default=None, metavar='PATH',
                        help='Base folder for lowercase condition (for --by_participant)')
    parser.add_argument('--by_participant_out', default='figures/nll_by_participant.png',
                        help='Output PNG for --by_participant plot')
    args = parser.parse_args()

    if args.by_participant or (args.underscore and args.lowercase):
        if not args.participants:
            parser.error('--by_participant requires --participants with at least one ID')
        if not args.underscore or not args.lowercase:
            parser.error('--by_participant requires both --underscore and --lowercase paths')
        figs = plot_nll_by_participant(
            underscore_path=args.underscore,
            lowercase_path=args.lowercase,
            participant_ids=args.participants,
            nll_column='nll',
            out_png=args.by_participant_out,
        )
    elif args.per_participant:
        if not args.participants:
            parser.error('--per_participant requires --participants with at least one ID')
        fig = plot_nll_per_participant(
            base_paths=args.per_participant,
            participant_ids=args.participants,
            labels=args.per_participant_labels,
            nll_column='nll',
            out_png=args.per_participant_out,
        )
    elif args.compare:
        fig = plot_nll_by_condition(
            base_paths=args.compare,
            labels=args.compare_labels,
            conditions=args.conditions,
            participant_ids=args.participants,
            nll_column='nll',
            out_png=args.compare_out,
        )
    elif args.cascade:
        df = pd.read_csv(args.cascade)
        pid = int(re.search(r'participant_(\d+)', args.cascade).group(1)) if re.search(r'participant_(\d+)', args.cascade) else None
        plot_token_cascade(df, trial_indices=args.cascade_trials, participant_id=pid, out_png=args.cascade_out)
    else:
        load_and_plot(base_path=args.base, out_png=args.out, nll_column='nll', include_participants=args.include_participants)