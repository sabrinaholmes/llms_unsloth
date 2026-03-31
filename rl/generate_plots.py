import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

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
    """
        if df is None or df.empty:
            print("No data available for normalization count.")
            return 0
        if 'nll' not in df.columns or 'raw_nll' not in df.columns:
            print("Required columns 'nll' and 'raw_nll' not found in DataFrame.")
            return 0
        normalized_series = df['nll'] != df['raw_nll']
        num_normalized = int(normalized_series.sum())
        total = len(df)
        pct = num_normalized / total if total > 0 else 0.0
        print(f"Number of normalized NLL: {num_normalized} out of {total} ({pct:.2%})")
        return num_normalized


    def normalization_stats_and_plots(df, raw_col='raw_nll', norm_col='nll', out_prefix='normalization'):
        """Compute normalization stats and save visualization files.

        Produces:
        - {out_prefix}_hist_diff.png : histogram of (raw - normalized) differences
        - {out_prefix}_absdiff_hist.png : histogram of absolute differences
        - {out_prefix}_scatter_raw_vs_norm.png : scatter of raw vs normalized with y=x line
        - {out_prefix}_box_by_model.png : boxplot of differences per `model_id` (if present)

        Returns a stats dict summarizing counts and summary statistics.
        """
        stats = {}
        if df is None or df.empty:
            print("No data to compute normalization stats.")
            return stats
        if raw_col not in df.columns or norm_col not in df.columns:
            print(f"Columns {raw_col} or {norm_col} not found, skipping stats/plots.")
            return stats

        sub = df[[raw_col, norm_col]].dropna()
        if sub.empty:
            print("No non-NA pairs of raw/norm values to analyze.")
            return stats

        raw = sub[raw_col].astype(float)
        norm = sub[norm_col].astype(float)
        diff = raw - norm
        absdiff = diff.abs()
        # relative change where raw != 0
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_change = diff / raw.replace(0, np.nan)

        num_total = len(sub)
        num_normalized = int((raw != norm).sum())
        pct_normalized = num_normalized / num_total if num_total > 0 else 0.0

        stats.update({
            'n_total': num_total,
            'n_normalized': num_normalized,
            'pct_normalized': pct_normalized,
            'mean_diff': float(diff.mean()),
            'median_diff': float(diff.median()),
            'std_diff': float(diff.std(ddof=1)),
            'mean_absdiff': float(absdiff.mean()),
            'median_absdiff': float(absdiff.median()),
            'quantiles_diff': diff.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict(),
        })

        # Create plots
        try:
            # Histogram of signed differences
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(diff, bins=50, color='#56B4E9', edgecolor='k')
            ax.axvline(0, color='k', lw=1)
            ax.set_title('Signed difference: raw_nll - nll')
            ax.set_xlabel('Difference')
            ax.set_ylabel('Count')
            fig.tight_layout()
            hist_path = f"{out_prefix}_hist_diff.png"
            fig.savefig(hist_path, dpi=200)
            plt.close(fig)

            # Histogram of absolute differences
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(absdiff, bins=50, color='#D55E00', edgecolor='k')
            ax.set_title('Absolute difference: |raw_nll - nll|')
            ax.set_xlabel('Absolute difference')
            ax.set_ylabel('Count')
            fig.tight_layout()
            abs_hist_path = f"{out_prefix}_absdiff_hist.png"
            fig.savefig(abs_hist_path, dpi=200)
            plt.close(fig)

            # Scatter raw vs normalized
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(raw, norm, s=8, alpha=0.6)
            mn = min(raw.min(), norm.min())
            mx = max(raw.max(), norm.max())
            ax.plot([mn, mx], [mn, mx], color='k', linestyle='--', linewidth=1)
            ax.set_xlabel('raw_nll')
            ax.set_ylabel('nll')
            ax.set_title('raw_nll vs nll')
            fig.tight_layout()
            sc_path = f"{out_prefix}_scatter_raw_vs_norm.png"
            fig.savefig(sc_path, dpi=200)
            plt.close(fig)

            # Boxplot by model_id if available
            if 'model_id' in df.columns:
                tmp = df[[raw_col, norm_col, 'model_id']].dropna()
                tmp['diff'] = tmp[raw_col] - tmp[norm_col]
                grouped = tmp.groupby('model_id')['diff'].apply(list)
                if not grouped.empty:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.boxplot(grouped.tolist(), labels=[str(x) for x in grouped.index], vert=True)
                    ax.set_xlabel('model_id')
                    ax.set_ylabel('raw_nll - nll')
                    ax.set_title('Per-model normalization differences')
                    fig.tight_layout()
                    box_path = f"{out_prefix}_box_by_model.png"
                    fig.savefig(box_path, dpi=200)
                    plt.close(fig)
                    stats['boxplot_path'] = box_path

            stats['hist_path'] = hist_path
            stats['abs_hist_path'] = abs_hist_path
            stats['scatter_path'] = sc_path
            print(f"Saved normalization plots with prefix '{out_prefix}_*.png'")
        except Exception as e:
            print(f"Failed to create/save normalization plots: {e}")

        return stats
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

    chance_nll = -np.log(0.5)
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
    ax.set_ylim(0, 0.75)
    plt.tight_layout()
    return fig


def load_and_plot(base_path='predictive', out_png='loglikelihood_bars.png',nll_column='log_likelihood'):
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
    fig = plot_loglikelihood_bars_dynamic(family_mapping=family_mapping, nll_column=nll_column, figsize=(12,10))
    print({k: len(v) for k, v in family_mapping.items()})
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot to {out_png}")

# function to tell how many nll were normalized
def count_normalized_nll(df):
    # how many nll different from raw nll
    df['normalized'] = df['nll'] != df['raw_nll']
    num_normalized = df['normalized'].sum()
    total = len(df)
    print(f"Number of normalized NLL: {num_normalized} out of {total} ({num_normalized/total:.2%})")
    # what percentage of nll were normalized
    print(f"Percentage of normalized NLL: {num_normalized/total:.2%}")
    return num_normalized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='predictive', help='Base predictive folder containing model subfolders')
    parser.add_argument('--out', default='loglikelihood_bars.png', help='Output PNG filename')
    args = parser.parse_args()
    load_and_plot(base_path=args.base, out_png=args.out, nll_column='raw_nll')
    # Provide counts and plots for common families if present
    for name in ('llama', 'centaur'):
        df = globals().get(f"{name}_df", pd.DataFrame())
        print(f"\nSummary for {name} (if available):")
        count_normalized_nll(df)
        stats = normalization_stats_and_plots(df, out_prefix=f"{name}_normalization")
        if stats:
            print(f"Normalization stats for {name}: {stats}")