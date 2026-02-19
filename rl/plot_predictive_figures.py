import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data_from_folder(folder_path):
    dfs = pd.DataFrame()
    file_count = 0  # counter for loaded files

    # Regex to extract the number after "participant_" and before ".csv"
    # This regex looks for "participant_" followed by one or more digits (\d+)
    # and captures these digits. It expects ".csv" at the end.
    participant_id_regex = re.compile(r'model_(\d+)\.csv')


    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Extract model_id from filename using regex
            match = participant_id_regex.search(filename)
            if match:
                model_id = int(match.group(1)) # Convert the captured digits to an integer
                df['model_id'] = model_id # Add the model_id column
            else:
                # Handle cases where the filename doesn't match the expected format
                print(f"Warning: Could not extract model_id from filename: {filename}")
                df['model_id'] = None # Or some other indicator of missing ID

            dfs = pd.concat([dfs, df], ignore_index=True)
            file_count += 1  # increment counter

    print(f"{file_count} CSV file(s) loaded.")
    return dfs

def calculate_log_likelihood_stats(df, column_name='log_likelihood'):
    """
    Calculates the mean and standard error of the mean (SEM)
    for the specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a
                           'log_likelihood' column.

    Returns:
        tuple: A tuple containing the mean log-likelihood and the SEM.
    """
    # Drop missing log-likelihoods (NaN)
    valid_ll = df[column_name].dropna()

    # Compute mean and SEM
    if not valid_ll.empty:
        mean_ll = valid_ll.mean()
        sem_ll = valid_ll.std(ddof=1) / np.sqrt(len(valid_ll))
        return mean_ll, sem_ll
    else:
        return None, None

def plot_loglikelihood_bars_large_only(mu_centaurB, sd_centaurB,
                                       mu_llamaB, sd_llamaB,
                                       mu_rw, sd_rw,
                                       variant_labels = ['70B', '70B', 'RW'],
                                       family_labels  = ['Centaur', 'LLaMA 3.1', 'Domain-Specific\nModel'],
                                       colors = ['#D55E00', '#0072B2', '#CC79A7'],
                                       figsize=(10, 8)):
    """
    Plot NLL bar chart for three models with labels.
    """
    # Values
    means = [-mu_centaurB, -mu_llamaB, mu_rw]
    errs  = [sd_centaurB, sd_llamaB, sd_rw]

    # Simple evenly spaced positions
    xpos = np.array([0, 1, 2])
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars
    bars = ax.bar(xpos, means, bar_width,
                  yerr=errs, capsize=5,
                  color=colors, edgecolor='black', linewidth=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    # Chance line
    chance_nll = -np.log(0.5)
    ax.axhline(chance_nll, ls='--', c='grey', lw=1.2)
    ax.text(xpos[-1]-0.1, chance_nll, 'Random guessing',
            va='bottom', ha='left', fontsize=14)

    # X-axis: tick labels
    ax.set_xticks(xpos)
    ax.set_xticklabels(variant_labels, ha='center', fontsize=16)

    # Family labels below tick labels
    for i, x in enumerate(xpos):
        ax.text(x, -0.12, family_labels[i],
                ha='center', va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=20)

    # Styling
    ax.set_ylabel('Negative Log-Likelihood (NLL)', fontsize=16)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    ax.margins(x=0.1)

    plt.tight_layout()
    return fig
def set_dynamic_font_size(fig_size, base_font_size=12):
    """
    Adjust font sizes dynamically based on figure size.
    """
    width, height = fig_size
    scale_factor = (width * height) / (10 * 8)  # Base size is for 10x8 inches
    new_font_size = int(base_font_size * scale_factor)
    plt.rcParams.update({'font.size': new_font_size})
    plt.rcParams.update({
        'font.size': new_font_size,
        'axes.titlesize': new_font_size + 4,
        'axes.labelsize': new_font_size + 2,
        'xtick.labelsize': new_font_size + 2,
        'ytick.labelsize': new_font_size + 2,
        'legend.fontsize': max(new_font_size - 2, 6)
    })


def load_predictive_families(root_folder):
    """Load CSVs from each subfolder under `root_folder`.

    Returns a dict mapping folder name -> DataFrame with per-model mean NLL.
    Each DataFrame has columns: `model_file`, `mean_nll`.
    """
    families = {}
    if not os.path.isdir(root_folder):
        raise ValueError(f"Root folder not found: {root_folder}")

    for entry in sorted(os.listdir(root_folder)):
        family_path = os.path.join(root_folder, entry)
        if not os.path.isdir(family_path):
            continue

        records = []
        for filename in sorted(os.listdir(family_path)):
            if not filename.endswith('.csv'):
                continue
            file_path = os.path.join(family_path, filename)
            try:
                df = pd.read_csv(file_path)
            except Exception:
                print(f"Warning: failed to read {file_path}")
                continue

            # Expect a `log_likelihood` column; compute mean NLL for this model file
            if 'log_likelihood' in df.columns:
                valid = df['log_likelihood'].dropna()
                if valid.empty:
                    continue
                mean_ll = valid.mean()
                mean_nll = -mean_ll
                records.append({'model_file': filename, 'mean_nll': mean_nll})
            else:
                # If a `nll` column already exists, use it
                if 'nll' in df.columns:
                    valid = df['nll'].dropna()
                    if valid.empty:
                        continue
                    records.append({'model_file': filename, 'mean_nll': valid.mean()})
                else:
                    print(f"Warning: no `log_likelihood` or `nll` in {file_path}")

        families[entry] = pd.DataFrame.from_records(records)

    return families


def compute_family_stats(family_dfs):
    """From dict family->df (with `mean_nll` per model), compute mean and std across models.

    Returns dict family -> (mean_nll, std_nll, count_models)
    """
    stats = {}
    for family, df in family_dfs.items():
        if df.empty or 'mean_nll' not in df.columns:
            stats[family] = (np.nan, np.nan, 0)
            continue
        vals = df['mean_nll'].dropna()
        if vals.empty:
            stats[family] = (np.nan, np.nan, 0)
            continue
        stats[family] = (vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0, len(vals))

    return stats


def plot_averaged_nll_bars(family_stats, colors=None, figsize=(10, 6), save_path=None):
    """Plot averaged NLL bars for each family.

    `family_stats` should be an ordered dict-like mapping label->(mean, std, count).
    """
    labels = list(family_stats.keys())
    means = [family_stats[l][0] for l in labels]
    errs = [family_stats[l][1] for l in labels]

    xpos = np.arange(len(labels))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=figsize)
    if colors is None:
        colors = plt.cm.tab10.colors
    bar_colors = [colors[i % len(colors)] for i in range(len(labels))]

    bars = ax.bar(xpos, means, bar_width, yerr=errs, capsize=5, color=bar_colors, edgecolor='black', linewidth=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    chance_nll = -np.log(0.5)
    ax.axhline(chance_nll, ls='--', c='grey', lw=1.2)
    ax.text(xpos[-1] if len(xpos) else 0, chance_nll, 'Random guessing', va='bottom', ha='left', fontsize=10)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, ha='center', fontsize=12)
    ax.set_ylabel('Average Negative Log-Likelihood (NLL)', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


if __name__ == '__main__':
    root = os.path.join('data', 'out', 'predictive')
    try:
        families = load_predictive_families(root)
    except ValueError as e:
        print(e)
        families = {}

    stats = compute_family_stats(families)

    if stats:
        fig = plot_averaged_nll_bars(stats, figsize=(10, 6))
        plt.show()