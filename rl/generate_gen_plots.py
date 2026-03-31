import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

def read_data_from_folder(folder_path):
    dfs = pd.DataFrame()
    # Prefer a 'singles' subfolder but fall back to the folder itself (useful for rw CSVs)
    singles_path = os.path.join(folder_path, 'singles')
    if os.path.isdir(singles_path):
        search_path = singles_path
    elif os.path.isdir(folder_path):
        search_path = folder_path
    else:
        print(f"Warning: folder not found at: {folder_path}")
        return dfs

    file_count = 0  # counter for loaded files
    auto_id = 0
    # Regex to extract the number after "participant_" or "model_".
    participant_id_regex = re.compile(r'(?:model|participant)_(\d+)')

    for filename in os.listdir(search_path):
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

def bandit_1_prop(df,choice):
  df_copy = df.copy()
  # Determine comparison value depending on column dtype and provided `choice`.
  col = df_copy['choice']
  # If the choice column is numeric but user passed a letter ('U'/'P'), map to numeric codes.
  if pd.api.types.is_numeric_dtype(col):
      if isinstance(choice, str):
          mapping = {'U': 0, 'P': 1}
          choice_val = mapping.get(choice, None)
          if choice_val is None:
              try:
                  choice_val = float(choice)
              except Exception:
                  choice_val = choice
      else:
          choice_val = choice
  else:
      choice_val = choice

  df_copy['bandit1_chosen'] = (df_copy['choice'] == choice_val).astype(int)
  df_copy['bandit_1_avg'] = df_copy.groupby("model_id")["bandit1_chosen"].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
  return df_copy

def plot_bandit_choice_trends_single_axis(human_df=None, df_rep=None, dfs=None, labels=None, colors=None,
                                          trial_col="trial_num", bandit_avg_col="bandit_1_avg",
                                          reversal_trials=[14, 50], timeline=None,
                                          xlim=(0, 100), margins=True, legend_anchor=(0.5, 0.5),
                                          fig_size=(14, 9)):
    """
    Plots bandit 1 choice trends (no reward probability subplot). Overlays human and model predictions.

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
        legend_anchor (tuple): Anchor position for external legend.
    """
    fig, ax = plt.subplots(figsize=fig_size)

    set_dynamic_fontsize(fig_width=fig_size[0], base_font=20)

    # --- Human Data ---
    if human_df is not None:
        grouped = human_df.groupby(trial_col)[bandit_avg_col]
        mean = grouped.mean()
        sem = grouped.std() / np.sqrt(grouped.count())
        human_color = colors[0]

        ax.plot(mean.index, mean.values, color=human_color, linewidth=0.5, label='Human', alpha=1)
        if margins:
            ax.fill_between(mean.index, mean - sem, mean + sem, color=human_color, alpha=0.2)

    # --- Repetitive Baseline ---
    if df_rep is not None:
        grouped = df_rep.groupby(trial_col)[bandit_avg_col]
        mean = grouped.mean()
        sem = grouped.std() / np.sqrt(grouped.count())
        rep_color = colors[-1]

        ax.plot(mean.index, mean.values, color=rep_color, linewidth=1.5, label=labels[-1], alpha=1)
        if margins:
            ax.fill_between(mean.index, mean - sem, mean + sem, color=rep_color, alpha=0.3, hatch='/')

    # --- Model Data ---
    if dfs is not None and labels is not None and colors is not None:
        model_colors = colors[1:]  # skip human
        for i, (df, label) in enumerate(zip(dfs, labels)):
            grouped = df.groupby(trial_col)[bandit_avg_col]
            mean = grouped.mean()
            sem = grouped.std() / np.sqrt(grouped.count())
            model_color = model_colors[i % len(model_colors)]

            ax.plot(mean.index, mean.values, label=label, color=model_color, linewidth=1.5, alpha=1)
            if margins:
                ax.fill_between(mean.index, mean - sem, mean + sem, color=model_color, alpha=0.4)

    # --- Reversal Markers ---
    for reversal in reversal_trials:
        ax.axvline(x=reversal, color='black', linestyle='--', linewidth=1.0, alpha=0.3)

    # --- Optional Ground Truth Rewards ---
    if timeline is not None:
        bandit_1_rewards = [trial["bandit_1"]["value"] for trial in timeline]
        ax.plot(range(1, len(bandit_1_rewards)+1), bandit_1_rewards,
                color='gray', marker='|', linestyle='', markersize=10, alpha=0.6,
                label='Ground Truth Rewards')
    pad_value = fig_size[1] * 3  # multiplier can be adjusted

    # --- Formatting ---
    ax.set_xlabel("Trial Number",labelpad=pad_value)
    ax.set_ylabel("Bandit 1 Choice Rate",labelpad=20)
    ax.set_xlim(xlim)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(False)
    ax.margins(x=0.04)

    # --- Legend saved separately ---
    handles, labels_legend = ax.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(6, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels_legend, loc='center', frameon=False)
    ax.margins(x=0.04)
    plt.tight_layout()
    return fig, legend_fig


def bandit_choice_trends(base_path='./llms_unsloth/rl/data/out/generative', choice='U', save_dir='./figures'):
    """
    Load all model single-run CSVs from `base_path`, compute bandit choice prop for `choice`,
    group models into families, pick colors, and plot combined bandit choice trends.

    Parameters
    - base_path: str path to the `generative` output folder containing model subfolders.
    - choice: character for bandit 1 (e.g., 'U' or 'P').
    - save_dir: directory to save plots.
    """
    base_path = os.path.abspath(base_path)
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)


    # Load all model folders
    model_dfs = {}
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Base path not found: {base_path}")

    for model_name in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, model_name)
        if not os.path.isdir(model_path):
            continue
        df = read_data_from_folder(model_path)
        if df.empty:
            continue
        # normalize name for internal mapping
        model_key = model_name
        # ensure trial column exists as `trial_num`
        if 'trial_num' not in df.columns and 'trial_index' in df.columns:
            df = df.rename(columns={'trial_index': 'trial_num'})
        # compute bandit prop
        df = bandit_1_prop(df, choice)
        model_dfs[model_key] = df

    if not model_dfs:
        print(f"No model data loaded from {base_path}")
        return None

    # Identify families and sizes
    family_mapping = identify_model_families(model_dfs)
    family_colors = define_colors_for_families(family_mapping)

    # Create mapping of model -> color by using family's color list cycling through variants
    model_color_map = {}
    for fam, models in family_mapping.items():
        colors = family_colors.get(fam, ['#000000'])
        for idx, (model_name, df, size_label) in enumerate(models):
            model_color_map[model_name] = colors[idx % len(colors)]

    # Build plotting lists (no human data available by default)
    labels = []
    dfs = []
    colors = []

    # prepend a placeholder human color so plotting function's indexing is preserved
    human_color = '#000000'
    colors.append(human_color)

    for model_name, df in model_dfs.items():
        labels.append(model_name)
        dfs.append(df)
        colors.append(model_color_map.get(model_name, '#000000'))

    # No explicit repetitive baseline by default; append a neutral color for end of colors
    rep_color = '#999999'
    colors.append(rep_color)

    # Use trial range from data if possible
    first_df = next(iter(dfs))
    max_trial = int(first_df['trial_num'].max()) if 'trial_num' in first_df.columns else 100

    fig, legend_fig = plot_bandit_choice_trends_single_axis(human_df=None, df_rep=None,
                                                           dfs=dfs, labels=labels, colors=colors,
                                                           trial_col='trial_num', bandit_avg_col='bandit_1_avg',
                                                           xlim=(1, max_trial), margins=True,
                                                           fig_size=(12, 10))

    out_fig = os.path.join(save_dir, f'bandit_choice_trends_choice_{choice}.png')
    out_legend = os.path.join(save_dir, f'bandit_choice_trends_legend_choice_{choice}.png')
    fig.savefig(out_fig, dpi=200, bbox_inches='tight')
    legend_fig.savefig(out_legend, dpi=200, bbox_inches='tight')
    plt.close(fig)
    plt.close(legend_fig)
    print(f"Saved plots: {out_fig}, {out_legend}")
    return out_fig, out_legend


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate bandit choice trend plots for generative outputs')
    parser.add_argument('--base', default='./llms_unsloth/rl/data/out/generative', help='Base generative output folder')
    parser.add_argument('--choice', default='U', help="Bandit choice character to treat as bandit 1 (e.g., 'U' or 'P')")
    parser.add_argument('--out', default='./figures', help='Output directory for saved figures')
    args = parser.parse_args()
    bandit_choice_trends(base_path=args.base, choice=args.choice, save_dir=args.out)
