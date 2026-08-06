# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binomtest
import os
import re
import sys
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from matplotlib.legend_handler import HandlerBase
import argparse
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils

FIGSIZE = (8, 6)
BASE_FONT = 24

# Allow overriding the base data folder via CLI or environment
default_path = os.path.join(os.path.dirname(__file__), 'data', 'out', 'generative')
figures_save_path = os.path.join(os.path.dirname(__file__), 'figures')
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--base-dir', dest='base_dir', default=default_path,
                    help='Base folder for model data (overrides default PATH)')
parser.add_argument('--save-dir', dest='save_dir', default=figures_save_path,
                    help='If provided, save all generated figures to this folder')
parser.add_argument('--show', dest='show', action='store_true',
                    help='If provided, show figures (default is notebook behavior)')
parser.add_argument('--original-data', dest='original_data',
                    default=os.path.join(os.path.dirname(__file__), 'data', 'in', 'allHorizonData_cut.csv'),
                    help='Path to the original human data CSV (wide format)')
parser.add_argument('--timeline', dest='timeline',
                    default=os.path.join(os.path.dirname(__file__), 'data', 'in', 'timeline_structure.csv'),
                    help='Path to the timeline structure CSV (long format, used to enrich generative outputs)')
parser.add_argument('--llm-to-plot', dest='llm_to_plot', default=None,
                    help='If provided, only plot this specific LLM (e.g., "centaur-70B" or "llama-13B")')
parser.add_argument('--llm-size', dest='llm_size', default=None,
                    help='If provided, only plot LLMs of this size (e.g., "8b" or "70b")')
args, _unknown = parser.parse_known_args()
PATH = args.base_dir

# Toggle: "orange" or "lila" — controls which shades Centaur is drawn in
# across every plot in this file (flows through plotting_utils.define_colors_for_families).
CENTAUR_COLOR_MODE = "orange"

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

def normalize_generative_df(df, participant_id, timeline_df):
    """
    Enrich a bare generative-output CSV (cols: game, trial, choice, reward, horizon)
    with game-level metadata from `timeline_df` (m1, m2, info_condition, block, gameLength).

    Parameters
    ----------
    df : pd.DataFrame
        Single-participant generative CSV, already containing a `model_id` column.
    participant_id : int
        The participant whose timeline stimuli the model was simulating.
    timeline_df : pd.DataFrame
        Long-format timeline (timeline_structure.csv).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame ready for `reshape_llm_df`.
    """
    tl = timeline_df[timeline_df['participant_id'] == participant_id].copy()

    # Normalise timeline column names to what the pipeline expects
    rename_tl = {}
    if 'trial_num_block' in tl.columns:
        rename_tl['trial_num_block'] = 'trial'
    if 'trial_num' in tl.columns and 'trial' not in tl.columns:
        rename_tl['trial_num'] = 'trial'
    if 'reward_mean_H' in tl.columns:
        rename_tl['reward_mean_H'] = 'm1'
    if 'reward_mean_I' in tl.columns:
        rename_tl['reward_mean_I'] = 'm2'
    if rename_tl:
        tl = tl.rename(columns=rename_tl)

    # Keep only the columns we need for the join
    meta_cols = ['game', 'trial', 'block', 'gameLength', 'info_condition', 'm1', 'm2']
    meta_cols = [c for c in meta_cols if c in tl.columns]
    dedup_subset = [c for c in ['game', 'trial'] if c in tl.columns]
    tl = tl[meta_cols].drop_duplicates(subset=dedup_subset or None)

    # Normalise df's trial column: read_data_from_folder may have renamed trial→trial_num
    df = df.copy()
    if 'trial_num' in df.columns and 'trial' not in df.columns:
        df = df.rename(columns={'trial_num': 'trial'})

    # Drop any pre-existing m1/m2 so we always use values from the timeline
    df = df.drop(columns=[c for c in ['m1', 'm2'] if c in df.columns])

    # choice_numeric is already 1/2 for free trials; choice has the value for forced trials.
    # Combine: prefer choice_numeric, fall back to choice (forced arms).
    if 'choice_numeric' in df.columns:
        df['choice'] = df['choice_numeric'].combine_first(df['choice'])
    elif df['choice'].dtype == object:
        df['choice'] = df.apply(map_choices_to_numeric, axis=1)
    

    # Merge to add game metadata
    merge_on = [c for c in ['game', 'trial'] if c in df.columns and c in tl.columns]
    df = df.merge(tl, on=merge_on, how='left')

    # Add participant_id column (pipeline uses this before reshape)
    df['participant_id'] = participant_id

    # gameLength: timeline may already have 5/10; if only horizon is present, derive it
    if 'gameLength' not in df.columns and 'horizon' in df.columns:
        df['gameLength'] = df['horizon'].map({1: 5, 6: 10})

    return df


def load_models(base_path=None, timeline_df=None):
    """
    Robustly load model CSVs from subfolders under `base_path`.

    Returns
    -------
    tuple
        (model_dfs, model_subjects)
        - model_dfs: dict mapping model_name -> DataFrame (concatenated CSVs)
        - model_subjects: dict mapping model_name -> list of processed subject dicts
    """
    if base_path is None:
        base_path = PATH

    model_dfs = {}
    model_subjects = {}

    if not os.path.isdir(base_path):
        print(f"Warning: base_path does not exist: {base_path}")
        return model_dfs, model_subjects

    for entry in sorted(os.listdir(base_path)):
        model_path = os.path.join(base_path, entry)
        if not os.path.isdir(model_path):
            continue

        # Try multiple common subfolder name variants
        candidates = [os.path.join(model_path, 'singles'), model_path]
        # also try lower-case folder name fallback
        candidates.append(os.path.join(base_path, entry.lower()))

        df = pd.DataFrame()
        for cand in candidates:
            if os.path.isdir(cand):
                df = read_data_from_folder(cand)
                if not df.empty:
                    break

        if df.empty:
            print(f"No CSVs found for model folder: {model_path} (skipping)")
            continue

        # Enrich bare generative CSVs (they lack m1, m2, info_condition, block)
        if timeline_df is not None and 'horizon' in df.columns:
            enriched_parts = []
            for mid in df['model_id'].unique():
                part = df[df['model_id'] == mid].copy()
                try:
                    part = normalize_generative_df(part, int(mid), timeline_df)
                    enriched_parts.append(part)
                except Exception as e:
                    print(f"Warning: could not enrich model_id={mid} in {entry}: {e}")
            if enriched_parts:
                df = pd.concat(enriched_parts, ignore_index=True)
                #print(f"enriched model columns {df.columns}")
            # map choice H/I → 1/2 if not already done in normalize_generative_df

        model_name_key = entry.replace('-', '_')
        model_dfs[model_name_key] = df

        # Determine whether DataFrame is long or wide
        subjects = []
        try:
            # If long-format (has participant_id/model_id or choice+trial/trial_num), reshape
            if ('participant_id' in df.columns or 'model_id' in df.columns or
                    ('choice' in df.columns and
                     any(c in df.columns for c in ['trial', 'trial_num']))):
                try:
                    # reshape_llm_df normalises model_id/participant_id/trial_n
                    wide = reshape_llm_df(df)
                    subjects = process_subjects_llm(wide)
                except Exception as inner_e:
                    print(f"  reshape failed ({inner_e}), skipping model")
                    subjects = []
        except Exception as e:
            print(f"Warning while processing model {entry}: {e}")

        model_subjects[model_name_key] = subjects
        print(f"Loaded model '{model_name_key}': df shape {df.shape}, subjects={len(subjects)}")

    return model_dfs, model_subjects
    


def identify_model_families(model_dfs):
    """
    Identify model families based on model names and return a mapping of family to
    model tuples (model_name, DataFrame, size_label).

    Size detection recognizes patterns like '-70B' or '-8B' (case-insensitive)
    and returns size_label as '70B' or '8B', otherwise None.
    """
    family_mapping = {}
    size_regex = re.compile(r"[_-](\d{1,3})\s*[bB]\b")
    for model_name, df in model_dfs.items():
        fam = model_name.split('_')[0].lower()  # base family name (e.g., 'centaur', 'llama', 'rw')
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
    return plotting_utils.define_colors_for_families(family_mapping, centaur_color_mode=CENTAUR_COLOR_MODE)


def read_original_data(PATH_TO_ORIGINAL_DATA):
    """Reads the original data from the specified CSV file and returns the full DataFrame and a filtered DataFrame for the pilot experiment."""
    df_wilson = pd.read_csv(PATH_TO_ORIGINAL_DATA)
    df_pilot=df_wilson[df_wilson['expt_name']=='pilot-v1']

    return df_wilson,df_pilot


def process_subjects(df):
    subjects = []
    subject_ids = df['subjectNumber'].unique()

    for sn in subject_ids:
        sub_df = df[df['subjectNumber'] == sn].reset_index(drop=True)
        sub = {}

        # Basic info
        sub['expt_name'] = sub_df.loc[0, 'expt_name']
        sub['subjectID'] = sub_df.loc[0, 'subjectID']
        sub['subjectNumber'] = sn

        # Columns
        sub['block'] = sub_df['block'].values
        sub['game'] = sub_df['game'].values
        sub['gameLength'] = sub_df['gameLength'].values
        sub['uc'] = sub_df['uc'].values
        sub['m1'] = sub_df['m1'].values
        sub['m2'] = sub_df['m2'].values
        sub['gID'] = sub_df['gID'].values
        sub['repeatNumber'] = sub_df['repeatNumber'].values

        r = sub_df[[f'r{i}' for i in range(1, 11)]].to_numpy(dtype=float)
        a = sub_df[[f'c{i}' for i in range(1, 11)]].to_numpy(dtype=float)
        RT = sub_df[[f'rt{i}' for i in range(1, 11)]].to_numpy()

        sub['r'] = r
        sub['a'] = a
        sub['RT'] = RT

        # z-scored RT
        sub['RTz'] = (RT - np.nanmean(RT)) / np.nanstd(RT)

        # Cumulative choices
        sub['n1'] = np.cumsum(a == 1, axis=1)
        sub['n2'] = np.cumsum(a == 2, axis=1)

        # Cumulative rewards
        sub['R1'] = np.cumsum(r * (a == 1), axis=1)
        sub['R2'] = np.cumsum(r * (a == 2), axis=1)

        # Observed mean — element-wise safe division avoids divide-by-zero warnings
        # for trials where an arm was never chosen yet (n1/n2 == 0).
        with np.errstate(divide='ignore', invalid='ignore'):
            sub['o1'] = np.where(sub['n1'] > 0,
                                 sub['R1'] / np.where(sub['n1'] > 0, sub['n1'], 1),
                                 0.0)
            sub['o2'] = np.where(sub['n2'] > 0,
                                 sub['R2'] / np.where(sub['n2'] > 0, sub['n2'], 1),
                                 0.0)

        # Correct choice
        m1 = sub['m1'][:, np.newaxis]
        m2 = sub['m2'][:, np.newaxis]
        co = ((m1 > m2) * (a == 1)) + ((m1 < m2) * (a == 2))
        co[np.isnan(a)] = np.nan
        sub['co'] = co

        # Low observed mean (random exploration)
        o1 = sub['o1']
        o2 = sub['o2']
        lm = ((o1[:, :-1] < o2[:, :-1]) * (a[:, 1:] == 1)) + \
             ((o1[:, :-1] > o2[:, :-1]) * (a[:, 1:] == 2))
        lm[o1[:, :-1] == o2[:, :-1]] = np.nan
        lm[np.isnan(a[:, 1:])] = np.nan
        lm_shifted = np.full_like(a, np.nan)
        lm_shifted[:, 1:] = lm
        sub['lm'] = lm_shifted

        # High info (directed exploration)
        n1 = sub['n1']
        n2 = sub['n2']
        hi = ((n1[:, :-1] < n2[:, :-1]) * (a[:, 1:] == 1)) + \
             ((n1[:, :-1] > n2[:, :-1]) * (a[:, 1:] == 2))
        hi[n1[:, :-1] == n2[:, :-1]] = np.nan
        hi[np.isnan(a[:, 1:])] = np.nan
        hi_shifted = np.full_like(a, np.nan)
        hi_shifted[:, 1:] = hi
        sub['hi'] = hi_shifted

        # Repetition (same choice as previous)
        rep = np.full_like(a, np.nan)
        rep[:, 1:] = (a[:, 1:] == a[:, :-1])
        sub['rep'] = rep

        # Trial 5 features
        sub['dR'] = sub['o2'][:, 3] - sub['o1'][:, 3]
        sub['dI'] = -(sub['n2'][:, 3] - sub['n1'][:, 3]) / 2
        sub['choice'] = a[:, 4]
        sub['rt'] = RT[:, 4]

        subjects.append(sub)

    return subjects

def bin_it(X, Y, bin_edges, bin_type='std'):
    # Bins Y, as a function of X, with bin edges given by bin_edges

    bin_space = np.diff(bin_edges)

    bin_centres = []
    bin_contents = []
    bin_means = []
    bin_sem = []

    for i in range(len(bin_edges) + 1):
        if i == 0:
            ind = X <= bin_edges[i]
            bin_centres.append(bin_edges[i] - max(bin_space) / 2)

        elif i == len(bin_edges):
            ind = X > bin_edges[-1]
            bin_centres.append(bin_edges[-1] + max(bin_space) / 2)

        else:
            ind = (X > bin_edges[i - 1]) & (X <= bin_edges[i])
            bin_centres.append((bin_edges[i - 1] + bin_edges[i]) / 2)

        bin_contents.append(Y[ind])
        bin_means.append(np.nanmean(bin_contents[i]))

        if bin_type == 'beta':
            alpha = np.sum(bin_contents[i] == 1)
            beta = np.sum(bin_contents[i] == 0)
            bin_sem.append(np.sqrt(alpha * beta / (alpha + beta) ** 2 / (alpha + beta + 1)))

        elif bin_type == 'std':
            bin_sem.append(np.nanstd(bin_contents[i]) / np.sqrt(np.sum(~np.isnan(bin_contents[i]))))

    return bin_means, bin_contents, bin_centres, bin_sem

def mean_sem(mat):
    mean = np.nanmean(mat, axis=0)
    sem = np.nanstd(mat, axis=0) / np.sqrt(mat.shape[0])
    return mean, sem


def plot_choice_curves_v2(ax, subjects, bin_edges, RTmin, RTmax, AZblue='black',
                          AZred='orange'):
    all_M_13_1 = []
    all_M_13_6 = []
    all_M_22_1 = []
    all_M_22_6 = []

    for sub in subjects:
        RT = sub['RT'][:, 4]
        mask = (RT > RTmin) & (RT < RTmax)

        dR = sub['o2'][mask, 3] - sub['o1'][mask, 3]
        A = sub['a'][mask, 4]

        n2 = sub['n2'][mask, 3]
        n1 = sub['n1'][mask, 3]

        game_length = sub['gameLength'][mask]

        i22 = n2 == 2
        i13 = ~i22
        i1 = game_length == 5
        i6 = game_length == 10

        dI = (n2 - n1) / 2
        uID = np.full_like(dI, np.nan)
        uID[dI > 0] = 1
        uID[dI < 0] = 2

        # For unequal [1 3]
        idx_13_1 = i13 & i1
        yvals = A[idx_13_1] == uID[idx_13_1]
        xvals = -dI[idx_13_1] * dR[idx_13_1]
        M_13_1, _, X, S_13_1 = bin_it(xvals, yvals, bin_edges)
        all_M_13_1.append(M_13_1)

        idx_13_6 = i13 & i6
        try:
            yvals = A[idx_13_6] == uID[idx_13_6]
            xvals = -dI[idx_13_6] * dR[idx_13_6]
            M_13_6, _, _, S_13_6 = bin_it(xvals, yvals, bin_edges)
        except Exception:
            M_13_6 = np.full(len(bin_edges) - 1, np.nan)
        all_M_13_6.append(M_13_6)

        # For equal [2 2]
        xvals_22_1 = dR[i22 & i1]
        yvals_22_1 = A[i22 & i1] == 2
        M_22_1, _, _, S_22_1 = bin_it(xvals_22_1, yvals_22_1, bin_edges)
        all_M_22_1.append(M_22_1)

        xvals_22_6 = dR[i22 & i6]
        yvals_22_6 = A[i22 & i6] == 2
        M_22_6, _, _, S_22_6 = bin_it(xvals_22_6, yvals_22_6, bin_edges)
        all_M_22_6.append(M_22_6)

    # Convert to arrays
    all_M_13_1 = np.vstack(all_M_13_1)
    all_M_13_6 = np.vstack(all_M_13_6)
    all_M_22_1 = np.vstack(all_M_22_1)
    all_M_22_6 = np.vstack(all_M_22_6)

    m_13_1, s_13_1 = mean_sem(all_M_13_1)
    m_13_6, s_13_6 = mean_sem(all_M_13_6)
    m_22_1, s_22_1 = mean_sem(all_M_22_1)
    m_22_6, s_22_6 = mean_sem(all_M_22_6)

    # Plot
    ax1, ax2 = ax
    ax1.set_title('unequal information [1 3]')
    ax2.set_title('equal information [2 2]')


    ax1.errorbar(X, m_13_1, yerr=s_13_1, fmt='.', color=AZblue, markersize=10, linestyle='--', label='Horizon 1')
    ax1.errorbar(X, m_13_6, yerr=s_13_6, fmt='.', color=AZred, markersize=10, linestyle='-', label='Horizon 6')
    ax1.set_xlabel('difference in means between\nmore and less informative option')
    ax1.set_ylabel('probability of choosing\nmore informative option')
    ax1.set_xlim([-35, 35])
    ax1.set_ylim([0, 1])
    plotting_utils.style_y_gridlines(ax1)
    ax2.errorbar(X, m_22_1, yerr=s_22_1, fmt='.', color=AZblue, markersize=10, linestyle='--', label='Horizon 1')
    ax2.errorbar(X, m_22_6, yerr=s_22_6, fmt='.', color=AZred, markersize=10, linestyle='-', label='Horizon 6')
    ax2.set_xlabel('difference in means between\nleft and right option')
    ax2.set_ylabel('probability of choosing\noption on the right')
    ax2.set_xlim([-35, 35])
    ax2.set_ylim([0, 1])
    plotting_utils.style_y_gridlines(ax2)

    return [ax1, ax2]

def plot_choice_curves_v2_h6(ax, subjects, bin_edges, RTmin, RTmax, AZblue='black',
                          AZred='orange'):
    all_M_13_1 = []
    all_M_13_6 = []
    all_M_22_1 = []
    all_M_22_6 = []

    for sub in subjects:
        RT = sub['RT'][:, 4]
        mask = (RT > RTmin) & (RT < RTmax)

        dR = sub['o2'][mask, 3] - sub['o1'][mask, 3]
        A = sub['a'][mask, 4]

        n2 = sub['n2'][mask, 3]
        n1 = sub['n1'][mask, 3]

        game_length = sub['gameLength'][mask]

        i22 = n2 == 2
        i13 = ~i22
        i1 = game_length == 5
        i6 = game_length == 10

        dI = (n2 - n1) / 2
        uID = np.full_like(dI, np.nan)
        uID[dI > 0] = 1
        uID[dI < 0] = 2

        # For unequal [1 3]
        idx_13_1 = i13 & i1
        yvals = A[idx_13_1] == uID[idx_13_1]
        xvals = -dI[idx_13_1] * dR[idx_13_1]
        M_13_1, _, X, S_13_1 = bin_it(xvals, yvals, bin_edges)
        all_M_13_1.append(M_13_1)

        idx_13_6 = i13 & i6
        try:
            yvals = A[idx_13_6] == uID[idx_13_6]
            xvals = -dI[idx_13_6] * dR[idx_13_6]
            M_13_6, _, _, S_13_6 = bin_it(xvals, yvals, bin_edges)
        except Exception:
            M_13_6 = np.full(len(bin_edges) - 1, np.nan)
        all_M_13_6.append(M_13_6)

        # For equal [2 2]
        xvals_22_1 = dR[i22 & i1]
        yvals_22_1 = A[i22 & i1] == 2
        M_22_1, _, _, S_22_1 = bin_it(xvals_22_1, yvals_22_1, bin_edges)
        all_M_22_1.append(M_22_1)

        xvals_22_6 = dR[i22 & i6]
        yvals_22_6 = A[i22 & i6] == 2
        M_22_6, _, _, S_22_6 = bin_it(xvals_22_6, yvals_22_6, bin_edges)
        all_M_22_6.append(M_22_6)

    # Convert to arrays
    all_M_13_1 = np.vstack(all_M_13_1)
    all_M_13_6 = np.vstack(all_M_13_6)
    all_M_22_1 = np.vstack(all_M_22_1)
    all_M_22_6 = np.vstack(all_M_22_6)

    m_13_1, s_13_1 = mean_sem(all_M_13_1)
    m_13_6, s_13_6 = mean_sem(all_M_13_6)
    m_22_1, s_22_1 = mean_sem(all_M_22_1)
    m_22_6, s_22_6 = mean_sem(all_M_22_6)

    # Plot
    ax1, ax2 = ax
    ax1.set_title('unequal information [1 3]')
    ax2.set_title('equal information [2 2]')


    #ax1.errorbar(X, m_13_1, yerr=s_13_1, fmt='.', color=AZblue, markersize=10, linestyle='none', label='Horizon 1')
    ax1.errorbar(X, m_13_6, yerr=s_13_6, fmt='.', color=AZred, markersize=10, linestyle='none', label='Horizon 6')
    ax1.set_xlabel('difference in means between\nmore and less informative option')
    ax1.set_ylabel('probability of choosing\nmore informative option')
    ax1.set_xlim([-35, 35])
    ax1.set_ylim([0, 1])
    plotting_utils.style_y_gridlines(ax1)
    #ax2.errorbar(X, m_22_1, yerr=s_22_1, fmt='.', color=AZblue, markersize=10, linestyle='none', label='Horizon 1')
    ax2.errorbar(X, m_22_6, yerr=s_22_6, fmt='.', color=AZred, markersize=10, linestyle='none', label='Horizon 6')
    ax2.set_xlabel('difference in means between\nleft and right option')
    ax2.set_ylabel('probability of choosing\noption on the right')
    ax2.set_xlim([-35, 35])
    ax2.set_ylim([0, 1])
    plotting_utils.style_y_gridlines(ax2)

    return [ax1, ax2]


def is_optimal_choice(row):
    if row['choice'] == 1 and row['m1'] > row['m2']:
        return True
    elif row['choice'] == 2 and row['m2'] > row['m1']:
        return True
    else:
        return False


def transform_wide_to_long(df_wilson):
    """
    Transforms the original wide-format DataFrame into a long-format DataFrame suitable for trial-level analysis.
    
    Parameters
    ----------
    df_wilson : pd.DataFrame
        The original wide-format DataFrame containing columns like 'c1', 'r1', ..., 'c10', 'r10'.
    
    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with one row per trial, including columns for choice, reward, and other relevant variables.
    """
    # Identify the columns that define each unique game/observation (the 'id' variables)
    id_cols = ['expt_name', 'subjectID', 'subjectNumber', 'block', 'game',
        'gameLength', 'uc', 'm1', 'm2', 'gID', 'repeatNumber']

    # Reshape from wide to long format
    # This will create 'trial' column and pivot 'c1', 'r1', etc., into 'c' and 'r' columns
    df_long = pd.wide_to_long(df_wilson,
                            stubnames=['c', 'r','rt'],  # Prefixes of the columns to pivot
                            i=id_cols,            # Columns to keep as identifiers for each observation
                            j='trial',            # Name of the new column for the trial number/index
                            sep='',               # Separator between stubname and number (e.g., 'c' and '1' in 'c1')
                            suffix=r'(\d+)')         # Pattern for the suffix (the numeric part like '1', '2')

    # Reset the index to make id_cols and 'trial' regular columns
    df_long = df_long.reset_index()

    # Rename the 'c' and 'r' columns to 'choice' and 'reward' for clarity
    df_long = df_long.rename(columns={'c': 'choice', 'r': 'reward', 'rt': 'reaction_time'})

    # Display the first few rows and columns of the transformed DataFrame
    print("Transformed DataFrame head:")
    print(df_long.head())
    print("\nTransformed DataFrame columns:")
    print(df_long.columns)
    print(f"\nShape of original DataFrame: {df_wilson.shape}")
    print(f"Shape of transformed DataFrame: {df_long.shape}")


    # Check for rows where 'choice' is NaN and 'reward' is not NaN
    inconsistent_rewards = df_long[df_long['choice'].isnull() & df_long['reward'].notnull()]

    # Display the number of such rows
    print(f"Number of rows where choice is NaN but reward is not NaN: {len(inconsistent_rewards)}")

    # Display the inconsistent rows if any
    if not inconsistent_rewards.empty:
        print("\nRows with inconsistent rewards:")
        print(inconsistent_rewards)

    df_long = df_long[df_long['trial'] <= df_long['gameLength']].copy()
    print("\nShape after filtering by gameLength:", df_long.shape)

    df_long=df_long.reset_index(inplace=False, drop=True)

    df_long[['choice','reward']] = df_long[['choice','reward']].astype('int')
    # z-score reaction times within each subject
    df_long['rt_z'] = df_long.groupby('subjectNumber')['reaction_time'].transform(lambda x: (x - x.mean()) / x.std())
    df_long.head()

    # add horizon column
    conditions = [
        (df_long['gameLength'] == 10),
        (df_long['gameLength'] == 5)
    ]
    choices = [6, 1]
    df_long['horizon'] = np.select(conditions, choices, default=0)
    # Drop the 'subjectID' column as it's redundant with 'subjectNumber'
    df_long = df_long.drop(columns=['subjectID'])

    # Rename 'subjectNumber' to 'participant_id' for clarity
    df_long.rename(columns={'subjectNumber': 'participant_id'}, inplace=True)
    # sum cumulative choices for each option
    df_long['n1']= (df_long['choice'] == 1).astype(int)
    df_long['n1']= df_long.groupby(['participant_id','game'])['n1'].cumsum()
    df_long['n2']= (df_long['choice'] == 2).astype(int)
    df_long['n2']= df_long.groupby(['participant_id','game'])['n2'].cumsum()

    # sum cumulative rewards for each option
    df_long["r1"] = (
        (df_long["reward"] * (df_long["choice"] == 1)).groupby([df_long["participant_id"], df_long["game"]])
        .cumsum()
    )
    df_long["r2"] = (
        (df_long["reward"] * (df_long["choice"] == 2)).groupby([df_long["participant_id"], df_long["game"]])
        .cumsum()
    )

    # calculate observed mean for each option, handling division by zero
    df_long['o1']=np.where(
        df_long['n1'] > 0,
        df_long['r1'] / df_long['n1'],
        0
    )
    df_long['o2']=np.where(
        df_long['n2'] > 0,
        df_long['r2'] / df_long['n2'],
        0
    )

    #add optimal choice column
    df_long['co'] = df_long.apply(is_optimal_choice, axis=1)

    # exclude long RTs on trial 5 (free trial) as these likely reflect missed responses rather than true choices
    invalid_rt = (df_long['trial'] == 5) & ((df_long['reaction_time'] < 0.1) | (df_long['reaction_time'] > 3.0))
    df_long.loc[invalid_rt, 'choice'] = np.nan

    return df_long

def significance_test(df):
    """
    Performs a one-sided binomial test for each participant to check if their
    proportion of 'optimal' choices is significantly greater than 0.5.

    Args:
        df (pd.DataFrame): DataFrame with at least 'participant_id' and 'co' columns,
                           where 'co' is 1 for an optimal choice and 0 otherwise.

    Returns:
        list: A list of participant IDs whose p-value from the binomial test
              is greater than 0.01 (indicating they should be excluded).
    """
    excluded_participants = []
    participants = df['participant_id'].unique()

    for p_id in participants:
        df_subj = df[df['participant_id'] == p_id].copy() # Use a copy to avoid SettingWithCopyWarning

        # Ensure 'co' column exists and is numeric
        if 'co' not in df_subj.columns:
            print(f"Warning: 'co' column not found for participant {p_id}. Skipping.")
            continue
        if not pd.api.types.is_numeric_dtype(df_subj['co']):
             print(f"Warning: 'co' column is not numeric for participant {p_id}. Skipping.")
             continue

        n_correct = df_subj['co'].sum()
        n_trials = df_subj['co'].count()

        # Only perform test if there are trials
        if n_trials > 0:
            # Run one-sided binomial test (H1: p > 0.5)
            # Check if k (n_correct) is within the bounds of n (n_trials)
            if 0 <= n_correct <= n_trials:
                result = binomtest(n_correct, n_trials, p=0.5, alternative='greater')

                # Check if p-value is > .01 → exclude participant
                if result.pvalue > 0.01:
                    excluded_participants.append(p_id)
            else:
                print(f"Warning: Invalid n_correct ({n_correct}) for n_trials ({n_trials}) for participant {p_id}. Skipping.")
        else:
            print(f"Warning: No trials found for participant {p_id}. Skipping.")


    return excluded_participants

def exclude_participants(df, excluded_ids):
    """
    Excludes participants with insignificant performance from the DataFrame based on a list of participant IDs.

    Args:
        df (pd.DataFrame): The original DataFrame containing a 'participant_id' column.
        excluded_ids (list): A list of participant IDs to be excluded.
        """
    excluded_ids = significance_test(df)
    print(f"Excluded participants (p > 0.01): {excluded_ids}")
    print(f"N excluded = {len(excluded_ids)}")
    df=df[df['participant_id'].isin(excluded_ids)==False].copy()
    print(f"Remaining participants after exclusion: {df['participant_id'].nunique()}")
#

def create_df_with_only_free_trials(df):
    """
    Creates a new DataFrame containing only the free-choice trials (trial 5) from the original DataFrame.

    Args:
        df (pd.DataFrame): The original DataFrame containing a 'trial' column.
    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where 'trial' equals 5.
    """
    free_trials_df = df[df['trial'] >= 5].copy()
    print(f"Created new DataFrame with only free-choice trials. Shape: {free_trials_df.shape}")
    return free_trials_df

def calculate_optimal_choices_per_subject(df):
    """
    Calculates the sum of optimal choices for each subject.

    Args:
        df (pd.DataFrame): DataFrame containing 'subjectID' and 'is_optimal' columns.

    Returns:
        pd.Series: A Series with the sum of optimal choices per subjectID.
    """
    return df.groupby('participant_id')['co'].sum()

def participant_summary(df):
    """
    Summarizes the number of participants and their optimal choices.

    Args:
        df (pd.DataFrame): DataFrame containing 'subjectID' and 'is_optimal' columns.

    Returns:
        pd.DataFrame: A DataFrame with the number of participants and their optimal choices.
    """
    summary = df.groupby('participant_id')['co'].agg(['count', 'sum']).reset_index()
    return summary

def accuracy_per_subject(df):
    """
    Calculates the accuracy per subject.

    Args:
        df (pd.DataFrame): DataFrame containing 'subjectID' and 'is_optimal' columns.

    Returns:
        pd.Series: A Series with the accuracy per subjectID.
    """
    accuracy = df.groupby('participant_id')['co'].mean()
    return accuracy


def accuracy_per_horizon(df):
    """
    Calculates the accuracy per horizon.

    Args:
        df (pd.DataFrame): DataFrame containing 'horizon' and 'is_optimal' columns.

    Returns:
        pd.Series: A Series with the accuracy per horizon.
    """
    accuracy = df.groupby('horizon')['co'].mean()
    return accuracy

def accuracy_per_horizon_and_subject(df):
    """
    Calculates the accuracy per horizon and subject.

    Args:
        df (pd.DataFrame): DataFrame containing 'horizon', 'subjectID', and 'is_optimal' columns.

    Returns:
        pd.DataFrame: A DataFrame with the accuracy per horizon and subject.
    """
    accuracy = df.groupby(['horizon', 'participant_id'])['co'].mean().reset_index()
    return accuracy


def trial_accuracy(df):
    """
    Calculates the accuracy per trial.

    Args:
        df (pd.DataFrame): DataFrame containing 'trial' and 'is_optimal' columns.

    Returns:
        pd.Series: A Series with the accuracy per trial.
    """
    # check whether all columns are present
    model_id = 'participant_id' if 'participant_id' in df.columns else 'subjectNumber'
    required_columns = {'trial', 'co', model_id}
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
    # First compute mean accuracy per trial and model
    trial_model_acc = df.groupby([model_id, 'trial'])['co'].mean().reset_index()

    # Then compute mean and SE across model_ids
    accuracy = trial_model_acc.groupby('trial')['co'].mean()
    se = trial_model_acc.groupby('trial')['co'].sem()

    accuracy = pd.DataFrame({'accuracy': accuracy, 'se': se})
    return accuracy


def plot_trial_accuracy(df, title,ylim=(0.65, 0.95)):
    """
    Plots the trial accuracy.

    Args:
        df (pd.DataFrame): DataFrame containing 'trial' and 'is_optimal' columns.
        title (str): Title for the plot.
    """
    horizon_1= df[df['horizon'] == 1]
    horizon_6= df[df['horizon'] == 6]
    accuracy_h1= trial_accuracy(horizon_1)
    accuracy_h6= trial_accuracy(horizon_6)

    plt.errorbar(
        accuracy_h1.index,
        accuracy_h1['accuracy'],
        yerr=accuracy_h1['se'],
        fmt='o-', capsize=3, label='Horizon 1',
        color='black'
    )
    plt.errorbar(
        accuracy_h6.index,
        accuracy_h6['accuracy'],
        yerr=accuracy_h6['se'],
        fmt='o-', capsize=3, label='Horizon 6',
        color='orange'
    )
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("Free-choice trial number")
    plt.ylabel("Fraction optimal")
    plt.title(f"Learning Curve by Free-Choice Trial Number for {title}")
    plt.ylim(ylim)
    plt.legend()
    plotting_utils.style_y_gridlines(plt.gca())
    plt.tight_layout()
    plt.show()




def rename_columns_for_plotting(df):
    """
    Renames columns in the DataFrame for consistency in plotting.

    Args:
        df (pd.DataFrame): The original DataFrame with columns to be renamed.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
    """
    if 'trial_num_block' in df.columns:
        df.rename(columns={'trial_num_block': 'trial'}, inplace=True)
    if 'trial_num' in df.columns:
        df.rename(columns={'trial_num': 'trial'}, inplace=True)
    if 'reward_mean_H' in df.columns:
        df.rename(columns={'reward_mean_H': 'm1'}, inplace=True)
    if 'reward_mean_I' in df.columns:
        df.rename(columns={'reward_mean_I': 'm2'}, inplace=True)
    return df

def map_choices_to_numeric(row):
    """
    Maps choice values from 'H' and 'I' to numeric values 1 and 2.

    Args:
        row (pd.Series): A row of the DataFrame containing a 'choice' column

    Returns:
        pd.DataFrame: The DataFrame with the 'choice' column mapped to numeric values.
    """
    # choice is a string column with values 'H' and 'I', we want to map these to 1 and 2
    if row['choice'] == 'H':
        return 1
    elif row['choice'] == 'I':
        return 2
    else:
        return np.nan  # or you could raise an error if you expect only 'H' and 'I'
    # print how many choices were mapped to 1 and how many to 2 and how many were unmapped



# ### Accuracy plot
# prompt: i want function like plot trial accuracy that plots human_df as red and blue lines with alpha=0.3 and above centaur performance
def plot_trial_accuracy_with_centaur(human_df, centaur_df, title, ylim=(0.4, 1.0),label='Centaur',AZblue='black',AZred='orange'):
    """
    Plots the trial accuracy for human data and overlays centaur performance.

    Args:
        human_df (pd.DataFrame): DataFrame containing human data with 'trial',
                                 'horizon', and 'co' columns.
        centaur_df (pd.DataFrame): DataFrame containing centaur data with 'trial',
                                   'horizon', and 'co' columns.
        title (str): Title for the plot.
        ylim (tuple): Y-axis limits for the plot.
    """
    plt.figure(figsize=(16, 12))
    # Calculate accuracy for human data per horizon and trial
    human_h1 = human_df[human_df['horizon'] == 1]
    human_h6 = human_df[human_df['horizon'] == 6]
    accuracy_h1 = trial_accuracy(human_h1)
    accuracy_h6 = trial_accuracy(human_h6)

    # Plot human data
    plt.errorbar(
        accuracy_h1.index,
        accuracy_h1['accuracy'],
        yerr=accuracy_h1['se'],
        fmt='o', capsize=3, label='Horizon 1',
        color=AZblue
    )
    plt.errorbar(
        accuracy_h6.index,
        accuracy_h6['accuracy'],
        yerr=accuracy_h6['se'],
        fmt='o', capsize=3, label='Horizon 6',
        color=AZred
    )

    # Calculate accuracy for centaur data per horizon and trial
    centaur_h1 = centaur_df[centaur_df['horizon'] == 1]
    centaur_h6 = centaur_df[centaur_df['horizon'] == 6]
    accuracy_centaur_h1 = trial_accuracy(centaur_h1)
    accuracy_centaur_h6 = trial_accuracy(centaur_h6)

    # Plot centaur data
    plt.plot(
        accuracy_centaur_h1.index,
        accuracy_centaur_h1['accuracy'], linestyle='-', label=f'{label}(Horizon 1)',marker='s',
        color=AZblue,
        alpha=1
    )
    plt.plot(
        accuracy_centaur_h6.index,
        accuracy_centaur_h6['accuracy'],
        linestyle='-', label=f'{label}(Horizon 6)',
        color=AZred,
        alpha=1
    )

    plt.xlabel("free-choice trial number")
    plt.ylabel("accuracy (optimal choice rate)")
    plt.title(f"Learning Curve: {title}")
    plt.ylim(ylim)
    plt.legend(loc='upper left')
    plotting_utils.style_y_gridlines(plt.gca())
    plt.tight_layout()
    plt.show()

# Assuming 'free_trials' contains the relevant human data
# and 'centaur_free_df' contains the relevant centaur data


def plot_trial_accuracy_with_centaur_margin(human_df, centaur_df, title, ylim=(0.4, 1.0)):
    """
    Plots the trial accuracy for Horizon 6 human data with error bars and overlays
    centaur performance with a standard error margin.

    Args:
        human_df (pd.DataFrame): Human data with 'trial', 'horizon', and 'co' columns.
        centaur_df (pd.DataFrame): Centaur data with 'trial', 'horizon', and 'co' columns.
        title (str): Plot title.
        ylim (tuple): Y-axis limits.
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    # Filter human df to first 100 games to match llms amount of games
    human_df = human_df[human_df['game'] <= 100]
    # Filter and compute Horizon 6 only
    human_h6 = human_df[human_df['horizon'] == 6]
    accuracy_h6 = trial_accuracy(human_h6)

    ax.errorbar(
        accuracy_h6.index,
        accuracy_h6['accuracy'],
        yerr=accuracy_h6['se'],
        fmt='o-', capsize=3, label='Human Horizon 6',
        color='orange', alpha=0.7
    )

    centaur_h6 = centaur_df[centaur_df['horizon'] == 6]
    accuracy_centaur_h6 = trial_accuracy(centaur_h6)

    ax.plot(
        accuracy_centaur_h6.index,
        accuracy_centaur_h6['accuracy'],
        marker='x', linestyle='--', label='Centaur Horizon 6',
        color='orange'
    )
    ax.fill_between(
        accuracy_centaur_h6.index,
        accuracy_centaur_h6['accuracy'] - accuracy_centaur_h6['se'],
        accuracy_centaur_h6['accuracy'] + accuracy_centaur_h6['se'],
        color='orange', alpha=0.2
    )

    ax.set_xlabel("Free-choice trial number")
    ax.set_ylabel("Fraction optimal")
    ax.set_title(f"Learning Curve (Horizon 6): {title}")
    ax.set_ylim(ylim)
    ax.legend()
    plotting_utils.style_y_gridlines(ax)
    fig.tight_layout()
    plt.show()

class HandlerLineWithFill(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        center_y = y0 + height / 2
        fill_height = height * 0.8  # Increase fill height (was 0.5 before)

        # Shaded fill (larger vertical margin)
        patch = Rectangle((x0, center_y - fill_height / 2), width, fill_height,
                          facecolor=orig_handle.get_facecolor(), edgecolor='none', alpha=0.3,
                          transform=trans)

        # Line across the center
        line = Line2D([x0, x0 + width], [center_y, center_y],
                      linestyle='-', color=orig_handle.get_edgecolor(), linewidth=2,
                      transform=trans)

        return [patch, line]

class HandlerErrorbar(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        center_x = x0 + width / 2
        center_y = y0 + height / 2

        # Marker
        marker = Line2D([center_x], [center_y], marker='s', color=orig_handle.get_color(),
                        markersize=8, linestyle='None', transform=trans)

        # Vertical error bar
        err_line = Line2D([center_x, center_x], [center_y - height/2, center_y + height/2],
                          color=orig_handle.get_color(), linestyle='-', linewidth=1.5, transform=trans)

        return [err_line, marker]


def plot_trial_accuracy_with_multiple_centaur_models(human_df, centaur_dfs=None, labels=None,
                                                      title=None, ylim=(0.65, 0.9),
                                                      centaur_colors=None,
                                                      yticks=None, xticks=None,
                                                      show_legend=False, legend_labels=[ 'Human (H6)','Centaur-70B(H6)', 'Llama-Instruct-3.1-70B(H6)']):
    """
    Plots human trial accuracy for Horizon 6 with overlay of multiple Centaur models.

    Args:
        human_df (pd.DataFrame): Human data with 'trial', 'horizon', and 'co' columns.
        centaur_dfs (list of pd.DataFrame): List of Centaur model DataFrames.
        labels (list of str): List of labels for each Centaur model.
        title (str): Title of the plot.
        ylim (tuple): Y-axis range.
        centaur_colors (list of str): List of colors for Centaur models.
        show_legend (bool): Whether to draw legend below the plot. Legend figure always saved.
        legend_labels (list of str): Custom legend labels; if None, uses auto-detected labels.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotting_utils.set_dynamic_fontsize(fig_width=FIGSIZE[0], base_font=BASE_FONT)
    bar_w = 0.25
    # take first 100 games from human df
    human_df = human_df[human_df['game'] <= 100]

    # --- Human Data: Horizon 6 ---
    human_h6 = human_df[human_df['horizon'] == 6]
    acc_h6 = trial_accuracy(human_h6)
    ax.plot(range(1, len(acc_h6) + 1), acc_h6['accuracy'], linestyle='-',
                 label='Human (H6)', color=centaur_colors[0], alpha=0.5, zorder=1)
    ax.fill_between(range(1, len(acc_h6) + 1),
                         acc_h6['accuracy'] - acc_h6['se'],
                         acc_h6['accuracy'] + acc_h6['se'],
                         color=centaur_colors[0], alpha=0.1)

    # --- Centaur Models ---
    if centaur_colors is None:
        from matplotlib.cm import get_cmap
        cmap = get_cmap('tab10')
        centaur_colors = [cmap(i) for i in range(len(centaur_dfs))]
    if centaur_dfs is not None:
        for df, label, color in zip(centaur_dfs, labels, centaur_colors[1:]):
            print(f"Mapped to Nan in {label}: {(df['choice'].isna()).sum()}")
            # what percentages of choices are None
            print(f"Percentages of None choices in {label}: {(df['choice'].isna()).sum() / len(df) * 100:.2f}%")
            df_h6 = df[df['horizon'] == 6]
            acc_h6 = trial_accuracy(df_h6)
            print(f"{label} Horizon 6 accuracy: {acc_h6['accuracy'].values}")
            ax.plot(range(1, len(acc_h6) + 1), acc_h6['accuracy'], linestyle='-',
                    color=color, alpha=1, label=f'{label} (H6)', zorder=2)
            ax.fill_between(range(1, len(acc_h6) + 1),
                            acc_h6['accuracy'] - acc_h6['se'],
                            acc_h6['accuracy'] + acc_h6['se'],
                            color=color, alpha=0.2)

    # --- Plot Decoration ---
    ax.set_xlabel("Trial",labelpad=10)
    ax.set_ylabel("Accuracy rate")
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_yticks(yticks if yticks is not None else [0.6, 0.7, 0.8, 0.9, 1.0])
    ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
    #ax.set_title(f"{title}")
    ax.set_ylim(ylim)
    transform=ax.get_xaxis_transform()
    ax.tick_params(axis='x', which='major', pad=15)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_visible(True)
    plotting_utils.style_y_gridlines(ax)
    #plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    # Horizon 6 line with shaded margin

    # --- Legend below x-axis label (2 per row) ---
    handles, auto_labels = ax.get_legend_handles_labels()
    display_labels = legend_labels if legend_labels is not None else auto_labels

    if show_legend:
        leg = ax.legend(handles, display_labels,
                        loc='upper center', bbox_to_anchor=(0.5, -0.12),
                        ncols=2, frameon=False)
        for line in leg.get_lines():
            line.set_linewidth(4)

    # --- Legend figure saved separately (always) ---
    legend_fig = plt.figure(figsize=(6, 2))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")
    if show_legend:
        leg2 = legend_ax.legend(handles, display_labels, loc='center', ncols=2, frameon=False)
        for line in leg2.get_lines():
            line.set_linewidth(4)

    ax.margins(x=0.04)
    plt.tight_layout()
    return fig, legend_fig


def plot_fifth_trial_accuracy_bar(dfs, labels=None, colors=None,
                                   family_labels=None, family_slices=None,
                                   title="Accuracy on Trial 5 (Horizon 1 vs 6)"):
    n = len(dfs)
    w = 0.5  # bar width
    gap_in  = 0.2
    gap_out = 0.5

    xpos = []
    current_pos = 0
    for i in range(n):
        xpos.append(current_pos)
        current_pos += w*2 + gap_in

    acc_h1_vals, acc_h1_se = [], []
    acc_h6_vals, acc_h6_se = [], []

    for df in dfs:
        model_id = 'participant_id' if 'participant_id' in df.columns else 'subjectNumber'
        h1 = df[(df['horizon'] == 1) & (df['trial'] == 5)]
        h6 = df[(df['horizon'] == 6) & (df['trial'] == 5)]
        h1_subj = h1.groupby(model_id)['co'].mean()
        h6_subj = h6.groupby(model_id)['co'].mean()
        acc_h1_vals.append(h1_subj.mean())
        acc_h1_se.append(h1_subj.sem())
        acc_h6_vals.append(h6_subj.mean())
        acc_h6_se.append(h6_subj.sem())
        lbl = labels[len(acc_h1_vals)-1] if labels else len(acc_h1_vals)-1
        print(f"{lbl} | H1: n={len(h1_subj)}, mean={h1_subj.mean():.4f}, sem={h1_subj.sem():.4f} | H6: n={len(h6_subj)}, mean={h6_subj.mean():.4f}, sem={h6_subj.sem():.4f}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotting_utils.set_dynamic_fontsize(fig_width=FIGSIZE[0], base_font=BASE_FONT)

    xpos = np.array(xpos)

    ax.bar(xpos - w/2, acc_h1_vals, w, yerr=acc_h1_se,
           label='Horizon 1', color=colors, capsize=5)
    ax.bar(xpos + w/2, acc_h6_vals, w, yerr=acc_h6_se,
           label='Horizon 6', color=colors, capsize=5,hatch='//',alpha=0.6)

    ax.set_ylabel('Accuracy rate on trial 1')
    ax.set_ylim(0.65, 0.85)
    # set y-ticks from 0 to 1 with step of 0.1
    ax.set_yticks(np.arange(0.6, 0.85, 0.1))
    ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
    #ax.set_title(title, pad=20)
    ax.set_xticks(xpos)
    ax.set_xticklabels([])
    #ax.set_xticklabels(labels)
    for i, family_slice in enumerate(family_slices):
        if isinstance(family_slice, tuple):
            i0, i1 = family_slice
            center = ((xpos[i0] + xpos[i1]) / 2)
        else:
            center = xpos[family_slice]
        ax.text(center, -0.10, family_labels[i],
                ha='center', va='top',
                transform=ax.get_xaxis_transform())

    # Visual group dividers
    for i, family_slice in enumerate(family_slices[:-1]):
        if isinstance(family_slice, tuple):
            i0, i1 = family_slice
    
    #change legend color to white
    legend_handles = [
        Patch(facecolor='white', alpha=0.9, label='Horizon 1', edgecolor='black', linewidth=0.8),
        Patch(facecolor='white', alpha=0.9, hatch='//', label='Horizon 5', edgecolor='black', linewidth=0.8),
    ]
    ax.legend(handles=legend_handles, loc='upper right', ncol=1, frameon=False,
              columnspacing=1.2, handletextpad=0.5)
    plotting_utils.remove_bar_frame(ax)
    plotting_utils.style_y_gridlines(ax)
    plt.tight_layout()
    return fig



def reshape_llm_df(llm_df):
    """
    Reshapes the centaur DataFrame from long to wide format.

    Args:
        centaur_df (pd.DataFrame): The input DataFrame in long format.

    Returns:
        pd.DataFrame: The reshaped DataFrame in wide format.
    """
    llm_df = llm_df.copy()
    # Normalise column names that may differ across data sources
    if 'trial_num' in llm_df.columns and 'trial' not in llm_df.columns:
        llm_df = llm_df.rename(columns={'trial_num': 'trial'})
    if 'model_id' in llm_df.columns and 'participant_id' not in llm_df.columns:
        llm_df = llm_df.rename(columns={'model_id': 'participant_id'})
    if 'block' not in llm_df.columns:
        llm_df['block'] = 1

    # Ensure trials are ordered
    llm_df_wide = llm_df.sort_values(by=['participant_id', 'block', 'game', 'trial'])

    # Add subjectNumber if needed
    llm_df_wide['subjectNumber'] = llm_df_wide['participant_id']

    # Create a unique game ID if needed
    llm_df_wide['gID'] = llm_df_wide.groupby(['participant_id', 'block', 'game']).ngroup()

    # Pivot to wide format for choices and rewards.
    # reset_index() moves the MultiIndex back to regular columns so the subsequent
    # merge(on=[...]) works cleanly without creating duplicate column names.
    _key_cols = ['participant_id', 'subjectNumber', 'block', 'game']

    choice_wide = llm_df_wide.pivot_table(
        index=_key_cols, columns='trial', values='choice', aggfunc='first'
    ).reset_index()
    choice_wide.columns.name = None
    choice_wide = choice_wide.rename(
        columns={c: f'c{int(c)}' for c in choice_wide.columns if c not in _key_cols}
    )

    reward_wide = llm_df_wide.pivot_table(
        index=_key_cols, columns='trial', values='reward', aggfunc='first'
    ).reset_index()
    reward_wide.columns.name = None
    reward_wide = reward_wide.rename(
        columns={c: f'r{int(c)}' for c in reward_wide.columns if c not in _key_cols}
    )

    # Pad missing trial columns so horizon-1 (5-trial) and horizon-6 (10-trial) games
    # always share the same schema (NaN for unused slots).
    for i in range(1, 11):
        if f'c{i}' not in choice_wide.columns:
            choice_wide[f'c{i}'] = np.nan
        if f'r{i}' not in reward_wide.columns:
            reward_wide[f'r{i}'] = np.nan

    # Keep one row per game with game-level variables
    game_info = llm_df_wide.groupby(['participant_id', 'subjectNumber', 'block', 'game'], as_index=False).first()
    cols_keep = ['participant_id', 'subjectNumber', 'block', 'game', 'horizon', 'info_condition', 'm1', 'm2', 'gID']
    game_info = game_info[cols_keep]

    # Merge with wide trial data
    llm_df_reshaped = game_info.merge(choice_wide, on=['participant_id', 'subjectNumber', 'block', 'game'])
    llm_df_reshaped = llm_df_reshaped.merge(reward_wide, on=['participant_id', 'subjectNumber', 'block', 'game'])

    # Rename columns to match df_2
    llm_df_reshaped = llm_df_reshaped.rename(columns={
        'participant_id': 'subjectID',
        'horizon': 'gameLength',
        'info_condition': 'uc'
    })
    # horizon values are 1/6 but process_subjects_llm expects gameLength 5/10
    if llm_df_reshaped['gameLength'].isin([1, 6]).any():
        llm_df_reshaped['gameLength'] = llm_df_reshaped['gameLength'].map({1: 5, 6: 10})

    # Reorder columns to match df_2
    cols_order = ['subjectID', 'subjectNumber', 'block', 'game', 'gameLength', 'uc', 'm1', 'm2', 'gID'] + \
                 [f'r{i}' for i in range(1, 11)] + [f'c{i}' for i in range(1, 11)]

    llm_df_reshaped = llm_df_reshaped[cols_order]

    return llm_df_reshaped


def process_subjects_llm(df):
    subjects = []
    subject_ids = df['subjectNumber'].unique()

    for sn in subject_ids:
        sub_df = df[df['subjectNumber'] == sn].reset_index(drop=True)
        sub = {}

        # Basic info
        #sub['expt_name'] = sub_df.loc[0, 'expt_name']
        sub['subjectID'] = sub_df.loc[0, 'subjectID']
        sub['subjectNumber'] = sn

        # Columns
        sub['block'] = sub_df['block'].values
        sub['game'] = sub_df['game'].values
        sub['gameLength'] = sub_df['gameLength'].values
        sub['uc'] = sub_df['uc'].values
        sub['m1'] = sub_df['m1'].values
        sub['m2'] = sub_df['m2'].values
        sub['gID'] = sub_df['gID'].values
        #sub['repeatNumber'] = sub_df['repeatNumber'].values

        r = sub_df[[f'r{i}' for i in range(1, 11)]].to_numpy(dtype=float)
        a = sub_df[[f'c{i}' for i in range(1, 11)]].to_numpy(dtype=float)
        #RT = sub_df[[f'rt{i}' for i in range(1, 11)]].to_numpy()

        sub['r'] = r
        sub['a'] = a
        #sub['RT'] = RT

        # z-scored RT
        #sub['RTz'] = (RT - np.nanmean(RT)) / np.nanstd(RT)

        # Cumulative choices
        sub['n1'] = np.cumsum(a == 1, axis=1)
        sub['n2'] = np.cumsum(a == 2, axis=1)

        # Cumulative rewards
        sub['R1'] = np.cumsum(r * (a == 1), axis=1)
        sub['R2'] = np.cumsum(r * (a == 2), axis=1)

        # Observed mean — element-wise safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            sub['o1'] = np.where(sub['n1'] > 0,
                                 sub['R1'] / np.where(sub['n1'] > 0, sub['n1'], 1),
                                 0.0)
            sub['o2'] = np.where(sub['n2'] > 0,
                                 sub['R2'] / np.where(sub['n2'] > 0, sub['n2'], 1),
                                 0.0)
    

        # Correct choice
        m1 = sub['m1'][:, np.newaxis]
        m2 = sub['m2'][:, np.newaxis]
        co = ((m1 > m2) * (a == 1)) + ((m1 < m2) * (a == 2))
        co[np.isnan(a)] = np.nan
        sub['co'] = co

        # Low observed mean (random exploration)
        o1 = sub['o1']
        o2 = sub['o2']
        lm = ((o1[:, :-1] < o2[:, :-1]) * (a[:, 1:] == 1)) + \
             ((o1[:, :-1] > o2[:, :-1]) * (a[:, 1:] == 2))
        lm[o1[:, :-1] == o2[:, :-1]] = np.nan
        lm[np.isnan(a[:, 1:])] = np.nan
        lm_shifted = np.full_like(a, np.nan)
        lm_shifted[:, 1:] = lm
        sub['lm'] = lm_shifted

        # High info (directed exploration)
        n1 = sub['n1']
        n2 = sub['n2']
        hi = ((n1[:, :-1] < n2[:, :-1]) * (a[:, 1:] == 1)) + \
             ((n1[:, :-1] > n2[:, :-1]) * (a[:, 1:] == 2))
        hi[n1[:, :-1] == n2[:, :-1]] = np.nan
        hi[np.isnan(a[:, 1:])] = np.nan
        hi_shifted = np.full_like(a, np.nan)
        hi_shifted[:, 1:] = hi
        sub['hi'] = hi_shifted

        # Repetition (same choice as previous)
        rep = np.full_like(a, np.nan)
        rep[:, 1:] = (a[:, 1:] == a[:, :-1])
        sub['rep'] = rep

        # Trial 5 features
        sub['dR'] = sub['o2'][:, 3] - sub['o1'][:, 3]
        sub['dI'] = -(sub['n2'][:, 3] - sub['n1'][:, 3]) / 2
        sub['choice'] = a[:, 4]
        #sub['rt'] = RT[:, 4]

        subjects.append(sub)

    return subjects


def transform_row_into_int(subjects_list, column_name):

    for sub in range(len(subjects_list)):
        subjects_list[sub][column_name]=subjects_list[sub][column_name].astype(int)
    return subjects_list


def plot_model_lines(ax, model_subjects, bin_edges,AZblue='black',AZred='orange',label='model'):
    all_M_13_1 = []
    all_M_13_6 = []
    all_M_22_1 = []
    all_M_22_6 = []

    for sub in model_subjects:
        dR = sub['o2'][:, 3] - sub['o1'][:, 3]
        A = sub['a'][:, 4]

        n2 = sub['n2'][:, 3]
        n1 = sub['n1'][:, 3]

        game_length = sub['gameLength']

        i22 = n2 == 2
        i13 = ~i22
        i1 = game_length == 5
        i6 = game_length == 10

        dI = (n2 - n1) / 2
        uID = np.full_like(dI, np.nan)
        uID[dI > 0] = 1
        uID[dI < 0] = 2

        # For unequal [1 3]
        idx_13_1 = i13 & i1
        #print(i13)
        #print(i1)
        yvals = A[idx_13_1] == uID[idx_13_1]
        xvals = -dI[idx_13_1] * dR[idx_13_1]
        M_13_1,_,X, S_13_1 = bin_it(xvals, yvals, bin_edges)
        all_M_13_1.append(M_13_1)

        idx_13_6 = i13 & i6
        try:
            yvals = A[idx_13_6] == uID[idx_13_6]
            xvals = -dI[idx_13_6] * dR[idx_13_6]
            M_13_6,_,_,S_13_6 = bin_it(xvals, yvals, bin_edges)
        except Exception:
            M_13_6 = np.full(len(bin_edges) - 1, np.nan)
        all_M_13_6.append(M_13_6)

        # For equal [2 2]
        xvals_22_1 = dR[i22 & i1]
        yvals_22_1 = A[i22 & i1] == 2
        M_22_1,_,_, S_22_1= bin_it(xvals_22_1, yvals_22_1, bin_edges)
        all_M_22_1.append(M_22_1)

        xvals_22_6 = dR[i22 & i6]
        yvals_22_6 = A[i22 & i6] == 2
        M_22_6, _,_,S_22_6 = bin_it(xvals_22_6, yvals_22_6, bin_edges)
        all_M_22_6.append(M_22_6)

    # Convert to arrays
    all_M_13_1 = np.vstack(all_M_13_1)
    all_M_13_6 = np.vstack(all_M_13_6)
    all_M_22_1 = np.vstack(all_M_22_1)
    all_M_22_6 = np.vstack(all_M_22_6)
    m_13_1,s13_1=mean_sem(all_M_13_1)
    m_13_6,s13_6=mean_sem(all_M_13_6)
    m_22_1,s22_1=mean_sem(all_M_22_1)
    m_22_6,s22_6=mean_sem(all_M_22_6)

    ax1, ax2 = ax
    # Overlay model lines with shaded error bands
    ax1.plot(X, m_13_1, color=AZblue, linestyle='--', label=f'{label}(Horizon 1)')
    ax1.fill_between(X, m_13_1 - s13_1, m_13_1 + s13_1, color=AZblue, alpha=0.2)

    ax1.plot(X, m_13_6, color=AZred, linestyle='-', label=f'{label}(Horizon 6)')
    ax1.fill_between(X, m_13_6 - s13_6, m_13_6 + s13_6, color=AZred, alpha=0.2)

    ax2.plot(X, m_22_1, color=AZblue, linestyle='--', label=f'{label}(Horizon 1)')
    ax2.fill_between(X, m_22_1 - s22_1, m_22_1 + s22_1, color=AZblue, alpha=0.2)

    ax2.plot(X, m_22_6, color=AZred, linestyle='solid', label=f'{label}(Horizon 6)')
    ax2.fill_between(X, m_22_6 - s22_6, m_22_6 + s22_6, color=AZred, alpha=0.2)

    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))


def plot_model_lines_h6(ax, model_subjects, bin_edges,AZblue='black',AZred='orange',label='model'):
    all_M_13_1 = []
    all_M_13_6 = []
    all_M_22_1 = []
    all_M_22_6 = []

    for sub in model_subjects:
        dR = sub['o2'][:, 3] - sub['o1'][:, 3]
        A = sub['a'][:, 4]

        n2 = sub['n2'][:, 3]
        n1 = sub['n1'][:, 3]

        game_length = sub['gameLength']

        i22 = n2 == 2
        i13 = ~i22
        i1 = game_length == 5
        i6 = game_length == 10

        dI = (n2 - n1) / 2
        uID = np.full_like(dI, np.nan)
        uID[dI > 0] = 1
        uID[dI < 0] = 2

        # For unequal [1 3]
        idx_13_1 = i13 & i1
        #print(i13)
        #print(i1)
        yvals = A[idx_13_1] == uID[idx_13_1]
        xvals = -dI[idx_13_1] * dR[idx_13_1]
        M_13_1,_,X, S_13_1 = bin_it(xvals, yvals, bin_edges)
        all_M_13_1.append(M_13_1)

        idx_13_6 = i13 & i6
        try:
            yvals = A[idx_13_6] == uID[idx_13_6]
            xvals = -dI[idx_13_6] * dR[idx_13_6]
            M_13_6,_,_,S_13_6 = bin_it(xvals, yvals, bin_edges)
        except Exception:
            M_13_6 = np.full(len(bin_edges) - 1, np.nan)
        all_M_13_6.append(M_13_6)

        # For equal [2 2]
        xvals_22_1 = dR[i22 & i1]
        yvals_22_1 = A[i22 & i1] == 2
        M_22_1,_,_, S_22_1= bin_it(xvals_22_1, yvals_22_1, bin_edges)
        all_M_22_1.append(M_22_1)

        xvals_22_6 = dR[i22 & i6]
        yvals_22_6 = A[i22 & i6] == 2
        M_22_6, _,_,S_22_6 = bin_it(xvals_22_6, yvals_22_6, bin_edges)
        all_M_22_6.append(M_22_6)

    # Convert to arrays
    all_M_13_1 = np.vstack(all_M_13_1)
    all_M_13_6 = np.vstack(all_M_13_6)
    all_M_22_1 = np.vstack(all_M_22_1)
    all_M_22_6 = np.vstack(all_M_22_6)
    m_13_1,s13_1=mean_sem(all_M_13_1)
    m_13_6,s13_6=mean_sem(all_M_13_6)
    m_22_1,s22_1=mean_sem(all_M_22_1)
    m_22_6,s22_6=mean_sem(all_M_22_6)

    ax1, ax2 = ax
    # Overlay model lines with shaded error bands
    #ax1.plot(X, m_13_1, color=AZblue, linestyle='--', label=f'{label}(Horizon 1)')
    #ax1.fill_between(X, m_13_1 - s13_1, m_13_1 + s13_1, color=AZblue, alpha=0.2)

    ax1.plot(X, m_13_6, color=AZred, linestyle='-', label=f'{label}(Horizon 6)')
    ax1.fill_between(X, m_13_6 - s13_6, m_13_6 + s13_6, color=AZred, alpha=0.2)

    #ax2.plot(X, m_22_1, color=AZblue, linestyle='--', label=f'{label}(Horizon 1)')
    #ax2.fill_between(X, m_22_1 - s22_1, m_22_1 + s22_1, color=AZblue, alpha=0.2)

    ax2.plot(X, m_22_6, color=AZred, linestyle='solid', label=f'{label}(Horizon 6)')
    ax2.fill_between(X, m_22_6 - s22_6, m_22_6 + s22_6, color=AZred, alpha=0.2)

    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))





def build_free_trial_dfs(df_pilot, model_dfs, model_color_map, human_color):
    """
    Build long-format free-trial DataFrames for human and all models.

    Returns
    -------
    tuple
        (human_free, model_free_dfs, model_labels, model_plot_colors)
    """
    df_long_human = transform_wide_to_long(df_pilot)
    human_free = create_df_with_only_free_trials(df_long_human)

    model_free_dfs = []
    model_labels = []
    model_plot_colors = [human_color]  # index-0 used for human by the plot function

    for mname, mdf in model_dfs.items():
        if mdf.empty:
            continue
        mdf2 = mdf.copy()
        if 'trial_num' in mdf2.columns and 'trial' not in mdf2.columns:
            mdf2 = mdf2.rename(columns={'trial_num': 'trial'})
        if 'model_id' in mdf2.columns and 'participant_id' not in mdf2.columns:
            mdf2 = mdf2.rename(columns={'model_id': 'participant_id'})
        if mdf2['choice'].dtype == object:
            mdf2['choice'] = mdf2['choice'].map({'H': 1, 'I': 2})
        # Diagnostics: check if m1/m2 are present and non-NaN
        if 'm1' in mdf2.columns and 'm2' in mdf2.columns:
            nan_m1 = mdf2['m1'].isna().sum()
            nan_m2 = mdf2['m2'].isna().sum()
            print(f"\n[diag] {mname}: m1 NaN={nan_m1}/{len(mdf2)}, m2 NaN={nan_m2}/{len(mdf2)}, choices={sorted(mdf2['choice'].dropna().unique().tolist())}")
        else:
            print(f"\n[diag] {mname}: m1/m2 columns MISSING — normalize_generative_df merge likely failed")
        mdf2['co'] = mdf2.apply(is_optimal_choice, axis=1)
        #print(f"head of {mname} free-trial df with columns m1,m2, and co:\n{mdf2[['m1', 'm2', 'choice', 'co']].head()}")
        #print(f"{mname} free-trial optimal choice rate: {mdf2['co'].mean():.3f} (n={len(mdf2)})")
        model_free_dfs.append(create_df_with_only_free_trials(mdf2))
        model_labels.append(mname.replace('_', '-'))
        model_plot_colors.append(model_color_map.get(mname, '#000000'))

    return human_free, model_free_dfs, model_labels, model_plot_colors


# ── Data loading ──────────────────────────────────────────────────────────────
timeline_df = None
if os.path.isfile(args.timeline):
    timeline_df = pd.read_csv(args.timeline)
    print(f"Loaded timeline: {timeline_df.shape}")
else:
    print(f"Warning: timeline not found at {args.timeline}; generative CSVs will not be enriched")

if not os.path.isfile(args.original_data):
    raise FileNotFoundError(f"Original data not found: {args.original_data}")
df_wilson, df_pilot = read_original_data(args.original_data)
#load only first 100 games
df_pilot = df_pilot[df_pilot['game'] <= 100]
subjects_pilot = process_subjects(df_pilot)
print(f"Loaded {len(subjects_pilot)} pilot subjects")

# Auto-discover all model subdirectories under PATH
model_dfs, model_subjects = load_models(base_path=PATH, timeline_df=timeline_df)

# Filter by LLM size if requested (e.g. --llm-size 8b or --llm-size 70b)
if args.llm_size:
    size_key = args.llm_size.lower()
    model_dfs = {k: v for k, v in model_dfs.items() if size_key in k.lower()}
    model_subjects = {k: v for k, v in model_subjects.items() if size_key in k.lower()}
    print(f"Filtered to llm-size='{args.llm_size}': {list(model_dfs.keys())}")

# ── Colors ────────────────────────────────────────────────────────────────────
family_mapping = identify_model_families(model_dfs)
family_colors  = define_colors_for_families(family_mapping)
print(f"Family Colors: {family_colors}")

# One color per model (cycle through the family palette)
model_color_map = {}
for fam, models in family_mapping.items():
    fam_colors = family_colors[fam]
    for i, (mname, _, _) in enumerate(models):
        model_color_map[mname] = fam_colors[i % len(fam_colors)]

human_color = '#333333'
# ── Choice curves: per-model ──────────────────────────────────────────────────
bin_edges = [-25, -15, -5, 5, 15, 25]
RTmin, RTmax = 0.1, 3.0

for mname, msubjects in model_subjects.items():
    if not msubjects:
        continue
    color = model_color_map.get(mname, '#000000')
    label = mname.replace('_', '-')
    fig_m, ax_m = plt.subplots(1, 2, figsize=(24, 8))
    plotting_utils.set_dynamic_fontsize(fig_width=8, base_font=20)
    plot_choice_curves_v2(ax_m, subjects_pilot, bin_edges, RTmin, RTmax,
                          AZblue=human_color, AZred=human_color)
    plot_model_lines(ax_m, msubjects, bin_edges,
                     AZblue=color, AZred=color, label=label)
    fig_m.suptitle(f'Choice Curves: Equal vs. Unequal Info ({label})')
    plt.tight_layout()

# ── Choice curves: all models combined ───────────────────────────────────────
fig_all, ax_all = plt.subplots(1, 2, figsize=(24, 8))
plot_choice_curves_v2(ax_all, subjects_pilot, bin_edges, RTmin, RTmax,
                      AZblue=human_color, AZred=human_color)
for mname, msubjects in model_subjects.items():
    if not msubjects:
        continue
    color = model_color_map.get(mname, '#000000')
    label = mname.replace('_', '-')
    plot_model_lines(ax_all, msubjects, bin_edges,
                     AZblue=color, AZred=color, label=label)
fig_all.suptitle('Choice Curves: Equal vs. Unequal Info (all models)')
plt.tight_layout()

# ── Horizon-6 only: all models combined ──────────────────────────────────────
fig_h6, ax_h6 = plt.subplots(1, 2, figsize=(24, 8))
plot_choice_curves_v2_h6(ax_h6, subjects_pilot, bin_edges, RTmin, RTmax,
                          AZblue=human_color, AZred=human_color)
for mname, msubjects in model_subjects.items():
    if not msubjects:
        continue
    color = model_color_map.get(mname, '#000000')
    label = mname.replace('_', '-')
    plot_model_lines_h6(ax_h6, msubjects, bin_edges,
                        AZblue=color, AZred=color, label=label)
fig_h6.suptitle('Choice Curves (Horizon 6): Equal vs. Unequal Info (all models)')
plt.tight_layout()
#print

# ── Free-trial accuracy: all models combined ─────────────────────────────────
human_free, model_free_dfs, model_labels, model_plot_colors = build_free_trial_dfs(
    df_pilot, model_dfs, model_color_map, human_color
)
fig_acc, leg_acc = plot_trial_accuracy_with_multiple_centaur_models(
    human_free,
    centaur_dfs=model_free_dfs,
    labels=model_labels,
    title='Free-Trial Accuracy (All Models)',
    centaur_colors=model_plot_colors,
)

# ── Trial-5 accuracy bar chart ───────────────────────────────────────────────
all_dfs    = [human_free] + model_free_dfs
all_labels = ['Human'] + model_labels
all_colors = model_plot_colors[:len(all_dfs)]
fig_t5 = plot_fifth_trial_accuracy_bar(
    dfs=all_dfs,
    labels=['Human','Centaur-70B','Llama-Instruct\n-3.1-70B',],
    colors=all_colors,
    family_labels=['' for _ in all_dfs],
    family_slices=list(range(len(all_dfs))),
)

# ── Save figures ──────────────────────────────────────────────────────────────
if args.save_dir:
    os.makedirs(args.save_dir, exist_ok=True)
    saved = 0
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fname = os.path.join(args.save_dir, f'figure_{num}.png')
        try:
            fig.savefig(fname, bbox_inches='tight')
            saved += 1
        except Exception as e:
            print(f"Warning: failed to save figure {num}: {e}")
    print(f"Saved {saved} figure(s) to {args.save_dir}")

if args.show:
    plt.show()