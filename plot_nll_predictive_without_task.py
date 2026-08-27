"""Plot NLL bars for predictive_without_task/<task>/nll_sem_summary.csv.

predictive_without_task/ holds pre-aggregated mean_nll/sem_nll per
(task, experiment, method_id) rather than per-participant CSVs under a
'singles/' folder, so family_mapping is built directly from numeric
(mean, sem, label) triplets -- the same path predictive_plots.py already
uses for the RW json-metrics case -- instead of from DataFrames.

Each task's two Centaur methods (context-free / full-context) go in the
'centaur' family, which plot_loglikelihood_bars_dynamic only special-cases
for exactly 1 or 2 entries -- so a third, classic-cognitive-model bar is
added as its own single-entry 'domain-specific' family instead of a third
entry in 'centaur', reusing the same family predictive_plots.py already
uses elsewhere for the RW cognitive-model baseline (same pink/grey color).
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import predictive_plots as pp
import plotting_utils

# rl_waltmann's RW model (rescorla_wagner.py) is fit on 50 train participants
# and evaluated per-trial on 6 held-out test participants -- restrict the
# Centaur bars here to that same 6-participant test set (rather than all 56),
# and add the RW bar itself, computed from its own per-trial singles CSVs so
# it's pooled over trials the same way the Centaur bars are below.
RL_WALTMANN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_waltmann')
RW_TEST_SPLIT_PATH = os.path.join(RL_WALTMANN_DIR, 'data', 'out', 'rw', 'train_test_split.json')
RW_FULL_SINGLES_DIR = os.path.join(RL_WALTMANN_DIR, 'data', 'out', 'rw_full', 'singles')
TEXT_WIDTH = 5.6  # inches, for a single-column figure in the eLife template
TARGETED_FIG_HEIGHT = 2.2
RATIO_NARROW=0.5
RATIO_WIDE=0.6
FIGSIZE_ALL={
    'narrow': (TEXT_WIDTH * RATIO_NARROW, TARGETED_FIG_HEIGHT),
    'wide': (TEXT_WIDTH * RATIO_WIDE, TARGETED_FIG_HEIGHT),
}
FIGSIZE_BAR = FIGSIZE_ALL['narrow']
BASE_FONT = 12

def _rl_waltmann_test_participants():
    with open(RW_TEST_SPLIT_PATH) as f:
        return json.load(f)['test']


def _rl_waltmann_rw_bar(test_participants):
    """Pool per-trial NLL across rw_full's test-participant singles CSVs."""
    vals = []
    for pid in test_participants:
        path = os.path.join(RW_FULL_SINGLES_DIR, f'participant_{pid}.csv')
        vals.append(pd.read_csv(path)['nll'])
    vals = pd.concat(vals, ignore_index=True)
    mean = float(vals.mean())
    sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, sem

# predictive_without_task/<folder> -> predictive_plots.TASK_NUM_CHOICES key,
# so the chance-guessing line matches the number of discrete choices in that task.
FOLDER_TO_TASK_KEY = {
    'rl_waltmann': 'rl_waltmann',
    'wilson_horizon': 'horizon',
    'wu_spatial': 'spatially_correlated',
    'multi_cue_judgement': 'multi_cue_judgment',
}

METHOD_LABELS = {
    'context_free_centaur': 'History\nonly',
    'centaur_full_context_simulation': 'Full\nprompt',
}
METHOD_ORDER = ['centaur_full_context_simulation', 'context_free_centaur']

# Classic cognitive-model NLL per task (mean only -- no per-participant SD was
# given, so SEM is set to 0.0, matching predictive_plots.py's own convention
# for the RW baseline when only a scalar mean is available). rl_waltmann's RW
# bar is computed separately (see _rl_waltmann_rw_bar) since it comes from
# our own per-trial fit rather than an externally supplied scalar.
COGNITIVE_MODEL_NLL = {
    'wu_spatial': 2.792812824249268,
    'multi_cue_judgement': 1.9236862861598658,
    'wilson_horizon': 0.63955120742321,
}
#COGNITIVE MODEL OF EACH TASK
COGNITIVE_MODEL_LABELS = {
    'wu_spatial': 'GP-UCB',
    'multi_cue_judgement': 'Linear\nregression',
    'wilson_horizon': 'RW'

}
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictive_without_task')


def _family_mapping_with_cognitive_model(centaur_models, folder_name):
    family_mapping = {'centaur': centaur_models}
    cog_nll = COGNITIVE_MODEL_NLL.get(folder_name)
    if cog_nll is not None:
        family_mapping['domain-specific'] = [(cog_nll, 0.0, COGNITIVE_MODEL_LABELS[folder_name])]
    return family_mapping


def _save(family_mapping, num_choices, out_png):
    fig = pp.plot_loglikelihood_bars_dynamic(family_mapping=family_mapping, num_choices=num_choices,
                                               show_xticklabels=True,figsize=FIGSIZE_BAR)
    plotting_utils.save_panel(fig, out_png, figsize=FIGSIZE_BAR)
    print(f"Saved {out_png}")


def plot_task(folder_name, out_dir='figures/'):
    task_key = FOLDER_TO_TASK_KEY[folder_name]
    num_choices = pp.TASK_NUM_CHOICES[task_key]
    task_dir = os.path.join(BASE_DIR, folder_name)

    if folder_name == 'rl_waltmann' and os.path.exists(RW_TEST_SPLIT_PATH):
        test_participants = _rl_waltmann_test_participants()
        raw = pd.read_csv(os.path.join(task_dir, 'raw_results.csv'))
        raw = raw[raw['participant'].isin(test_participants)]
        centaur_models = []
        for method_id in METHOD_ORDER:
            vals = raw.loc[raw['method_id'] == method_id, 'trial_nll'].dropna()
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            centaur_models.append((mean, sem, METHOD_LABELS[method_id]))
        family_mapping = {'centaur': centaur_models}
        rw_mean, rw_sem = _rl_waltmann_rw_bar(test_participants)
        family_mapping['domain-specific'] = [(rw_mean, rw_sem, 'RW')]
        no_context_dir = os.path.join(out_dir)
        if not os.path.exists(no_context_dir):
            os.makedirs(no_context_dir)
        out_png = os.path.join(no_context_dir, f'nll_bars_without_task_{folder_name}.png')
        _save(family_mapping, num_choices, out_png)
        return

    if folder_name == 'wilson_horizon':
        # The cognitive-model NLL we have is a single overall value for
        # wilson2014humans, not split by experiment -- so pool the 4
        # experiment subgroups (exp1/exp3/exp4/exp5) into one overall chart
        # instead of plotting them separately, to match that granularity.
        raw = pd.read_csv(os.path.join(task_dir, 'raw_results.csv'))
        centaur_models = []
        for method_id in METHOD_ORDER:
            vals = raw.loc[raw['method_id'] == method_id, 'trial_nll'].dropna()
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            centaur_models.append((mean, sem, METHOD_LABELS[method_id]))
        family_mapping = _family_mapping_with_cognitive_model(centaur_models, folder_name)
        
        no_context_dir = os.path.join(out_dir)
        if not os.path.exists(no_context_dir):
            os.makedirs(no_context_dir)
        out_png = os.path.join(no_context_dir, f'nll_bars_without_task_{folder_name}.png')
        _save(family_mapping, num_choices, out_png)
        return

    df = pd.read_csv(os.path.join(task_dir, 'nll_sem_summary.csv'))
    for experiment, exp_df in df.groupby('experiment'):
        centaur_models = []
        for method_id in METHOD_ORDER:
            row = exp_df[exp_df['method_id'] == method_id]
            if row.empty:
                continue
            r = row.iloc[0]
            centaur_models.append((float(r['mean_nll']), float(r['sem_nll']), METHOD_LABELS[method_id]))
        if not centaur_models:
            continue

        family_mapping = _family_mapping_with_cognitive_model(centaur_models, folder_name)
        suffix = '' if experiment == 'all' else f'_{experiment}'
        # in figures folder create folder no context to save the figures
        no_context_dir = os.path.join(out_dir)
        if not os.path.exists(no_context_dir):
            os.makedirs(no_context_dir)
        out_png = os.path.join(no_context_dir, f'nll_bars_without_task_{folder_name}{suffix}.png')
        _save(family_mapping, num_choices, out_png)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default='figures/',
                         help="Output folder for the figures (default: figures/)")
    args = parser.parse_args()

    for folder_name in FOLDER_TO_TASK_KEY:
        plot_task(folder_name, out_dir=args.out_dir)
