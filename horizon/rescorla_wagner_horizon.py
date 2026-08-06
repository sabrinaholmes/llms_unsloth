"""
Rescorla-Wagner model for the Horizon Task.

Three modes (run via argparse or call functions directly):
  1. simulate  -- generate timeline, simulate data, fit model to sim data
  2. fit        -- fit model to provided train/test CSV files
  3. generate   -- simulate choices given a pre-built timeline CSV
"""

import argparse
import json
import os
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import plotting_utils

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_seeds(num_seeds=32, seed=42):
    random.seed(seed)
    return [random.randint(1, 100_000) for _ in range(num_seeds)]


OPTION2IDX = {'H': 0, 'I': 1}

# ---------------------------------------------------------------------------
# RW Model
# ---------------------------------------------------------------------------

class RWModel:
    """
    Rescorla-Wagner with asymmetric learning rates, stickiness, and
    an information bonus.

    Parameters
    ----------
    N            : number of arms
    alpha_plus   : raw learning rate for positive PEs  (passed through sigmoid)
    alpha_minus  : raw learning rate for negative PEs  (passed through sigmoid)
    a            : inverse temperature (temperature weight on V)
    b            : stickiness weight
    c            : information-count bonus weight
    d            : initial value V0
    """

    def __init__(self, N, alpha_plus, alpha_minus, a, b, c, d):
        self.N = N
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = np.full(N, d)
        self.S = np.zeros(N)   # stickiness (1 for last-chosen arm)
        self.I = np.zeros(N)   # information counts

    def action_probs(self):
        z = self.a * self.V + self.b * self.S + self.c * self.I
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def choose(self):
        return np.random.choice(self.N, p=self.action_probs())

    def update(self, c_t, r_t):
        delta = r_t - self.V[c_t]
        lr = sigmoid(self.alpha_plus) if delta >= 0 else sigmoid(self.alpha_minus)
        self.V[c_t] += lr * delta
        self.S[:] = 0
        self.S[c_t] = 1
        self.I[c_t] += 1


# ---------------------------------------------------------------------------
# Simplified RW Model (3 parameters)
# ---------------------------------------------------------------------------

class RWModelSimple:
    """
    Simplified Rescorla-Wagner with a single learning rate, softmax temperature,
    and initial value.

    Parameters
    ----------
    N     : number of arms
    alpha : learning rate in [0, 1]
    theta : inverse temperature (softmax)
    d     : initial value V0
    """

    def __init__(self, N, alpha, theta, d):
        self.N = N
        self.alpha = alpha
        self.theta = theta
        self.V = np.full(N, d)

    def action_probs(self):
        z = self.theta * self.V
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def choose(self):
        return np.random.choice(self.N, p=self.action_probs())

    def update(self, c_t, r_t):
        self.V[c_t] += self.alpha * (r_t - self.V[c_t])


def nll_rw_simple(params, df, N=2):
    """Negative log-likelihood for the simplified 3-parameter RW model."""
    alpha, theta, d = params
    model = RWModelSimple(N, alpha, theta, d)
    loglik = 0.0
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    is_free = (df['trial_type'] == 'free').to_numpy()
    for i in range(len(choices)):
        if is_free[i]:
            loglik += np.log(model.action_probs()[choices[i]] + 1e-12)
        model.update(choices[i], rewards[i])
    return -loglik


def fit_rw_simple(df, initial_guesses=None, N=2):
    """Fit simplified RW model. Returns (best_params_array, scipy_result)."""
    if initial_guesses is None:
        initial_guesses = [
            [0.5, 5.0, 0.3],
            [0.2, 3.0, 0.1],
            [0.8, 8.0, 0.5],
        ]
    bounds = [(0, 1), (0, 50), (0, 10)]  # alpha in [0,1], theta, d
    best_result, best_loss = None, float('inf')
    for guess in initial_guesses:
        res = minimize(nll_rw_simple, x0=guess, args=(df, N),
                       bounds=bounds, method='L-BFGS-B')
        if res.fun < best_loss:
            best_result, best_loss = res, res.fun
    return best_result.x, best_result


def fit_rw_simple_per_participant(df, participant_col='participant_id', N=2):
    """Fit simplified RW independently for each participant."""
    results = {}
    group_cols = ['exp', participant_col] if 'exp' in df.columns else [participant_col]
    for key, group in df.groupby(group_cols):
        params, _ = fit_rw_simple(group.reset_index(drop=True), N=N)
        print(f"  participant {key}: {dict(zip(['alpha','theta','d'], params.round(4)))}")
        results[str(key)] = params
    return results


# ---------------------------------------------------------------------------
# Timeline generation
# ---------------------------------------------------------------------------

class HorizonTimeline:
    """Generates forced + free trials for one game of the Horizon Task."""

    def __init__(self, game_number=1, info_condition='[1 3]', std=8, horizon=1):
        assert info_condition in ['[1 3]', '[2 2]']
        self.game_number = game_number
        self.info_condition = info_condition
        self.horizon = horizon
        self.std = std
        self.H_rewards = []
        self.I_rewards = []
        self.free_choice_rewards = []
        self.trial_data = []
        self._set_reward_means()

    def _set_reward_means(self):
        mean_diff = np.random.choice([4, 8, 12, 20, 30])
        base_mean = np.random.choice([40, 60])
        means = [base_mean, base_mean + mean_diff]
        np.random.shuffle(means)
        self.means = {'H': means[0], 'I': means[1]}

    def _add_forced_trials(self):
        counts = {'[1 3]': [1, 3], '[2 2]': [2, 2]}[self.info_condition]
        order = ['H'] * counts[0] + ['I'] * counts[1]
        np.random.shuffle(order)
        forced = []
        for label in order:
            reward = round(np.random.normal(self.means[label], self.std))
            idx = 0 if label == 'H' else 1
            self.trial_data.append({
                'trial_type': 'forced',
                'option': idx,
                'reward': reward,
                'free_H_reward': None,
                'free_I_reward': None,
            })
            (self.H_rewards if label == 'H' else self.I_rewards).append(reward)
            forced.append((label, reward))
        return forced

    def _add_free_choice_trials(self):
        for _ in range(self.horizon):
            h_r = round(np.random.normal(self.means['H'], self.std))
            i_r = round(np.random.normal(self.means['I'], self.std))
            self.trial_data.append({
                'trial_type': 'free',
                'option': None,
                'reward': None,
                'free_H_reward': h_r,
                'free_I_reward': i_r,
            })
            self.free_choice_rewards.append((h_r, i_r))

    def get_game_data(self):
        forced = self._add_forced_trials()
        self._add_free_choice_trials()
        return {
            'game_number': self.game_number,
            'means': self.means,
            'info_condition': self.info_condition,
            'horizon': self.horizon,
            'forced_trials': forced,
            'trial_data': self.trial_data,
        }


def generate_experiment_df(n_participants=5, blocks_per_participant=4,
                            games_per_block=80, std=8):
    rows = []
    for participant in range(1, n_participants + 1):
        for block in range(1, blocks_per_participant + 1):
            for game_in_block in range(1, games_per_block + 1):
                game_number = (block - 1) * games_per_block + game_in_block
                info_condition = np.random.choice(['[1 3]', '[2 2]'])
                horizon = np.random.choice([1, 6])
                game = HorizonTimeline(game_number, info_condition, std, horizon)
                gd = game.get_game_data()
                for i, trial in enumerate(gd['trial_data']):
                    rows.append({
                        'participant': participant,
                        'block': block,
                        'game_number': game_number,
                        'trial_number': i + 1,
                        'trial_type': trial['trial_type'],
                        'choice': trial['option'],
                        'reward': trial['reward'],
                        'free_H_reward': trial['free_H_reward'],
                        'free_I_reward': trial['free_I_reward'],
                        'info_condition': info_condition,
                        'horizon': horizon,
                        'H_mean': gd['means']['H'],
                        'I_mean': gd['means']['I'],
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_rw(df_trials, seeds, params, N=2):
    """
    Simulate the RW model over df_trials for each seed.
    Free-trial choices are filled in by the model.
    Returns concatenated DataFrame across seeds.
    """
    alpha_plus, alpha_minus, a, b, c, d = params
    out = []

    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)

        agent = RWModel(N, alpha_plus, alpha_minus, a, b, c, d)
        cum_reward = 0
        df = df_trials.copy()
        df['model_id'] = seed
        df['score'] = np.nan

        for idx, row in df.iterrows():
            if row.trial_type == 'forced':
                choice_idx = OPTION2IDX[row.choice] if isinstance(row.choice, str) else int(row.choice)
                reward = row.reward
            else:
                choice_idx = agent.choose()
                reward = row.free_H_reward if choice_idx == 0 else row.free_I_reward
                df.at[idx, 'choice'] = choice_idx
                df.at[idx, 'reward'] = reward

            cum_reward += reward
            df.at[idx, 'score'] = cum_reward
            agent.update(choice_idx, reward)

        print(f"  seed {seed:>6}: cumulative reward = {cum_reward}")
        out.append(df)

    return pd.concat(out, ignore_index=True)


def simulate_rw_per_participant(df_trials, seeds, params, N=2, participant_col='participant_id'):
    """
    Simulate the RW model separately for each participant in df_trials.
    Expects columns: <participant_col>, trial_type, choice, reward,
                     reward_H / reward_I (or reward_mean_H / reward_mean_I).
    """
    alpha_plus, alpha_minus, a, b, c, d = params
    out = []
    group_cols = ['exp', participant_col] if 'exp' in df_trials.columns else [participant_col]
    groups = list(df_trials.groupby(group_cols, sort=False))

    if len(seeds) < len(groups):
        seeds = (seeds * (len(groups) // len(seeds) + 1))[:len(groups)]

    for i, (key, df) in enumerate(groups):
        pid = key if isinstance(key, (int, str, float)) else key[-1]
        seed = seeds[i]
        np.random.seed(seed)
        random.seed(seed)

        agent = RWModel(N, alpha_plus, alpha_minus, a, b, c, d)
        cum_reward = 0
        df = df.copy()
        df['model_id'] = seed
        df['score'] = np.nan

        for idx, row in df.iterrows():
            if row.trial_type == 'forced':
                choice_idx = OPTION2IDX[row.choice] if isinstance(row.choice, str) else int(row.choice)
                reward = row.reward
            else:
                choice_idx = agent.choose()
                if pd.isna(row.get('reward_H', np.nan)) or pd.isna(row.get('reward_I', np.nan)):
                    h_r = round(np.random.normal(row.reward_mean_H, 8))
                    i_r = round(np.random.normal(row.reward_mean_I, 8))
                else:
                    h_r, i_r = row.reward_H, row.reward_I
                reward = h_r if choice_idx == 0 else i_r
                df.at[idx, 'choice'] = choice_idx
                df.at[idx, 'reward'] = reward

            cum_reward += reward
            df.at[idx, 'score'] = cum_reward
            agent.update(choice_idx, reward)

        print(f"  participant {pid}, seed {seed}: cumulative reward = {cum_reward}")
        out.append(df)

    return pd.concat(out, ignore_index=True)


def _simulate_simple_participant(df_trials, seed, params, N=2):
    """Simulate RWModelSimple for a single participant's trial data."""
    alpha, theta, d = params
    np.random.seed(seed)
    random.seed(seed)

    agent = RWModelSimple(N, alpha, theta, d)
    cum_reward = 0
    df = df_trials.copy()
    df['model_id'] = seed
    df['score'] = np.nan

    for idx, row in df.iterrows():
        if row.trial_type == 'forced':
            choice_idx = OPTION2IDX[row.choice] if isinstance(row.choice, str) else int(row.choice)
            reward = row.reward
        else:
            choice_idx = agent.choose()
            if pd.isna(row.get('reward_H', np.nan)) or pd.isna(row.get('reward_I', np.nan)):
                h_r = round(np.random.normal(row.reward_mean_H, 8))
                i_r = round(np.random.normal(row.reward_mean_I, 8))
            else:
                h_r, i_r = row.reward_H, row.reward_I
            reward = h_r if choice_idx == 0 else i_r
            df.at[idx, 'choice'] = choice_idx
            df.at[idx, 'reward'] = reward

        cum_reward += reward
        df.at[idx, 'score'] = cum_reward
        agent.update(choice_idx, reward)

    return df


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalise_df(df):
    """
    Ensure a DataFrame has a 'trial_type' column ('forced'/'free') regardless
    of how the source CSV encoded it:
      - 'type' column (string)  → rename to trial_type
      - 'forced' column (int)   → 1 = 'forced', 0 = 'free'
    Also maps string choices ('H'/'I', 'J'/'K', etc.) to integers 0/1.
    """
    df = df.copy()

    if 'trial_type' not in df.columns:
        if 'type' in df.columns:
            df.rename(columns={'type': 'trial_type'}, inplace=True)
        elif 'forced' in df.columns:
            df['trial_type'] = df['forced'].map({1: 'forced', 0: 'free'})

    if df['choice'].dtype == object:
        unique_vals = df['choice'].dropna().unique()
        mapping = {v: i for i, v in enumerate(sorted(unique_vals))}
        df['choice'] = df['choice'].map(mapping)

    return df


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def nll_rw(params, df, N=2):
    """Negative log-likelihood; accumulates log P(choice) on free trials only."""
    model = RWModel(N, *params)
    loglik = 0.0
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    is_free = (df['trial_type'] == 'free').to_numpy()
    for i in range(len(choices)):
        if is_free[i]:
            loglik += np.log(model.action_probs()[choices[i]] + 1e-12)
        model.update(choices[i], rewards[i])
    return -loglik


def eval_rw_per_trial(params, df, participant_id=None, N=2, use_simple=False):
    """
    Replay the model over df and return one row per free trial with
    columns: participant_id, trial_index, choice, p_choice, nll.
    trial_index is the position within df (0-based).
    """
    model = RWModelSimple(N, *params) if use_simple else RWModel(N, *params)
    rows = []
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    is_free = (df['trial_type'] == 'free').to_numpy()
    for i in range(len(choices)):
        if is_free[i]:
            p = model.action_probs()[choices[i]]
            rows.append({
                'participant_id': participant_id,
                'trial_index': i+1,
                'choice': choices[i],
                'p_choice': float(p),
                'nll': float(-np.log(p + 1e-12)),
            })
        model.update(choices[i], rewards[i])
    return pd.DataFrame(rows)


def fit_rw_model(df, initial_guesses=None, N=2):
    """
    Fit RW model parameters by minimising NLL.
    Returns (best_params_array, scipy_result).
    """
    if initial_guesses is None:
        initial_guesses = [
            [0.5, 0.3, 5.0, 0.2, 0.05, 0.3],
            [1.0, 0.5, 8.0, 0.5, 0.1,  0.5],
            [0.2, 0.1, 3.0, 0.1, 0.01, 0.1],
        ]

    bounds = [(0, 10)] * 6   # all params bounded [0, 10]

    best_result, best_loss = None, float('inf')
    for guess in initial_guesses:
        res = minimize(nll_rw, x0=guess, args=(df, N),
                       bounds=bounds, method='L-BFGS-B')
        if res.fun < best_loss:
            best_result, best_loss = res, res.fun

    return best_result.x, best_result


def fit_rw_per_participant(df, participant_col='participant_id', N=2):
    """
    Fit RW model independently for each participant.
    Returns a dict {participant_id: params_array}.
    """
    results = {}
    group_cols = ['exp', participant_col] if 'exp' in df.columns else [participant_col]
    for key, group in df.groupby(group_cols):
        params, _ = fit_rw_model(group.reset_index(drop=True), N=N)
        print(f"  participant {key}: {dict(zip(['alpha_plus','alpha_minus','a','b','c','d'], params.round(4)))}")
        results[str(key)] = params
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_parameter_comparison(true_params, fitted_params, param_names,
                               title='True vs Fitted Parameters'):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_params, fitted_params, c='steelblue', zorder=3)
    lims = [min(min(true_params), min(fitted_params)) * 0.9,
            max(max(true_params), max(fitted_params)) * 1.1]
    plt.plot(lims, lims, 'k--', label='Perfect fit')
    for i, name in enumerate(param_names):
        plt.annotate(name, (true_params[i], fitted_params[i]),
                     textcoords='offset points', xytext=(5, 5))
    plt.xlabel('True parameters')
    plt.ylabel('Fitted parameters')
    plt.title(title)
    plt.legend()
    plotting_utils.style_y_gridlines(plt.gca())
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Mode 1: simulate
# ---------------------------------------------------------------------------

def run_simulate(args):
    print('\n=== Mode: simulate ===')
    print('Generating timeline …')
    timeline_df = generate_experiment_df(
        n_participants=args.n_participants,
        blocks_per_participant=args.blocks,
        games_per_block=args.games,
    )
    print(f'  {len(timeline_df)} rows generated')

    true_params = [
        args.alpha_plus, args.alpha_minus,
        args.a, args.b, args.c, args.d,
    ]
    seeds = generate_seeds(num_seeds=args.num_seeds)

    print('Simulating …')
    sim_df = simulate_rw(timeline_df, seeds, true_params)

    print('Fitting to simulated data …')
    fitted_params, result = fit_rw_model(sim_df, [true_params])
    param_names = ['alpha_plus', 'alpha_minus', 'a', 'b', 'c', 'd']
    print('  True:   ', dict(zip(param_names, true_params)))
    print('  Fitted: ', dict(zip(param_names, fitted_params.tolist())))

    if not args.no_plot:
        plot_parameter_comparison(true_params, fitted_params.tolist(),
                                  param_names, 'Simulate: True vs Fitted')

    sim_df = normalise_df(sim_df)
    nll = nll_rw(fitted_params, sim_df)
    n_free = len(sim_df[sim_df['trial_type'] == 'free'])
    print(f'  NLL = {nll:.2f}  |  avg log-loss/free-trial = {nll/n_free:.4f}')

    if args.out_sim:
        sim_df.to_csv(args.out_sim, index=False)
        print(f'  Simulated data saved → {args.out_sim}')

    if args.out_params:
        with open(args.out_params, 'w') as f:
            json.dump(dict(zip(param_names, fitted_params.tolist())), f, indent=4)
        print(f'  Fitted params saved → {args.out_params}')

    return sim_df, fitted_params


# ---------------------------------------------------------------------------
# Mode 2: fit
# ---------------------------------------------------------------------------

def run_fit(args):
    print('\n=== Mode: fit ===')
    train = normalise_df(pd.read_csv(args.train_csv))
    test  = normalise_df(pd.read_csv(args.test_csv))
    print(f'  Train: {len(train)} rows  |  Test: {len(test)} rows')

    participant_col = 'participant_id' if 'participant_id' in train.columns else 'participant'
    use_simple = getattr(args, 'model', 'full') == 'simple'
    fit_mode   = getattr(args, 'fit_mode', 'per_participant')

    if use_simple:
        param_names = ['alpha', 'theta', 'd']
        fit_single_fn = fit_rw_simple
        fit_per_fn    = fit_rw_simple_per_participant
        nll_fn        = nll_rw_simple
    else:
        param_names = ['alpha_plus', 'alpha_minus', 'a', 'b', 'c', 'd']
        fit_single_fn = fit_rw_model
        fit_per_fn    = fit_rw_per_participant
        nll_fn        = nll_rw

    total_nll, total_free = 0.0, 0
    per_participant_metrics = {}

    if fit_mode == 'pooled':
        model_label = 'simple' if use_simple else 'full'
        print(f'Fitting {model_label} model on pooled train data …')
        pooled_params, _ = fit_single_fn(train.reset_index(drop=True), N=2)
        print(f'  Pooled params: {dict(zip(param_names, pooled_params.round(4)))}')

        print('Evaluating pooled params on each test participant …')
        test_group_cols = ['exp', participant_col] if 'exp' in test.columns else [participant_col]
        trial_nll_chunks = []
        for key, ptest in test.groupby(test_group_cols):
            ptest = ptest.reset_index(drop=True)
            val_nll = nll_fn(pooled_params, ptest)
            n_free  = len(ptest[ptest['trial_type'] == 'free'])
            total_nll += val_nll
            total_free += n_free
            per_participant_metrics[str(key)] = {
                'fitted_params': dict(zip(param_names, pooled_params.tolist())),
                'validation_nll': val_nll,
                'avg_log_loss_per_free_trial': val_nll / n_free if n_free else None,
            }
            if args.out_trial_nll:
                pid = key if isinstance(key, (int, str, float)) else key[-1]
                trial_nll_chunks.append(
                    eval_rw_per_trial(pooled_params, ptest, participant_id=pid,
                                      N=2, use_simple=use_simple)
                )

        print(f'  Total validation NLL = {total_nll:.2f}  |  avg log-loss/free-trial = {total_nll/total_free:.4f}')

        metrics = {
            'model': model_label,
            'fit_mode': 'pooled',
            'pooled_params': dict(zip(param_names, pooled_params.tolist())),
            'per_participant': per_participant_metrics,
            'total_validation_nll': total_nll,
            'avg_log_loss_per_free_trial': total_nll / total_free if total_free else None,
        }
        if args.out_metrics:
            with open(args.out_metrics, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f'  Metrics saved → {args.out_metrics}')

        if args.out_params:
            with open(args.out_params, 'w') as f:
                json.dump({'pooled': dict(zip(param_names, pooled_params.tolist()))}, f, indent=4)
            print(f'  Pooled params saved → {args.out_params}')

        if args.out_trial_nll and trial_nll_chunks:
            os.makedirs(args.out_trial_nll, exist_ok=True)
            for chunk in trial_nll_chunks:
                pid = chunk['participant_id'].iloc[0]
                path = os.path.join(args.out_trial_nll, f'participant_{pid}.csv')
                chunk.to_csv(path, index=False)
            print(f'  Per-trial NLL saved → {args.out_trial_nll}/ ({len(trial_nll_chunks)} files)')

        return pooled_params, metrics

    else:  # per_participant
        model_label = 'simple' if use_simple else 'full'
        print(f'Fitting {model_label} model per participant …')
        per_participant_params = fit_per_fn(train, participant_col=participant_col)

        test_group_cols = ['exp', participant_col] if 'exp' in test.columns else [participant_col]
        trial_nll_chunks = []
        for key, ptest in test.groupby(test_group_cols):
            key_str = str(key)
            if key_str not in per_participant_params:
                continue
            params = per_participant_params[key_str]
            ptest = ptest.reset_index(drop=True)
            val_nll = nll_fn(params, ptest)
            n_free  = len(ptest[ptest['trial_type'] == 'free'])
            total_nll += val_nll
            total_free += n_free
            per_participant_metrics[key_str] = {
                'fitted_params': dict(zip(param_names, params.tolist())),
                'validation_nll': val_nll,
                'avg_log_loss_per_free_trial': val_nll / n_free if n_free else None,
            }
            if args.out_trial_nll:
                pid = key if isinstance(key, (int, str, float)) else key[-1]
                trial_nll_chunks.append(
                    eval_rw_per_trial(params, ptest, participant_id=pid,
                                      N=2, use_simple=use_simple)
                )

        print(f'  Total validation NLL = {total_nll:.2f}  |  avg log-loss/free-trial = {total_nll/total_free:.4f}')

        metrics = {
            'model': model_label,
            'fit_mode': 'per_participant',
            'per_participant': per_participant_metrics,
            'total_validation_nll': total_nll,
            'avg_log_loss_per_free_trial': total_nll / total_free if total_free else None,
        }
        if args.out_metrics:
            with open(args.out_metrics, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f'  Metrics saved → {args.out_metrics}')

        if args.out_params:
            all_params = np.array(list(per_participant_params.values()))
            exportable = {
                'per_participant': {
                    str(pid): dict(zip(param_names, p.tolist()))
                    for pid, p in per_participant_params.items()
                },
                'mean':   dict(zip(param_names, all_params.mean(axis=0).tolist())),
                'median': dict(zip(param_names, np.median(all_params, axis=0).tolist())),
            }
            with open(args.out_params, 'w') as f:
                json.dump(exportable, f, indent=4)
            print(f'  Per-participant params saved → {args.out_params}')
            print(f'  Mean params:   {dict(zip(param_names, all_params.mean(axis=0).round(4)))}')
            print(f'  Median params: {dict(zip(param_names, np.median(all_params, axis=0).round(4)))}')

        if args.out_trial_nll and trial_nll_chunks:
            os.makedirs(args.out_trial_nll, exist_ok=True)
            for chunk in trial_nll_chunks:
                pid = chunk['participant_id'].iloc[0]
                path = os.path.join(args.out_trial_nll, f'participant_{pid}.csv')
                chunk.to_csv(path, index=False)
            print(f'  Per-trial NLL saved → {args.out_trial_nll}/ ({len(trial_nll_chunks)} files)')

        return per_participant_params, metrics


# ---------------------------------------------------------------------------
# Mode 3: generate
# ---------------------------------------------------------------------------

def _param_dict_to_list(param_dict):
    full_param_names   = ['alpha_plus', 'alpha_minus', 'a', 'b', 'c', 'd']
    simple_param_names = ['alpha', 'theta', 'd']
    is_simple = 'alpha' in param_dict and 'alpha_plus' not in param_dict
    pnames = simple_param_names if is_simple else full_param_names
    return [param_dict[k] for k in pnames], is_simple, pnames


def run_generate(args):
    print('\n=== Mode: generate ===')
    timeline = normalise_df(pd.read_csv(args.timeline_csv))

    with open(args.params_json) as f:
        raw = json.load(f)

    seeds = generate_seeds(num_seeds=args.num_seeds)
    aggregation = getattr(args, 'aggregation', 'mean')
    print(f'Generating choices (aggregation={aggregation}) …')

    # New format: {"per_participant": {...}, "mean": {...}, "median": {...}}
    if isinstance(raw, dict) and 'mean' in raw and 'median' in raw:
        if aggregation in ('mean', 'median'):
            param_dict = raw[aggregation]
            params, is_simple, _ = _param_dict_to_list(param_dict)
            print(f'  Using {aggregation} params: {param_dict}')
            results = simulate_rw(timeline, seeds, params)
        else:  # per_participant
            per_p = raw['per_participant']
            participant_col = 'participant_id' if 'participant_id' in timeline.columns else 'participant'
            out = []
            for pid, param_dict in per_p.items():
                params, is_simple, _ = _param_dict_to_list(param_dict)
                pid_val = type(timeline[participant_col].iloc[0])(pid)
                pdata = timeline[timeline[participant_col] == pid_val].copy()
                if len(pdata) == 0:
                    print(f'  warning: no timeline rows for participant {pid}, skipping')
                    continue
                if is_simple:
                    chunk = _simulate_simple_participant(pdata, seeds[0], params)
                else:
                    chunk = simulate_rw_per_participant(
                        pdata.assign(**{participant_col: pid_val}),
                        seeds, params, participant_col=participant_col)
                out.append(chunk)
            results = pd.concat(out, ignore_index=True)

    # Legacy format: flat {pid: {param: val}} dict or list
    else:
        first_val = next(iter(raw.values())) if isinstance(raw, dict) else None
        if isinstance(first_val, dict):
            participant_col = 'participant_id' if 'participant_id' in timeline.columns else 'participant'
            out = []
            for pid, param_dict in raw.items():
                params, is_simple, _ = _param_dict_to_list(param_dict)
                pid_val = type(timeline[participant_col].iloc[0])(pid)
                pdata = timeline[timeline[participant_col] == pid_val].copy()
                if len(pdata) == 0:
                    print(f'  warning: no timeline rows for participant {pid}, skipping')
                    continue
                if is_simple:
                    chunk = _simulate_simple_participant(pdata, seeds[0], params)
                else:
                    chunk = simulate_rw_per_participant(
                        pdata.assign(**{participant_col: pid_val}),
                        seeds, params, participant_col=participant_col)
                out.append(chunk)
            results = pd.concat(out, ignore_index=True)
        else:
            params = raw if isinstance(raw, list) else [raw[k] for k in ['alpha_plus', 'alpha_minus', 'a', 'b', 'c', 'd']]
            if 'participant_id' in timeline.columns or 'participant' in timeline.columns:
                results = simulate_rw_per_participant(timeline, seeds, params)
            else:
                results = simulate_rw(timeline, seeds, params)

    results['choice_label'] = results['choice'].map({0: 'H', 1: 'I'})

    if args.out_generated:
        results.to_csv(args.out_generated, index=False)
        print(f'  Generated data saved → {args.out_generated}')

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='Rescorla-Wagner model for the Horizon Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='mode', required=True)

    # --- simulate ---
    s = sub.add_parser('simulate', help='Generate timeline, simulate, and fit')
    s.add_argument('--n-participants', type=int, default=1)
    s.add_argument('--blocks',         type=int, default=4)
    s.add_argument('--games',          type=int, default=80)
    s.add_argument('--num-seeds',      type=int, default=32)
    s.add_argument('--alpha-plus',  type=float, default=1.0)
    s.add_argument('--alpha-minus', type=float, default=0.3)
    s.add_argument('--a',  type=float, default=10.0)
    s.add_argument('--b',  type=float, default=0.2)
    s.add_argument('--c',  type=float, default=0.05)
    s.add_argument('--d',  type=float, default=0.3)
    s.add_argument('--out-sim',    default=None, help='CSV path for simulated data')
    s.add_argument('--out-params', default=None, help='JSON path for fitted params')
    s.add_argument('--no-plot', action='store_true')

    # --- fit ---
    f = sub.add_parser('fit', help='Fit model to human data')
    f.add_argument('--train-csv',    required=True)
    f.add_argument('--test-csv',     required=True)
    f.add_argument('--model', choices=['full', 'simple'], default='full',
                   help='full (6 params) or simple (3 params: alpha, theta, d)')
    f.add_argument('--fit-mode', choices=['per_participant', 'pooled'], default='per_participant',
                   help='per_participant: fit each train participant independently; '
                        'pooled: fit one set of params on all train data, evaluate on test participants')
    f.add_argument('--out-metrics',   default=None, help='JSON path for metrics')
    f.add_argument('--out-params',    default=None, help='JSON path for per-participant params (used by generate)')
    f.add_argument('--out-trial-nll', default=None, help='Directory to save per-participant trial NLL CSVs (e.g. rw/singles)')

    # --- generate ---
    g = sub.add_parser('generate', help='Generate choices given a timeline CSV')
    g.add_argument('--timeline-csv', required=True)
    g.add_argument('--params-json',  required=True, help='JSON with fitted params')
    g.add_argument('--num-seeds',    type=int, default=32)
    g.add_argument('--aggregation', choices=['mean', 'median', 'per_participant'], default='mean',
                   help='How to derive a general model from per-participant fits')
    g.add_argument('--out-generated', default=None, help='CSV path for generated data')

    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == 'simulate':
        run_simulate(args)
    elif args.mode == 'fit':
        run_fit(args)
    elif args.mode == 'generate':
        run_generate(args)
