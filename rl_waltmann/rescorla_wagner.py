"""
Rescorla-Wagner model fitting for the rl_waltmann reversal-bandit task.

56 participants x 140 trials, binary choice/reward, no forced trials.

Two model variants, both fit with ONE shared ("pooled") parameter set across
50 training participants and evaluated per-trial on the remaining 6 held-out
participants:

  simple (3 params): alpha, theta (softmax inverse temperature), d (V0)
      pi = softmax(theta * V)
      delta = r - V[c]; V[c] += alpha * delta

  full (6 params): alpha_plus, alpha_minus, a, b, c, d
      pi = softmax(a*V + b*S + c*I)
      delta = r - V[c]
      V[c] += sigmoid(alpha_plus)  * delta   if delta >= 0
      V[c] += sigmoid(alpha_minus) * delta   if delta <  0
      S = one-hot(last choice); I = running per-arm choice counts

Usage:
    python3 rescorla_wagner.py --data-csv data/in/test_waltmann_data_cleaned.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Models (single-participant trajectories)
# ---------------------------------------------------------------------------

class RWModelSimple:
    """3-parameter RW: single learning rate + softmax inverse temperature."""

    def __init__(self, N, alpha, theta, d):
        self.N = N
        self.alpha = alpha
        self.theta = theta
        self.V = np.full(N, d, dtype=float)

    def action_probs(self):
        z = self.theta * self.V
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def update(self, c_t, r_t):
        self.V[c_t] += self.alpha * (r_t - self.V[c_t])


class RWModel:
    """6-parameter RW: asymmetric learning rates, stickiness, choice-count bonus."""

    def __init__(self, N, alpha_plus, alpha_minus, a, b, c, d):
        self.N = N
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.V = np.full(N, d, dtype=float)
        self.S = np.zeros(N)
        self.I = np.zeros(N)

    def action_probs(self):
        z = self.a * self.V + self.b * self.S + self.c * self.I
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum()

    def update(self, c_t, r_t):
        delta = r_t - self.V[c_t]
        lr = sigmoid(self.alpha_plus) if delta >= 0 else sigmoid(self.alpha_minus)
        self.V[c_t] += lr * delta
        self.S[:] = 0
        self.S[c_t] = 1
        self.I[c_t] += 1


# ---------------------------------------------------------------------------
# Single-sequence NLL / per-trial eval
# ---------------------------------------------------------------------------

def nll_rw_simple(params, df, N=2):
    alpha, theta, d = params
    model = RWModelSimple(N, alpha, theta, d)
    loglik = 0.0
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    for i in range(len(choices)):
        loglik += np.log(model.action_probs()[choices[i]] + 1e-12)
        model.update(choices[i], rewards[i])
    return -loglik


def nll_rw(params, df, N=2):
    model = RWModel(N, *params)
    loglik = 0.0
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    for i in range(len(choices)):
        loglik += np.log(model.action_probs()[choices[i]] + 1e-12)
        model.update(choices[i], rewards[i])
    return -loglik


def eval_rw_per_trial(params, df, participant_id, N=2, use_simple=False):
    """Replay the model over one participant's trials, resetting state fresh.

    Returns one row per trial: participant_id, trial, choice, p_choice, nll.
    """
    model = RWModelSimple(N, *params) if use_simple else RWModel(N, *params)
    rows = []
    trials = df['trial'].to_numpy()
    choices = df['choice'].to_numpy(dtype=int)
    rewards = df['reward'].to_numpy(dtype=float)
    for i in range(len(choices)):
        p = model.action_probs()[choices[i]]
        rows.append({
            'participant_id': participant_id,
            'trial': int(trials[i]),
            'choice': int(choices[i]),
            'reward': float(rewards[i]),
            'p_choice': float(p),
            'nll': float(-np.log(p + 1e-12)),
        })
        model.update(choices[i], rewards[i])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled fitting: one shared parameter vector, per-participant reset state
# ---------------------------------------------------------------------------

def nll_pooled(params, df, nll_fn, participant_col='subject', N=2):
    total = 0.0
    for _, group in df.groupby(participant_col):
        total += nll_fn(params, group.reset_index(drop=True), N=N)
    return total


SIMPLE_PARAM_NAMES = ['alpha', 'theta', 'd']
SIMPLE_BOUNDS = [(0, 1), (0, 50), (0, 10)]
SIMPLE_INITIAL_GUESSES = [
    [0.5, 5.0, 0.3],
    [0.2, 3.0, 0.1],
    [0.8, 8.0, 0.5],
]

FULL_PARAM_NAMES = ['alpha_plus', 'alpha_minus', 'a', 'b', 'c', 'd']
FULL_BOUNDS = [(0, 10)] * 6
FULL_INITIAL_GUESSES = [
    [0.5, 0.3, 5.0, 0.2, 0.05, 0.3],
    [1.0, 0.5, 8.0, 0.5, 0.1, 0.5],
    [0.2, 0.1, 3.0, 0.1, 0.01, 0.1],
]


def fit_pooled(train_df, nll_fn, bounds, initial_guesses, N=2):
    """Fit one shared parameter vector across all participants in train_df."""
    best_result, best_loss = None, float('inf')
    for guess in initial_guesses:
        res = minimize(nll_pooled, x0=guess, args=(train_df, nll_fn, 'subject', N),
                        bounds=bounds, method='L-BFGS-B')
        if res.fun < best_loss:
            best_result, best_loss = res, res.fun
    return best_result.x, best_result


# ---------------------------------------------------------------------------
# Data loading / split
# ---------------------------------------------------------------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[['subject', 'trial', 'reward', 'choice']].copy()
    df['choice'] = df['choice'].astype(int) - 1  # 1/2 -> 0/1
    return df


def split_participants(df, n_test=6, seed=42):
    subjects = np.sort(df['subject'].unique())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects)
    test_ids = sorted(perm[:n_test].tolist())
    train_ids = sorted(perm[n_test:].tolist())
    return train_ids, test_ids


# ---------------------------------------------------------------------------
# End-to-end run for one model variant
# ---------------------------------------------------------------------------

def run_model(df, train_ids, test_ids, use_simple, out_dir):
    if use_simple:
        param_names, bounds, guesses, nll_fn = (
            SIMPLE_PARAM_NAMES, SIMPLE_BOUNDS, SIMPLE_INITIAL_GUESSES, nll_rw_simple,
        )
        label = 'simple (3-param)'
    else:
        param_names, bounds, guesses, nll_fn = (
            FULL_PARAM_NAMES, FULL_BOUNDS, FULL_INITIAL_GUESSES, nll_rw,
        )
        label = 'full (6-param)'

    train_df = df[df['subject'].isin(train_ids)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_ids)].reset_index(drop=True)

    print(f'\n=== Fitting {label} model on {len(train_ids)} training participants ===')
    fitted_params, result = fit_pooled(train_df, nll_fn, bounds, guesses)
    fitted_dict = dict(zip(param_names, fitted_params.tolist()))
    print(f'  Fitted params: { {k: round(v, 4) for k, v in fitted_dict.items()} }')
    print(f'  Train pooled NLL: {result.fun:.2f}  (converged={result.success})')

    print(f'Evaluating on {len(test_ids)} held-out participants (per-trial NLL) …')
    singles_dir = os.path.join(out_dir, 'singles')
    os.makedirs(singles_dir, exist_ok=True)

    total_nll, total_trials = 0.0, 0
    per_participant_summary = []
    for pid in test_ids:
        pdf = test_df[test_df['subject'] == pid].reset_index(drop=True)
        trial_df = eval_rw_per_trial(fitted_params, pdf, participant_id=pid,
                                      use_simple=use_simple)
        trial_df.to_csv(os.path.join(singles_dir, f'participant_{pid}.csv'), index=False)

        p_nll = trial_df['nll'].sum()
        total_nll += p_nll
        total_trials += len(trial_df)
        per_participant_summary.append({
            'participant_id': int(pid),
            'n_trials': len(trial_df),
            'nll': float(p_nll),
            'avg_nll_per_trial': float(p_nll / len(trial_df)),
        })
        print(f'  participant {pid}: NLL={p_nll:.2f}  avg/trial={p_nll/len(trial_df):.4f}')

    print(f'  Total test NLL = {total_nll:.2f}  |  avg NLL/trial = {total_nll/total_trials:.4f}')

    metrics = {
        'model': label,
        'param_names': param_names,
        'fitted_params': fitted_dict,
        'train_participants': [int(x) for x in train_ids],
        'test_participants': [int(x) for x in test_ids],
        'train_pooled_nll': float(result.fun),
        'per_participant_test': per_participant_summary,
        'total_test_nll': float(total_nll),
        'avg_test_nll_per_trial': float(total_nll / total_trials),
    }
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f'  Metrics + params saved -> {os.path.join(out_dir, "params.json")}')

    return fitted_params, metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='RW fitting for rl_waltmann (pooled train / per-trial test NLL)')
    p.add_argument('--data-csv', default='data/in/test_waltmann_data_cleaned.csv')
    p.add_argument('--out-dir', default='data/out')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-test', type=int, default=6)
    args = p.parse_args()

    df = load_data(args.data_csv)
    n_subjects = df['subject'].nunique()
    print(f'Loaded {len(df)} trials from {n_subjects} participants ({args.data_csv})')

    train_ids, test_ids = split_participants(df, n_test=args.n_test, seed=args.seed)
    print(f'Split: {len(train_ids)} train / {len(test_ids)} test  (seed={args.seed})')
    print(f'  test participants: {test_ids}')

    split_dir = os.path.join(args.out_dir, 'rw')
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, 'train_test_split.json'), 'w') as f:
        json.dump({'seed': args.seed, 'train': train_ids, 'test': test_ids}, f, indent=4)

    run_model(df, train_ids, test_ids, use_simple=True,
              out_dir=os.path.join(args.out_dir, 'rw_simple'))
    run_model(df, train_ids, test_ids, use_simple=False,
              out_dir=os.path.join(args.out_dir, 'rw_full'))


if __name__ == '__main__':
    main()
