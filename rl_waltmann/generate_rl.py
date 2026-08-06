import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_models
import pandas as pd
import numpy as np
import random
import torch
import os
import gc
from rl_prompt import build_llama_prompt, build_centaur_prompt,build_llama_prompt_no_rewards, build_centaur_prompt_no_rewards
import hashlib
import gzip


MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative_no_rewards/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
SIMULATION_NUMBER = 56 # Number of simulated participants
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
NO_REWARDS = True  # Set to True to use prompts that do not include reward information
BASE_SEED = 42  # Base seed; each run_id gets BASE_SEED + run_id so runs are reproducible individually


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_prompt(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    path = os.path.join(PROMPT_DIR, f"{h}.txt.gz")
    if not os.path.exists(path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(prompt)
    return h


def generate_timeline(num_trials=140, seed=None, reversed_state=None,
                       reversal_points=(35, 55, 70, 85, 105), p_correct=0.8):
    """Generates a timeline of trials for the slot machine task, matching the
    real PRLT design: fixed reversal points, and within each between-reversal
    block an exact quota of `p_correct` fraction of trials go to the
    currently-correct bandit (not an independent per-trial Bernoulli draw,
    which lets the realized ratio drift from 0.8/0.2 within a block).
    Outcomes are zero-sum per trial: exactly one bandit gets reward=1.

    Args:
        num_trials: The number of trials to generate.
        seed: RNG seed for the within-block shuffle order. Leave as None so
            each call (each simulated participant) gets an independent draw;
            pass an int only if you want that participant's draw reproducible.
        reversed_state: True if bandit_1 starts as the correct bandit, False
            if bandit_2 does. Pass this in explicitly (e.g. from a
            counterbalanced list in `main`) rather than leaving it random per
            call, so the split across participants is controlled.
        reversal_points: Trial indices (0-indexed) at which the correct
            bandit flips.
        p_correct: Reward probability for the currently-correct bandit.

    Returns:
        A list of dicts, one per trial, with "bandit_1"/"bandit_2" reward values.
    """
    if reversed_state is None:
        reversed_state = random.choice([True, False])

    rng = np.random.default_rng(seed)
    boundaries = [0] + list(reversal_points) + [num_trials]

    timeline = [None] * num_trials
    bandit_1_correct = reversed_state

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block_size = end - start
        n_correct_wins = round(block_size * p_correct)
        outcomes = np.array([1] * n_correct_wins + [0] * (block_size - n_correct_wins))
        rng.shuffle(outcomes)

        for offset, correct_bandit_wins in enumerate(outcomes):
            if bandit_1_correct:
                bandit_1_reward, bandit_2_reward = int(correct_bandit_wins), int(1 - correct_bandit_wins)
            else:
                bandit_2_reward, bandit_1_reward = int(correct_bandit_wins), int(1 - correct_bandit_wins)
            timeline[start + offset] = {
                "bandit_1": {"color": "orange", "value": bandit_1_reward},
                "bandit_2": {"color": "blue", "value": bandit_2_reward}
            }
        bandit_1_correct = not bandit_1_correct  # reversal flips which bandit is correct

    return timeline
    

def simulate_participant(timeline: list, wrapper) -> pd.DataFrame:
    """Simulates a participant with log-likelihood tracking"""
    #reate random choice options from all capital letters
    choice_options = random.sample(
        [chr(i) for i in range(65, 91)], 2
    )

    history = []
    cumulative_reward = 0
    total_trials = 140  # Total number of trials to simulate
    for trial in range(1,total_trials+1):
        current_trial_data = timeline[trial - 1]  # Ensure `timeline` is defined
        if LLM_TYPE == 'centaur':
            if NO_REWARDS:
                prompt_model = build_centaur_prompt_no_rewards(history, choice_options=choice_options)
            else:
                prompt_model = build_centaur_prompt(history, choice_options=choice_options)
        elif LLM_TYPE == 'llama':
            if NO_REWARDS:
                prompt_model = build_llama_prompt_no_rewards(history, choice_options=choice_options)
            else:
                prompt_model = build_llama_prompt(history, choice_options=choice_options)
        bandit_1_value = current_trial_data["bandit_1"]["value"]
        bandit_2_value = current_trial_data["bandit_2"]["value"]
        model_choice = wrapper.generate(prompt_model, choice_options=choice_options, max_new_tokens=1, processor_type="prefix_tree")
        print(f"this is model choice {model_choice}")

        # Map letter back to bandit index (1 or 2); 0 if unrecognized
        if model_choice == choice_options[0]:
            chosen_bandit = 1
            reward = bandit_1_value
        elif model_choice == choice_options[1]:
            chosen_bandit = 2
            reward = bandit_2_value
        else:
            chosen_bandit = 0
            reward = 0
        cumulative_reward += reward

        print(f"Trial {trial}: "
              f"Choice {model_choice} (bandit {chosen_bandit}), "
              f"Reward {reward}, "
              f"Total {cumulative_reward}")

        history.append({
            "trial_index": trial,
            "choice_mapped": model_choice,
            "choice": chosen_bandit,
            "choice_option_1": choice_options[0],
            "choice_option_2": choice_options[1],
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "prompt_hash": save_prompt(prompt_model)
        })



    return pd.DataFrame(history)

def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    # Initialize model
    wrapper = get_models.ModelWrapper(MODEL, use_unsloth=True)
    torch.cuda.empty_cache()

    # counterbalanced assignment of which bandit starts correct, one draw per participant
    # seeded so the assignment is the same across script re-runs
    seed_everything(BASE_SEED)
    reversed_states = [True] * (SIMULATION_NUMBER // 2) + [False] * (SIMULATION_NUMBER - SIMULATION_NUMBER // 2)
    random.shuffle(reversed_states)

    # widely-spread per-run seeds, drawn once from BASE_SEED so the whole array
    # is itself reproducible across script re-runs (not just BASE_SEED, BASE_SEED+1, ...)
    run_seeds = np.random.default_rng(BASE_SEED).integers(0, 2**31 - 1, size=SIMULATION_NUMBER)

    # Run simulation for each seed
    for run_id in range(SIMULATION_NUMBER):
        out_path = f'{DATA_FOLDER_OUT}/participant_{run_id}.csv'
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {run_id}.")
            continue
        gc.collect()
        torch.cuda.empty_cache()
        # fix the seed per run_id so each participant's simulation (choice_options,
        # timeline, and model sampling) is reproducible across re-runs
        run_seed = int(run_seeds[run_id])
        seed_everything(run_seed)
        # fresh, independently-drawn timeline per participant (same as one draw per human subject)
        timeline = generate_timeline(num_trials=140, seed=run_seed, reversed_state=reversed_states[run_id])
        # Run simulation
        history = simulate_participant(timeline, wrapper)
        # Save results, including the seed used for this run for reproducibility
        history['seed'] = run_seed
        # which bandit started as the correct/rewarded arm (1 or 2); reversed_states[run_id]
        # is the randomized/counterbalanced draw used to build the timeline above
        history['initial_correct_bandit'] = 1 if reversed_states[run_id] else 2
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()