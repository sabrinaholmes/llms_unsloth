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
from rl_prompt import build_llama_prompt, build_centaur_prompt
import hashlib
import gzip


MODEL = 'llama-8B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
SIMULATION_NUMBER = 32 # Number of simulated participants
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'

def save_prompt(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    path = os.path.join(PROMPT_DIR, f"{h}.txt.gz")
    if not os.path.exists(path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(prompt)
    return h


def generate_timeline(num_trials=100, seed=42):
    """Generates a timeline of trials for the slot machine task.

    Args:
        num_trials: The number of trials to generate.
        seed: The initial seed for the random number generator (for reproducibility).

    Returns:
        A DataFrame containing the trial data with columns: 'trial', 'choice', 'reward'.
    """
    random.seed(seed)

    # Number of trials
    num_trials = 100

    # Define the timeline
    timeline = []
    for i in range(num_trials):
        while True:
            if i < (num_trials / 2):
                bandit_1_reward = random.choices([1, 0], weights=[0.8, 0.2])[0]
                bandit_2_reward = random.choices([1, 0], weights=[0.2, 0.8])[0]
            else:
                bandit_1_reward = random.choices([1, 0], weights=[0.2, 0.8])[0]
                bandit_2_reward = random.choices([1, 0], weights=[0.8, 0.2])[0]

            if not (bandit_1_reward == 0 and bandit_2_reward == 0):
                break

        timeline.append({
            "bandit_1": {"color": "orange", "value": bandit_1_reward},
            "bandit_2": {"color": "blue", "value": bandit_2_reward}
        })
    return timeline
    

def simulate_participant(timeline: list, wrapper) -> pd.DataFrame:
    """Simulates a participant with log-likelihood tracking"""
    #reate random choice options from all capital letters
    choice_options = random.sample(
        [chr(i) for i in range(65, 91)], 2
    )

    history = []
    cumulative_reward = 0
    total_trials = 100
    for trial in range(1,total_trials+1):
        current_trial_data = timeline[trial - 1]  # Ensure `timeline` is defined
        if LLM_TYPE == 'centaur':
            prompt_model = build_centaur_prompt(history, choice_options=choice_options)
        elif LLM_TYPE == 'llama':
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
    # generate the timeline once
    timeline = generate_timeline(num_trials=100)
    # Initialize model
    wrapper = get_models.ModelWrapper(MODEL, use_unsloth=True)
    torch.cuda.empty_cache()

    # Run simulation for each seed
    for run_id in range(SIMULATION_NUMBER):
        out_path = f'{DATA_FOLDER_OUT}/participant_{run_id}.csv'
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {run_id}.")
            continue
        gc.collect()
        torch.cuda.empty_cache()
        # Run simulation
        history = simulate_participant(timeline, wrapper)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
