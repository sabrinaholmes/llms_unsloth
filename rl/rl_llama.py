import get_models
from unsloth import FastLanguageModel
import pandas as pd
import numpy as np
import random
import torch
import os
import gc


MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/{MODEL}/singles'
SIMULATION_NUMBER = 32


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
    

def build_generate_prompt(current_trial: int, past_trials: list, total_trials: int) -> str:
    """
    Builds a multi-turn prompt where each past choice is an 'assistant' message
    and each reward is a 'user' message.
    """
    # 1. System Prompt (The Rules)
    system_msg = [
        "In this task, you have to repeatedly choose between two slot machines labeled U and P."
        "When you select one of the machines, you will win 1 or 0 points."
        "Your goal is to choose the slot machines that will give you the most points."
        "You will receive feedback about the outcome after making a choice."
        "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
        "You will play 1 game in total, consisting of 100 trials."
        "Respond with ONLY the character 'U' or 'P'."
    ]
    system_text = "".join(system_msg)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_text}<|eot_id|>"
    # 2. THE FIX: Add the initial User prompt to start the game
    prompt += f"<|start_header_id|>user<|end_header_id|>\nGame 1."
    
    if not past_trials:
        prompt += "No trials completed yet."
    else:
        # Build a concise table of history
        # 3. Interleave History
        for trial in past_trials:
            # The choice is what the Assistant did
            prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{trial['choice']}<|eot_id|>"
            # The reward is what the User (Environment) gave back
            prompt += f"<|start_header_id|>user<|end_header_id|>\n->{trial['reward']} points."
    
    # 3. Final Assistant Header
    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def simulate_participant(timeline: list, pipe):
    history = []
    cumulative_reward = 0
    total_trials = 100

    for trial_idx in range(total_trials):
        current_trial_num = trial_idx + 1

        # Build prompt using existing history
        prompt_model = build_generate_prompt(current_trial_num, history, total_trials)

        # Generate the single character 'U' or 'P'
        # Tip: Set max_new_tokens=2 and temperature=0 for deterministic play

        # Extract the new choice (it will be the very last character of the output)
        model_choice = get_models.generate(prompt_model, pipe)

        # Determine reward from timeline
        trial_data = timeline[trial_idx]
        reward = trial_data["bandit_1"]["value"] if model_choice == 'U' else trial_data["bandit_2"]["value"]

        cumulative_reward += reward

        # Append to history for the NEXT prompt build
        history.append({
            "trial_index": current_trial_num,
            "choice": model_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "prompt": prompt_model
        })

        print(f"Trial {current_trial_num}: {model_choice} -> {reward} pts (Total: {cumulative_reward})")

    return pd.DataFrame(history)

def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)
    # generate the timeline once
    timeline = generate_timeline(num_trials=100)
    # Initialize model
    model,tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    FastLanguageModel.for_inference(model)
    model._past = None  # Reset past states if necessary
    torch.cuda.empty_cache()  # Clear GPU memory again
    pipe=get_models.create_text_generation_pipeline(model,tokenizer,max_new_tokens=1)

    # Run simulation for each seed
    for run_id in range(SIMULATION_NUMBER):
        out_path = f'{DATA_FOLDER_OUT}/participant_{run_id}.csv'
        gc.collect()
        torch.cuda.empty_cache()
        # Run simulation
        history = simulate_participant(timeline,pipe)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
