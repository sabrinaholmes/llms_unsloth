import get_models
from unsloth import FastLanguageModel
import transformers
import pandas as pd
import numpy as np
import random
import torch
import os
import gc


MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/{MODEL}_unsloth_seeds/singles'



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
    # 1. System Identity
    prompt = (
       "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "In this task, you have to repeatedly choose between two slot machines labeled U and P.\n"
        "You can choose a slot machine by pressing its corresponding key."
        "When you select one of the machines, you will win 1 or 0 points."
        "Your goal is to choose the slot machines that will give you the most points."
        "You will receive feedback about the outcome after making a choice.\n"
        "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
        f"You will play 1 game in total, consisting of {total_trials} trials."
        " Game 1:"
        "Respond with ONLY the character 'U' or 'P'.<|eot_id|>"
    )

    # 2. History of turns
    if not past_trials:
        # First turn trigger
        prompt += (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Trial 1 of {total_trials}. Make your choice.<|eot_id|>"
        )
    else:
        # Reconstruct the conversation
        for i, trial in enumerate(past_trials):
            # What the user (environment) said
            if i == 0:
                user_msg = f"Trial 1 of {total_trials}. Make your choice."
            else:
                user_msg = f"Result: {past_trials[i-1]['reward']} points. Trial {trial['trial_num']} of {total_trials}."

            prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"

            # What the model (assistant) chose
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{trial['choice']}<|eot_id|>"

        # 3. The current trial request
        last_reward = past_trials[-1]['reward']
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Result: {last_reward} points. Trial {current_trial} of {total_trials}. "
            "Make your choice.<|eot_id|>"
        )

    # 4. Final Assistant Header to trigger generation
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    #print(prompt)

    return 


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
    for run_id, seed in enumerate(seeds):
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(seed) + '.csv'
        gc.collect()
        torch.cuda.empty_cache()
        fix_seed(seed)  # Ensure reproducibility
        # Run simulation
        history = simulate_participant(timeline,pipe)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
