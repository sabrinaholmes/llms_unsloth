import get_models
from unsloth import FastLanguageModel
import pandas as pd
import numpy as np
import random
import torch
import os
import gc


MODEL = 'centaur-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
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
    

def build_rl_prompt(past_trials: list) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    instructions = [
        "In this task, you have to repeatedly choose between two slot machines labeled U and P.",
        "You can choose a slot machine by pressing its corresponding key.",
        "When you select one of the machines, you will win 1 or 0 points.",
        "Your goal is to choose the slot machines that will give you the most points.",
        "You will receive feedback about the outcome after making a choice.",
        "The environment may change unpredictably, and past success does not guarantee future results.",
        "You’ll need to adapt to these changes to keep finding the better machine."
        "You will play 1 game in total, consisting of 100 trials.\n",
        "Game 1:"
    ]
    
    prompt = "\n".join(instructions)
    # join instruction with \n
    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}>> and get {past_trial['reward']} points.\n"

    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

def simulate_participant(timeline: list, pipe) -> pd.DataFrame:
    """Simulates a participant with log-likelihood tracking"""
    history = []
    cumulative_reward = 0
    total_trials = 100


    for trial in range(1,total_trials+1):
        current_trial_data = timeline[trial - 1]  # Ensure `timeline` is defined
        prompt_model = build_rl_prompt(history)
        bandit_1_value = current_trial_data["bandit_1"]["value"]
        bandit_2_value = current_trial_data["bandit_2"]["value"]
        #print(f"this is {prompt_model}")
        model_choice = get_models.generate(prompt_model,pipe)
        #print(f"this is choice raw {choice_raw}")
        print(f"this is model choice {model_choice}")

        # Determine reward
        reward = bandit_1_value if model_choice == 'U' else (bandit_2_value if model_choice == 'P' else 0)
        cumulative_reward += reward
        #outputs=generate_test(prompt_model,pipe)
        #print(f"this is whole output {outputs}")

        print(f"Trial {trial}: "
              f"Choice {model_choice}, "
              f"Reward {reward}, "
              f"Total {cumulative_reward}")

        history.append({
            "trial_index": trial,
            "choice": model_choice,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "prompt": prompt_model
        })



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
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {run_id}.")
            continue
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
