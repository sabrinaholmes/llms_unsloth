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
import hashlib
import gzip
import horizon_prompt
from horizon_prompt import build_multi_game_prompt, define_choice_options_from_df

DATA_IN_ = 'data/in/timeline_structure_mapped.csv'
MODEL = 'llama-8B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
RUN_TEST_MODE = True if '8B' in MODEL else False  # If True, restricts simulations to first 20 games for faster testing
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
SIMULATION_GAME_LIMIT = 20 if RUN_TEST_MODE else 100

def save_prompt(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    path = os.path.join(PROMPT_DIR, f"{h}.txt.gz")
    if not os.path.exists(path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(prompt)
    return h


def simulate_participant(timeline_df, wrapper,llm_type='llama'):
    history = []
    choice_numeric = None
    reward = None
    choice_options = random.sample(
        [chr(i) for i in range(65, 91)], 2
    )
    print(f"Choice options for participant {timeline_df['participant_id'].iloc[0]}: {choice_options}")
    # map choices to letters
    timeline_df['mapped_choice'] = timeline_df.apply(lambda row: choice_options[0] if row['choice'] == 1 else choice_options[1], axis=1)
    #print(timeline_df.head())
    for game_id, game_data in timeline_df.groupby('game'):
        # Get horizon (useful for your records)
        horizon = game_data['horizon'].iloc[0] if 'horizon' in game_data.columns else None
        
        # A. Process Forced Trials
        forced_trials = game_data[game_data['type'] == 'forced'].sort_values('trial_num_block')
        for _, row in forced_trials.iterrows():
            history.append({
                "game": game_id, "trial": row['trial_num_block'], "type": "forced",
                "choice": row['choice'],"mapped_choice": row['mapped_choice'], "reward": row['reward'], "horizon": horizon
            })

        # B. Process Free Trials
        free_trials = game_data[game_data['type'] == 'free'].sort_values('trial_num_block')
        
        for i, (_, row) in enumerate(free_trials.iterrows()):
            current_history_df = pd.DataFrame(history)
            prompt = build_multi_game_prompt(current_history_df, game_id, i, trial_col='trial', choice_options=choice_options, llm_type=llm_type, eval=True)
            #print(prompt)
            # 1. Generate Choice
            model_choice = wrapper.generate(prompt,choice_options=choice_options,max_new_tokens=1,processor_type='prefix_tree')
            model_choice = model_choice.strip()
            if model_choice not in choice_options:
                print(f"⚠️ Invalid choice '{model_choice}' at Game {game_id}, Trial {row['trial_num_block']}. Retrying...")
                # Optional: Re-run with a tiny bit of temperature if it failed
                # For now, let's use a simple fallback to 'I' or 'H' based on random 
                # or just record it as 'INVALID' to filter later.
                model_choice = 'INVALID'

            # 3. Get reward (Fixed syntax)
            if model_choice == choice_options[0]:
                reward = row['reward_1']
                choice_numeric = 1
            elif model_choice == choice_options[1]:
                reward = row['reward_2']
                choice_numeric = 2

            history.append({
                "game": game_id,
                "trial": row['trial_num_block'],
                "type": "free",
                "mapped_choice": model_choice,
                "choice_numeric": choice_numeric,
                "reward": reward,
                "prompt_hash": save_prompt(prompt),
                "horizon": horizon
            })

            print(f"Game {game_id} (H={horizon}) Trial {row['trial_num_block']}: {model_choice} -> {reward} pts")

    return pd.DataFrame(history)

def main():

    os.makedirs(DATA_FOLDER_OUT, exist_ok=True)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    # import timeline structure
    timeline = pd.read_csv(DATA_IN_)
    participant_ids = timeline['participant_id'].unique()
    # Initialize model
    wrapper = get_models.ModelWrapper(MODEL, use_unsloth=True)
    torch.cuda.empty_cache()
    # Run simulation for each seed
    for participant_id in participant_ids:
        out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id}.csv'
        participant_data = timeline[timeline['participant_id'] == participant_id]
        #simulate only first 100 games
        participant_data = participant_data[participant_data['game'] <= SIMULATION_GAME_LIMIT]
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {participant_id}.")
            continue
        gc.collect()
        torch.cuda.empty_cache()
        # Run simulation
        history = simulate_participant(participant_data, wrapper,llm_type=LLM_TYPE)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()