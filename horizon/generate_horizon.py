import get_models
from get_models import generate, create_text_generation_pipeline
from unsloth import FastLanguageModel
import transformers
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

DATA_IN_ = 'data/in/timeline_structure.csv'
MODEL = 'centaur-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")

LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
CHOICE_OPTIONS=['I', 'H']

def simulate_participant(timeline_df, pipe, choice_options, llm_type='llama'):
    history = []

    for game_id, game_data in timeline_df.groupby('game'):
        # Get horizon (useful for your records)
        horizon = game_data['horizon'].iloc[0] if 'horizon' in game_data.columns else None
        
        # A. Process Forced Trials
        forced_trials = game_data[game_data['type'] == 'forced'].sort_values('trial_num_block')
        for _, row in forced_trials.iterrows():
            history.append({
                "game": game_id, "trial": row['trial_num_block'], "type": "forced",
                "choice": row['choice'], "reward": row['reward'], "horizon": horizon
            })

        # B. Process Free Trials
        free_trials = game_data[game_data['type'] == 'free'].sort_values('trial_num_block')
        
        for i, (_, row) in enumerate(free_trials.iterrows()):
            current_history_df = pd.DataFrame(history)
            prompt = build_multi_game_prompt(current_history_df, game_id, i, trial_col='trial', choice_options=choice_options, llm_type=llm_type)
            #print(prompt)
            # 1. Generate Choice
            model_choice = generate(prompt, pipe)
            if model_choice not in choice_options:
                print(f"⚠️ Invalid choice '{model_choice}' at Game {game_id}, Trial {row['trial_num_block']}. Retrying...")
                # Optional: Re-run with a tiny bit of temperature if it failed
                # For now, let's use a simple fallback to 'I' or 'H' based on random 
                # or just record it as 'INVALID' to filter later.
                model_choice = 'INVALID'

            # 3. Get reward (Fixed syntax)
            if model_choice == choice_options[0]:
                reward = row['reward_I']
            elif model_choice == choice_options[1]:
                reward = row['reward_H']


            history.append({
                "game": game_id,
                "trial": row['trial_num_block'],
                "type": "free",
                "choice": model_choice,
                "reward": reward,
                "prompt":prompt,
                "horizon": horizon
            })

            print(f"Game {game_id} (H={horizon}) Trial {row['trial_num_block']}: {model_choice} -> {reward} pts")

    return pd.DataFrame(history)

def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)
    # import timeline structure
    timeline = pd.read_csv(DATA_IN_)
    participant_ids = timeline['participant_id'].unique()
    # Initialize model
    model,tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    FastLanguageModel.for_inference(model)
    model._past = None  # Reset past states if necessary
    torch.cuda.empty_cache()  # Clear GPU memory again
    pipe=get_models.create_text_generation_pipeline(model,tokenizer,max_new_tokens=1)
    # Run simulation for each seed
    for participant_id in participant_ids:
        out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id}.csv'
        participant_data = timeline[timeline['participant_id'] == participant_id]
        #simulate only first 100 games
        participant_data = participant_data[participant_data['game'] <= 100]
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {participant_id}.")
            continue
        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)
        gc.collect()
        torch.cuda.empty_cache()
        # Run simulation
        history = simulate_participant(participant_data, pipe, CHOICE_OPTIONS, llm_type=LLM_TYPE)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()