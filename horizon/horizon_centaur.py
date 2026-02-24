import get_models
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

DATA_IN_ = 'data/in/timeline_structure.csv'
MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")



def format_single_game_history(game_id, game_df, is_current_game=False, current_trial=None,trial_col=None):
    """Formats a single game's header and trials into a block of text."""
    total_trials = game_df[trial_col].max()
    
    # Header for the game
    lines = [f"Game {game_id}. There are {total_trials} trials in this game."]
    
    # 1. Add Forced Trials (always included)
    forced = game_df[game_df["type"] == "forced"]
    for _, row in forced.iterrows():
        lines.append(f"You are instructed to press {row['choice']} and get {row['reward']} points.")
    
    # 2. Add Free Trials
    # If it's the current game, only add trials BEFORE the current decision
    if is_current_game:
        free = game_df[(game_df["type"] == "free") & (game_df[trial_col] < current_trial)]
    else:
        free = game_df[game_df["type"] == "free"]
        
    for _, row in free.iterrows():
        lines.append(f"You press <<{row['choice']}>> and get {row['reward']} points.")
        
    return "\n".join(lines)

def build_multi_game_prompt(block_df, current_game_id, current_trial, trial_col='trial'):
    """Constructs the full prompt history across all games in a block."""
    
    # Initial Instructions (Only show this once at the very top)
    full_prompt = [
        "You are participating in multiple games involving two slot machines, labeled I and H",
        "The two slot machines are different across different games.",
        "Each time you choose a slot machine, you get some points.",
        "You choose a slot machine by pressing the corresponding key.",
        "Each slot machine tends to pay out about the same amount of points on average.",
        "Your goal is to choose the slot machines that will give you the most points across the experiment",
        "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose."
        "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials"
    ]
    # 1. Add all COMPLETED games
    past_games_ids = sorted(block_df[block_df['game'] < current_game_id]['game'].unique())
    for g_id in past_games_ids:
        game_data = block_df[block_df['game'] == g_id]
        full_prompt.append(format_single_game_history(g_id, game_data,trial_col=trial_col))
    
    # 2. Add the CURRENT game history
    current_game_data = block_df[block_df['game'] == current_game_id]
    full_prompt.append(format_single_game_history(
        current_game_id, 
        current_game_data, 
        is_current_game=True, 
        current_trial=current_trial,
        trial_col=trial_col  # <--- Essential fix
    ))    
    # 3. Add the final trigger
    return "\n".join(full_prompt) + "\nYou press <<"

def simulate_participant(timeline_df, pipe):
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
            prompt = build_multi_game_prompt(current_history_df, game_id, i)
            #print(prompt)
            # 1. Generate Choice
            model_choice = get_models.generate(prompt, pipe)
            if model_choice not in ['I', 'H']:
                print(f"⚠️ Invalid choice '{model_choice}' at Game {game_id}, Trial {row['trial_num_block']}. Retrying...")
                # Optional: Re-run with a tiny bit of temperature if it failed
                # For now, let's use a simple fallback to 'I' or 'H' based on random 
                # or just record it as 'INVALID' to filter later.
                model_choice = 'INVALID'

            # 3. Get reward (Fixed syntax)
            if model_choice == 'I':
                reward = row['reward_I']
            else:
                reward = row['reward_H']

            # 4. Save prompt to compressed file and update History with path
            def save_prompt_file(prompt_text: str, game: int, trial: int) -> str:
                h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:10]
                filename = f"g{game}_t{trial}_{h}.txt.gz"
                path = os.path.join(PROMPT_DIR, filename)
                # write gzipped UTF-8 text
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(prompt_text)
                return path

            prompt_path = save_prompt_file(prompt, game_id, row['trial_num_block'])

            history.append({
                "game": game_id,
                "trial": row['trial_num_block'],
                "type": "free",
                "choice": model_choice,
                "reward": reward,
                "prompt_path": prompt_path,
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
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping simulation for participant {participant_id}.")
            continue
        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)
        gc.collect()
        torch.cuda.empty_cache()
        # Run simulation
        history = simulate_participant(participant_data,pipe)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
