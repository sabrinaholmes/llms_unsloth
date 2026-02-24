import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import get_models
import os
import hashlib
import gzip
import re

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'centaur-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'
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



def save_prompt_file(prompt_text: str, participant_id: int) -> str:
    """Save a prompt as gzipped UTF-8 text and return the path."""
    h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    filename = f"participant_{participant_id}_{h}.txt.gz"
    path = os.path.join(PROMPT_DIR, filename)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(prompt_text)
    return path
import torch
import pandas as pd
import re

def predict_participant_horizon(participant_df, model, tokenizer):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []

    # 1. Build the full prompt representing the entire game history
    # We use the final state to get the full string, then index into it
    full_prompt = build_multi_game_prompt(participant_df, 
                                          participant_df['game'].max(), 
                                          participant_df['trial'].max())

    # 2. Tokenize once
    encoding = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)

    with torch.no_grad():
        # 3. Single Forward Pass
        outputs = model(input_ids)
        all_logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # 4. Use Regex to find every "choice trigger" in the prompt
        # Pattern matches your specific format: You press <<U>>
        trigger_pattern = r'You press <<([^>]+)>>'
        matches = list(re.finditer(trigger_pattern, full_prompt))

        for trial_idx, match in enumerate(matches):
            choice_char = match.group(1)
            char_pos = match.start(1)
            
            # Map character position to token index
            token_idx = encoding.char_to_token(0, char_pos)
            if token_idx is None: continue

            # ALIGNMENT: The logit predicting the choice is at [token_idx - 1]
            logits = all_logits[token_idx - 1]
            
            # Calculate Probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Get NLL for the actual human choice
            actual_token_id = input_ids[0, token_idx]
            nll = -log_probs[actual_token_id].item()
            
            # 6. Get Top-2 for analysis
            top2_probs, top2_indices = torch.topk(log_probs, 2)
            top2_tokens = tokenizer.convert_ids_to_tokens(top2_indices)
            top2_probs = top2_probs.exp().tolist()

            # Extract trial-specific metadata from the original DF
            # This ensures we match the correct reward/horizon for this trial
            row = participant_df.iloc[trial_idx]

            all_results.append({
                "participant_id": participant_df['participant_id'].iloc[0],
                "game": row["game"],
                "trial_index": row["trial"],
                "ground_truth": choice_char,
                "nll": nll,
                "is_free": row["type"] == "free",
                'top2': list(zip(top2_tokens, top2_probs))
            })
    
    # Compute summary statistics
    valid_trial_nlls = [r['nll'] for r in all_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_trial_nlls) / len(valid_trial_nlls) if valid_trial_nlls else float('inf')

    print(f"✅ Simulation complete")
    print(f"🎯 Overall NLL: {overall_nll:.4f}")

    return all_results, overall_nll, full_prompt

def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)
    # Ensure prompt directory exists
    os.makedirs(PROMPT_DIR, exist_ok=True)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    participant_ids = timeline['participant_id'].unique()

    # Initialize a list to store overall NLLs and prompts for all models
    all_model_results = []

    for participant_id_id in participant_ids:
        print(f"\n🧠 Simulating participant {participant_id_id}")
        out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id_id}.csv'

        if os.path.exists(out_path):
            print(f"Participant {participant_id_id} already simulated. Skipping...")
            continue
        if not os.path.exists(PROMPT_DIR):
            os.makedirs(PROMPT_DIR)
        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['participant_id'] == participant_id_id]
        results, overall_nll, prompt = predict_participant_horizon(model_data, model, tokenizer)
        result = pd.DataFrame(results)

        # Save the results for this model
        result.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

        # Append overall NLL and prompt to the list
        # save full concatenated prompt as gzipped file and record its path
        prompt_path = save_prompt_file(prompt, participant_id_id)
        all_model_results.append({
            'model_id': MODEL,
            'overall_nll': overall_nll,
            'prompt_path': prompt_path
        })

    # Create a DataFrame for all models
    all_model_df = pd.DataFrame(all_model_results)
    all_model_path = f'{DATA_FOLDER_OUT}/all_models_summary.csv'
    all_model_df.to_csv(all_model_path, index=False)
    print(f"Summary of all models saved to {all_model_path}")

if __name__ == "__main__":
    main()
