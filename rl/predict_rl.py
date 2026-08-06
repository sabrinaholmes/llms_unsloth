import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_models
import re
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import os
from rl_prompt import build_llama_prompt, build_centaur_prompt, build_llama_prompt_no_rewards, build_centaur_prompt_no_rewards

MODEL = 'llama-70B-adapter'  # Change this to the desired model name
SIMULATION_NUMBER = 32 # Number of simulated participants
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
DATA_FOLDER_OUT = f'data/out/predictive_no_rewards/{MODEL}/singles'
DATA_IN_TEST = 'data/in/test_data_randomized_choices_flipped_rewards.csv'  # This should be the test set created by create_test_df.py
NO_REWARDS = True  # Set to True to use prompts that do not include reward information

def predict_participant(df_participant, model, tokenizer):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []
    participant_id = df_participant['model_id'].iloc[0]
    #reate random choice options from all capital letters
    choice_options = df_participant['choice_mapped'].unique().tolist()
    history = []
    cumulative_reward = 0
    total_trials = len(df_participant)
    print(f"Total trials: {total_trials}")

    # Build the prompt once using all trials
    past_trials = []
    for trial in range(total_trials):
        row = df_participant.iloc[trial]
        past_trials.append({
            "trial": row['trial'],
            "choice": row['choice'],
            "choice_mapped": row['choice_mapped'],
            "reward": row['reward'],
            "cumulative_reward": df_participant.iloc[:trial+1]['reward'].sum()
        })
    if LLM_TYPE == 'centaur':
        if NO_REWARDS:
            prompt = build_centaur_prompt_no_rewards(past_trials, choice_options=choice_options)
        else:
            prompt = build_centaur_prompt(past_trials, choice_options=choice_options)
        trigger_pattern = r'You press <<([^>]+)>>'
    elif LLM_TYPE == 'llama':
        if NO_REWARDS:
            prompt = build_llama_prompt_no_rewards(past_trials, choice_options=choice_options)
        else:
            prompt = build_llama_prompt(past_trials, choice_options=choice_options)
        trigger_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n([A-Z])'
    
    # 2. Tokenize ONCE and keep the BatchEncoding object
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)
    
    # 3. Pre-compute valid choice token IDs (if provided)
    if choice_options:
        choice_token_ids = tokenizer.convert_tokens_to_ids(choice_options)
        choice_token_ids = [tok_id for tok_id in choice_token_ids if tok_id != tokenizer.unk_token_id]
    else:
        choice_token_ids = None

    with torch.no_grad():
        # 3. Single Forward Pass
        outputs = model(input_ids)
        all_logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # 4. Use Regex to find every "choice trigger" in the prompt
        matches = list(re.finditer(trigger_pattern, prompt))

        for trial_idx, match in enumerate(matches):
            choice_char = match.group(1)
            char_pos = match.start(1)
            
            # Map character position to token index
            token_idx = encoding.char_to_token(0, char_pos)
            if token_idx is None: continue

            # ALIGNMENT: The logit predicting the choice is at [token_idx - 1]
            logits = all_logits[token_idx - 1]
            
            # 6. Mask invalid choices if choice_options were provided
            if choice_token_ids is not None:
                mask = torch.full(logits.shape, float("-inf"), device=logits.device)
                mask[choice_token_ids] = 0.0
                logits = logits + mask
            
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

            all_results.append({
                "participant_id": participant_id,
                "trial_index": trial_idx+1,
                "ground_truth": choice_char,
                "nll": nll,
                'top2': list(zip(top2_tokens, top2_probs))
            })
    
    # Compute summary statistics
    valid_trial_nlls = [r['nll'] for r in all_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_trial_nlls) / len(valid_trial_nlls) if valid_trial_nlls else float('inf')

    print(f"✅ Simulation complete")
    print(f"🎯 Overall NLL: {overall_nll:.4f}")

    return all_results, overall_nll, prompt


def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    model_ids = timeline['model_id'].unique()


    # Initialize a list to store overall NLLs and prompts for all models
    all_model_results = []

    for model_id in model_ids:
        print(f"\n🧠 Simulating model {model_id}")
        out_path = f'{DATA_FOLDER_OUT}/model_' + str(model_id) + '.csv'

        if os.path.exists(out_path):
            print(f"Model {model_id} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['model_id'] == model_id]
        results, overall_nll, prompt = predict_participant(model_data, model, tokenizer)
        result = pd.DataFrame(results)

        # Save the results for this model
        result.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

        # Append overall NLL and prompt to the list
        all_model_results.append({
            'model_id': model_id,
            'overall_nll': overall_nll,
            'prompt': prompt
        })

    # Create a DataFrame for all models
    all_model_df = pd.DataFrame(all_model_results)
    all_model_path = f'{DATA_FOLDER_OUT}/all_models_summary.csv'
    all_model_df.to_csv(all_model_path, index=False)
    print(f"Summary of all models saved to {all_model_path}")

if __name__ == "__main__":
    main()
