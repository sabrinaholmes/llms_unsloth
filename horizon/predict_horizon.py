import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import random
import torch
import get_models
import os
import hashlib
import gzip
import re
from horizon_prompt import build_multi_game_prompt, define_choice_options_from_df
from compare_prompts import is_prompt_in_test_set

DATA_IN_TEST = 'data/in/test_horizon_all_experiments_choice_mapped.csv'

MODEL = 'centaur-8B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
LLM_TYPE="llama" if 'llama' in MODEL.lower() else "centaur"
RUN_TESTS_ONLY = True if LLM_TYPE == "centaur" else False  # Set to False to skip test set checks
RUN_TEST_MODE = True if '8B' in MODEL else False  # If True, restricts prompts to first 2000 chars for faster testing
EXPERIMENT_NAME = 'wilson2014humans'  # Used to filter the test set participants (only relevant if RUN_TESTS is True)

def predict_participant_horizon(participant_df, model, tokenizer,llm_type=LLM_TYPE):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []
    choice_options = define_choice_options_from_df(participant_df)
    participant = participant_df['participant'].iloc[0]
    # 1. Build the full prompt representing the entire game history
    # We use the final state to get the full string, then index into it
    prompt = build_multi_game_prompt(participant_df, 
                                          participant_df['game'].max(), 
                                          participant_df['trial'].max()+1,
                                          choice_options=choice_options,
                                          llm_type=LLM_TYPE,
                                          eval=False)
    # check the token length of the prompt
    tokenized_len = len(tokenizer(prompt)['input_ids'])
    print(f"Initial prompt token length: {tokenized_len}")
    if LLM_TYPE == 'llama':
        # check the prompt token length and truncate if it exceeds model limits (e.g., 32768 for 8k context)
        tokenized_len = len(tokenizer(prompt)['input_ids'])
        if tokenized_len > 32768:
            print(f"⚠️ Prompt token length ({tokenized_len}) exceeds model context window. Truncating to first 32768 tokens.")
            prompt = tokenizer.decode(tokenizer(prompt)['input_ids'][:32768])
    # Check if the prompt is in the test set (for debugging)
    if RUN_TESTS_ONLY:
        if is_prompt_in_test_set(prompt, experiment_name=EXPERIMENT_NAME):
            print("✅ Prompt found in test set. Proceeding with evaluation.")
        else:
            print("⚠️ Prompt NOT found in test set. This may affect evaluation validity.")
                
    if RUN_TEST_MODE:
        #restrict to first 2000 chars for test mode to speed up debugging
        prompt = prompt[:5000]
    if LLM_TYPE == 'centaur':
        trigger_pattern = r'You press <<([^>]+)>>'
    elif LLM_TYPE == 'llama':
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
                "participant_id": participant,
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
    
    # Initialize a list to store overall NLLs and prompts for all models
    all_model_results = []
    for exp in timeline['exp'].unique():
        print(f"\n🔍 Processing experiment: {exp}")
        exp_df = timeline[timeline['exp'] == exp]
        participant_ids = exp_df['participant'].unique()
        print(f"Found {len(participant_ids)} participants in experiment {exp}")
        print(f"Participant IDs: {participant_ids}")
        for participant_id_id in participant_ids:
            print(f"\n🧠 Simulating participant {participant_id_id}")
            out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id_id}_exp_{exp}.csv'

            if os.path.exists(out_path):
                continue
            # Run simulation with model and tokenizer passed
            model_data = exp_df[exp_df['participant'] == participant_id_id]
            results, overall_nll, prompt = predict_participant_horizon(model_data, model, tokenizer)
            result = pd.DataFrame(results)

            # Save the results for this model
            result.to_csv(out_path, index=False)
            print(f"Results saved to {out_path}")

            # Append overall NLL and prompt to the list
            all_model_results.append({
                'model_id': MODEL,
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
