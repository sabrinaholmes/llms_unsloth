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
from spatially_correlated_prompt import build_predict_centaur_prompt,build_predict_llama_prompt
from compare_prompts import is_prompt_in_test_set


DATA_IN_TEST = 'data/in/test/test_df.csv'  # This should be the test set created by create_test_df.py
MODEL = 'llama-8B-base-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'
TASK_TYPE = 'accumulation' # Set to 'accumulation' or 'maximization' based on the scenario
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
RUN_TEST_SET_ONLY = True if LLM_TYPE == 'centaur' else False  # Set to True to only run on prompts that are in the test set (for analysis purposes)


def predict_participant(participant_df, model, tokenizer,choice_options=None):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []
    # select the prompt building function based on the model type
    if LLM_TYPE == 'centaur':
        print("Using Centaur prompt builder")
        trigger_pattern = r'You press <<([^>]+)>>'
        if TASK_TYPE is not None:
            print(f"Building prompt with specified task type: {TASK_TYPE}")
            full_prompt = build_predict_centaur_prompt(participant_df, task_type=TASK_TYPE)
        else:
            print(f"Building prompt with task type determined by participant scenario")
            full_prompt = build_predict_centaur_prompt(participant_df)
    elif LLM_TYPE == 'llama':
        print("Using LLaMA prompt builder")
        trigger_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(\d+)'
        if TASK_TYPE is not None:
            print(f"Building prompt with specified task type: {TASK_TYPE}")
            full_prompt = build_predict_llama_prompt(participant_df, task_type=TASK_TYPE)
        else:
            print(f"Building prompt with task type determined by participant scenario")
            full_prompt = build_predict_llama_prompt(participant_df)

    participant_id = participant_df['id'].iloc[0]
    # 2. Tokenize once
    encoding = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)
    
    # 3. Pre-compute valid choice token IDs (if provided)
    if choice_options:
        choice_token_ids = tokenizer.convert_tokens_to_ids(choice_options)
        choice_token_ids = [tok_id for tok_id in choice_token_ids if tok_id != tokenizer.unk_token_id]
    else:
        choice_token_ids = None
    # Check if the prompt is in the test set (for analysis purposes)
    in_test_set = is_prompt_in_test_set(full_prompt)
    print(f"Participant {participant_id} - Prompt in test set: {in_test_set}")

    with torch.no_grad():
        # 3. Single Forward Pass
        outputs = model(input_ids)
        all_logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]

        # 4. Use Regex to find every "choice trigger" in the prompt
        matches = list(re.finditer(trigger_pattern, full_prompt))

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

    return all_results, overall_nll, full_prompt

def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    participant_ids = timeline['id'].unique()

    # Initialize a list to store overall NLLs and prompts for all models
    all_model_results = []

    for p in participant_ids:
        print(f"\n🧠 Simulating participant {p}")
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(p) + '.csv'

        if os.path.exists(out_path):
            print(f"Participant {p} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['id'] == p]

        if RUN_TEST_SET_ONLY:
            full_prompt = build_predict_centaur_prompt(model_data, task_type=TASK_TYPE)
            if not is_prompt_in_test_set(full_prompt):
                print(f"Participant {p} not in test set. Skipping...")
                continue

        results, overall_nll, prompt = predict_participant(
            model_data, model, tokenizer,
            choice_options=[str(i) for i in range(1, 31)]
        )
        result = pd.DataFrame(results)

        # Save the results for this model
        result.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

        # Append overall NLL and prompt to the list
        all_model_results.append({
            'model_id': p,
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
