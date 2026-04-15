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
from multi_cue_prompt import build_predict_centaur_prompt, build_predict_llama_prompt
from compare_prompts import is_prompt_in_test_set


DATA_IN_TEST = 'data/in/test/test_df_lowercase.csv'  # This should be the test set created by create_test_df.py
MODEL = 'llama-8B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive_lowercase_sequential/{MODEL}/singles'
RUN_TEST_SET_ONLY = False  # Set to True to only run on prompts that are in the test set (for analysis purposes)
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
EXPERIMENT_NAME = 'collsiöö2023MCPL'  # Used to filter the test set participants
RUN_VERBAL_ONLY = True  # If True, only run on participants with verbal conditions (1/3)
ENGLISH_OPTIONS_LOW = [
    'extremely low', 'very low', 'low', 'somewhat low', 'normal',
    'somewhat high', 'high', 'very high', 'extremely high'
]
ENGLISH_OPTIONS_UNDERLINE = [opt.replace(' ', '_') for opt in ENGLISH_OPTIONS_LOW]
if DATA_IN_TEST.endswith('lowercase.csv'):
    ENGLISH_OPTIONS = ENGLISH_OPTIONS_LOW
    print("Using lowercase English options for tokenization")
else:
    ENGLISH_OPTIONS = ENGLISH_OPTIONS_UNDERLINE
    print("Using underlined English options for tokenization")
NUMERIC_OPTIONS = ['10', '20', '30', '40', '50', '60', '70', '80', '90']

def choice_options_for_participant(participant_df):
    """
    Determines the valid choice options for a participant based on their scenario.
    This is used to mask out invalid choices
    """
    participant_conditions = participant_df['condition'].unique()
    #check whether participant has only one condition or multiple conditions
    if len(participant_conditions) == 1:
        condition = participant_conditions[0]
        if condition in [2, 4]: # Numeric choices
            return NUMERIC_OPTIONS
        elif condition in [1, 3]: # English choices:
            return ENGLISH_OPTIONS
        else:
            raise ValueError(f"Unknown condition '{condition}' for participant {participant_df['Fp'].iloc[0]}")
    else:
        # If participant has multiple conditions, we need to determine the valid choices for each trial
        # For simplicity, we can return all possible choices and rely on the prompt to guide the model
        print(f"Participant {participant_df['Fp'].iloc[0]} has multiple conditions: {participant_conditions}")
        return None  # No masking, allow all choices




def predict_participant(participant_df, model, tokenizer):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []
    # select the prompt building function based on the model type
    if LLM_TYPE == 'centaur':
        print("Using Centaur prompt builder")
        trigger_pattern = r'<<([^>]+)>>'
    elif LLM_TYPE == 'llama':
        print("Using LLaMA prompt builder")
        trigger_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>'


    participant_id = participant_df['Fp'].iloc[0]
    if LLM_TYPE == 'centaur':
        full_prompt = build_predict_centaur_prompt(participant_df)
    elif LLM_TYPE == 'llama':
        full_prompt = build_predict_llama_prompt(participant_df)
    # 2. Tokenize once
    encoding = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)

    # Pre-compute token IDs for numeric conditions (2/4) — single token per choice
    numeric_token_ids = tokenizer.convert_tokens_to_ids(NUMERIC_OPTIONS)
    numeric_token_ids = [t for t in numeric_token_ids if t != tokenizer.unk_token_id]

    # Pre-compute first token IDs for English conditions (1/3) — for masking at first-token position
    # Encode with leading space to match mid-sentence tokenization (choices appear after "is ")
    english_first_token_ids = list({
        tokenizer.encode(' ' + opt, add_special_tokens=False)[0]
        for opt in ENGLISH_OPTIONS
        if tokenizer.encode(' ' + opt, add_special_tokens=False)
    })

    # Check if the prompt is in the test set (for analysis purposes)
    in_test_set = is_prompt_in_test_set(full_prompt, experiment_name=EXPERIMENT_NAME)
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

            trial_condition = participant_df.iloc[trial_idx]['condition']
            is_english = trial_condition in [1, 3]

            # Map character position to token index
            token_idx = encoding.char_to_token(0, char_pos)
            if token_idx is None: continue

            # ALIGNMENT: The logit predicting the choice is at [token_idx - 1]
            if is_english:
                # --- Conditions 1/3: multi-token NLL + first-token masking ---

                # Mask at first-token position to valid English first tokens
                logits = all_logits[token_idx - 1].clone()
                mask = torch.full(logits.shape, float('-inf'), device=logits.device)
                mask[english_first_token_ids] = 0.0
                logits = logits + mask

                # Tokenize the actual choice to determine its token span
                choice_token_span = tokenizer.encode(' ' + choice_char, add_special_tokens=False)
                n_tokens = len(choice_token_span)

                # Sum NLL across all tokens of the phrase
                # Also record top-5 distribution at each token position
                nll = 0.0
                top_per_position = []
                for k in range(n_tokens):
                    actual_tok = input_ids[0, token_idx + k]
                    if k == 0:
                        # Use masked logits at first-token position
                        log_probs_k = F.log_softmax(logits, dim=-1)
                    else:
                        # Use unmasked logits at subsequent positions
                        log_probs_k = F.log_softmax(all_logits[token_idx - 1 + k], dim=-1)
                    nll += -log_probs_k[actual_tok].item()
                    top_p, top_i = torch.topk(log_probs_k, 5)
                    top_per_position.append(list(zip(
                        tokenizer.convert_ids_to_tokens(top_i),
                        top_p.exp().tolist()
                    )))

                # Top-5 from the masked first-token distribution (kept for compatibility)
                log_probs_first = F.log_softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(log_probs_first, 5)
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
                top_probs = top_probs.exp().tolist()

            else:
                # --- Conditions 2/4: single-token NLL + numeric masking (unchanged) ---
                logits = all_logits[token_idx - 1]
                mask = torch.full(logits.shape, float('-inf'), device=logits.device)
                mask[numeric_token_ids] = 0.0
                logits = logits + mask

                log_probs = F.log_softmax(logits, dim=-1)
                actual_token_id = input_ids[0, token_idx]
                nll = -log_probs[actual_token_id].item()

                top_probs, top_indices = torch.topk(log_probs, 2)
                top_tokens = tokenizer.convert_ids_to_tokens(top_indices)
                top_probs = top_probs.exp().tolist()
                top_per_position = None

            all_results.append({
                "participant_id": participant_id,
                "trial_index": trial_idx+1,
                "ground_truth": choice_char,
                "nll": nll,
                'top': list(zip(top_tokens, top_probs)),
                'top_per_position': top_per_position,  # list of top-5 (token, prob) at each phrase token position; None for numeric
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
    participant_ids = timeline['Fp'].unique()

    # Initialize a list to store overall NLLs and prompts for all models
    all_model_results = []
    if RUN_VERBAL_ONLY:
        participant_ids = timeline[timeline['condition'].isin([1, 3])]['Fp'].unique()
        print(f"Running only verbal conditions (1/3). Participants: {participant_ids}")

    for p in participant_ids:
        print(f"\n🧠 Simulating participant {p}")
        out_path = f'{DATA_FOLDER_OUT}/participant_' + str(p) + '.csv'

        if os.path.exists(out_path):
            print(f"Participant {p} already simulated. Skipping...")
            continue

        # Run simulation with model and tokenizer passed
        model_data = timeline[timeline['Fp'] == p]

        if RUN_TEST_SET_ONLY and LLM_TYPE == 'centaur':
            full_prompt = build_predict_centaur_prompt(model_data)
            if not is_prompt_in_test_set(full_prompt, experiment_name=EXPERIMENT_NAME):
                print(f"Participant {p} not in test set. Skipping...")
                continue

        results, overall_nll, prompt = predict_participant(model_data, model, tokenizer)

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
