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

from horizon_prompt import build_full_prompt

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive_new_2/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")


def save_prompt_file(prompt_text: str, participant_id: int) -> str:
    """Save a prompt as gzipped UTF-8 text and return the path."""
    h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    filename = f"participant_{participant_id}_{h}.txt.gz"
    path = os.path.join(PROMPT_DIR, filename)
    os.makedirs(PROMPT_DIR, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(prompt_text)
    return path


def predict_participant_horizon(df_participant, model, tokenizer):
    full_prompt = build_prediction_prompt(df_participant)
    free_trials_df = df_participant[df_participant['type'] == 'free'].sort_values(['game', 'trial'])
    free_trials_iter = free_trials_df.iterrows()

    # Tokenize and get offsets
    inputs = tokenizer(full_prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = inputs.pop('offset_mapping')[0].tolist()
    input_ids = inputs['input_ids'][0]

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    per_trial_results = []

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        
        # This regex identifies where the Assistant choice is located
        choice_pattern = r"(?<=<\|start_header_id\|>assistant<\|end_header_id\|>)\s*([IH])\s*(?=<\|eot_id\|>)"
        matches = list(re.finditer(choice_pattern, full_prompt, flags=re.IGNORECASE))

        for match in matches:
            try:
                _, trial_row = next(free_trials_iter)
            except StopIteration:
                break

            # char_start is the index of the 'I' or 'H' in the string
            char_start = match.start(1)
            
            # Find which token corresponds to that character
            token_idx = None
            for ti, (o_start, o_end) in enumerate(offsets):
                if o_start <= char_start < o_end and o_start != o_end:
                    token_idx = ti
                    break

            if token_idx is None:
                continue

            # --- RAW NLL CALCULATION ---
            # We take the logits from the position preceding the actual token
            target_logits = logits[token_idx - 1]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
            
            # Identify the actual token ID that was used in the prompt
            actual_token_id = input_ids[token_idx]
            nll = -log_probs[actual_token_id].item()

            # Top-2 for debugging (shows you what else the model was considering)
            top2_probs, top2_indices = torch.topk(log_probs, 2)
            top2_decoded = [tokenizer.decode([idx]).strip() for idx in top2_indices.tolist()]
            top2_val = list(zip(top2_decoded, top2_probs.exp().tolist()))

            per_trial_results.append({
                'game': trial_row['game'],
                'trial_index': trial_row['trial'],
                'horizon': trial_row['horizon'],
                'ground_truth': trial_row['choice'], 
                'nll': nll,
                'top2': top2_val
            })

    valid_nlls = [r['nll'] for r in per_trial_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_nlls) / len(valid_nlls) if valid_nlls else float('inf')

    return per_trial_results, overall_nll, full_prompt


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
        # Check whether top-2 tokens are only I/H for all trials
        def check_top2_tokens_IH(trial_results_one_pass):
            all_top2_tokens_are_IH = True
            for trial_result in trial_results_one_pass:
                top2_tokens_list = [token_tuple[0] for token_tuple in trial_result.get('top2', [])]
                # normalize decoded tokens (remove non-letters, upper-case) before checking
                normed = [re.sub(r'[^A-Za-z]', '', t).upper() for t in top2_tokens_list]
                if not all(token in ['I', 'H'] for token in normed):
                    print(f"Trial {trial_result.get('trial_index')}: Top2 tokens contain non-I/H values: {top2_tokens_list}")
                    all_top2_tokens_are_IH = False
            if all_top2_tokens_are_IH:
                print("All top2 tokens across all trials consist only of 'I' and 'H'.")
            else:
                print("Some top2 tokens across trials contain values other than 'I' and 'H'.")
            return all_top2_tokens_are_IH

        # Run the check and print result
        check_top2_tokens_IH(results)
        # save number of tokens that were not I/H in top-2 across all trials        
        non_IH_top2_count = sum(
            1 for trial_result in results
            if not all(
                token in ['I', 'H']
                for token in [re.sub(r'[^A-Za-z]', '', token_tuple[0]).upper() for token_tuple in trial_result.get('top2', [])]
            )
        )
        print(f"Number of trials with non-I/H top-2 tokens: {non_IH_top2_count}")
        print(f"Results saved to {out_path}")

        # Append overall NLL and prompt to the list
        # save full concatenated prompt as gzipped file and record its path
        prompt_path = save_prompt_file(prompt, participant_id_id)
        all_model_results.append({
            'model_id': MODEL,
            'overall_nll': overall_nll,
            'non_IH_top2_count': non_IH_top2_count,
            'prompt': prompt
        })

    # Create a DataFrame for all models
    all_model_df = pd.DataFrame(all_model_results)
    all_model_path = f'{DATA_FOLDER_OUT}/all_models_summary.csv'
    all_model_df.to_csv(all_model_path, index=False)
    print(f"Summary of all models saved to {all_model_path}")

if __name__ == "__main__":
    main()
