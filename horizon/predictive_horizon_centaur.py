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

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'centaur-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive_h100/{MODEL}/singles'
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

def predict_participant_horizon(participant_df, model, tokenizer,llm_type='centaur'):
    """
    Simulates a participant by processing the entire game history in ONE forward pass.
    Calculates NLL, Top-2 probabilities, and aligns model predictions with human choices.
    """
    all_results = []
    choice_options = define_choice_options_from_df(participant_df)
    # Map choice labels (e.g., 'J', 'R') to their tokenizer token ids (first token)
    raw_choice_labels = [str(c).strip() for c in choice_options]
    choice_token_ids = {}
    for lbl in raw_choice_labels:
        enc = tokenizer(lbl, add_special_tokens=False)
        if enc.get('input_ids'):
            choice_token_ids[lbl.upper()] = enc['input_ids'][0]
        else:
            print(f"⚠️ Warning: Could not tokenize choice label '{lbl}'")
    # 1. Build the full prompt representing the entire game history
    # We use the final state to get the full string, then index into it
    full_prompt = build_multi_game_prompt(participant_df, 
                                          participant_df['game'].max(), 
                                          participant_df['trial'].max(),
                                          choice_options=choice_options,
                                          llm_type='centaur')

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
            if token_idx is None:
                print(f"⚠️ Warning: Could not map choice at position {char_pos} to a token.")
                continue

            # ALIGNMENT: The logit predicting the choice is at [token_idx - 1]
            logits = all_logits[token_idx - 1]
            
            # Calculate Probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Get NLL for the actual human choice
            actual_token_id = int(input_ids[0, token_idx].item())
            raw_nll = -log_probs[actual_token_id].item()

            # Compute normalized NLL by conditioning on the mass over choice-token ids
            choice_ids = list(choice_token_ids.values())
            if choice_ids:
                device = log_probs.device
                # Build a tensor of log_probs for the choice ids, guarding out-of-range ids
                choice_log_probs = []
                for cid in choice_ids:
                    if 0 <= cid < log_probs.shape[-1]:
                        choice_log_probs.append(log_probs[cid])
                    else:
                        choice_log_probs.append(torch.tensor(float('-inf'), device=device))
                choice_log_probs = torch.stack(choice_log_probs)
                logsum = torch.logsumexp(choice_log_probs, dim=0).item()
                normalized_nll = -(log_probs[actual_token_id].item() - logsum)
            else:
                normalized_nll = raw_nll

            # Save raw and normalized probabilities for each choice label
            choice_raw_probs = {}
            choice_probs = {}
            if choice_token_ids:
                # compute raw probs and normalized probs (conditioned on choice mass)
                for lbl, cid in choice_token_ids.items():
                    if 0 <= cid < log_probs.shape[-1]:
                        raw_p = float(torch.exp(log_probs[cid]).item())
                    else:
                        raw_p = 0.0
                    choice_raw_probs[lbl] = raw_p
                # normalize over the choice tokens mass
                total_raw = sum(choice_raw_probs.values())
                if total_raw > 0:
                    for lbl in choice_raw_probs:
                        choice_probs[lbl] = choice_raw_probs[lbl] / total_raw
                else:
                    for lbl in choice_raw_probs:
                        choice_probs[lbl] = 0.0
            else:
                choice_raw_probs = {}
                choice_probs = {}

            # Get Top-2 for analysis
            top2_probs, top2_indices = torch.topk(log_probs, 2)
            top2_tokens = tokenizer.convert_ids_to_tokens(top2_indices)
            top2_probs = top2_probs.exp().tolist()

            # Extract trial-specific metadata
            row = participant_df.iloc[trial_idx]
            all_results.append({
                "participant_id": participant_df['participant_id'].iloc[0],
                "game": row["game"],
                "ground_truth": match.group(1),
                "raw_nll": raw_nll,
                "nll": normalized_nll,
                "choice_raw_probs": choice_raw_probs,
                "choice_probs": choice_probs,
                'top2': list(zip(top2_tokens, top2_probs)),
            })
            

    # Compute summary statistics
    valid_trial_nlls = [r['nll'] for r in all_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_trial_nlls) / len(valid_trial_nlls) if valid_trial_nlls else float('inf')
    print(f"🎯 Overall NLL: {overall_nll:.4f}")
    # raw nll summary
    valid_raw_nlls = [r['raw_nll'] for r in all_results if r['raw_nll'] != float('inf')]
    overall_raw_nll = sum(valid_raw_nlls) / len(valid_raw_nlls) if valid_raw_nlls else float('inf')
    print(f"🎯 Overall NLL (raw): {overall_raw_nll}")
    # compute how many trials had a top-2 token that was not a choice label
    non_choice_top2_count = sum(
        1 for r in all_results
        if not all(
            token in choice_token_ids for token, _ in r.get('top2', [])
        )
    )
    print(f"Number of trials with non-choice-label top-2 tokens: {non_choice_top2_count}")
    
    print(f"\n✅ Simulation complete")

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
        })

    # Create a DataFrame for all models
    all_model_df = pd.DataFrame(all_model_results)
    all_model_path = f'{DATA_FOLDER_OUT}/all_models_summary.csv'
    all_model_df.to_csv(all_model_path, index=False)
    print(f"Summary of all models saved to {all_model_path}")

if __name__ == "__main__":
    main()
