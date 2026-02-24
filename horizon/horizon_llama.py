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



from horizon_prompt import build_open_prompt

def simulate_participant(timeline_df, pipe, model, tokenizer, device):
    history = []
    # KV-cache and last prompt tracking for incremental encoding
    past_key_values = None
    last_prompt_text = ""
    
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
            # Build an open prompt that stops at the assistant header
            full_prompt = build_open_prompt(game_id, i, current_history_df)

            # Safety check: ensure prompt token length is within model limits
            try:
                prompt_tokens = tokenizer(full_prompt, return_tensors='pt')
                prompt_len = prompt_tokens['input_ids'].size(1)
            except Exception:
                # fallback to a simple length check
                prompt_len = len(tokenizer(full_prompt)['input_ids'])
            max_pos = getattr(tokenizer, 'model_max_length', None) or getattr(model.config, 'max_position_embeddings', None)
            if max_pos is not None and prompt_len + 2 > max_pos:
                print(f"⚠️ Prompt too long ({prompt_len} tokens) for model max {max_pos}. Skipping generation for this trial.")
                model_choice = 'INVALID'
                # record and continue to next trial
                reward = None
                if model_choice == 'I':
                    reward = row.get('reward_I', None)
                elif model_choice == 'H':
                    reward = row.get('reward_H', None)
                else:
                    reward = None
                prompt_path = None
                history.append({
                    "game": game_id,
                    "trial": row['trial_num_block'],
                    "type": "free",
                    "choice": model_choice,
                    "reward": reward,
                    "prompt_path": prompt_path,
                    "horizon": horizon
                })
                last_prompt_text = build_open_prompt(game_id, i + 1, pd.DataFrame(history))
                print(f"Game {game_id} (H={horizon}) Trial {row['trial_num_block']}: {model_choice} -> {reward} pts (skipped due to length)")
                continue

            # Compute incremental text since last prompt (if possible)
            if last_prompt_text and full_prompt.startswith(last_prompt_text):
                incremental_text = full_prompt[len(last_prompt_text):]
            else:
                incremental_text = full_prompt

            # Tokenize incremental piece (cached)
            def tokenize_and_cache(prompt_text: str):
                h = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:10]
                tok_filename = f"tok_{h}.pt"
                tok_path = os.path.join(PROMPT_DIR, tok_filename)
                if os.path.exists(tok_path):
                    d = torch.load(tok_path, map_location='cpu')
                    return d['input_ids'], d.get('attention_mask', None)
                toks = tokenizer(prompt_text, return_tensors='pt')
                tosave = {'input_ids': toks['input_ids']}
                if 'attention_mask' in toks:
                    tosave['attention_mask'] = toks['attention_mask']
                torch.save(tosave, tok_path)
                return tosave['input_ids'], tosave.get('attention_mask', None)

            input_ids, attention_mask = tokenize_and_cache(incremental_text)
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # 1. Generate Choice using model.generate under no_grad() + autocast
            model_choice = None
            use_amp = torch.cuda.is_available()
            with torch.no_grad():
                if use_amp:
                    ctx = torch.cuda.amp.autocast()
                else:
                    from contextlib import nullcontext
                    ctx = nullcontext()
                with ctx:
                    gen_kwargs = {
                        'input_ids': input_ids,
                        'max_new_tokens': 1,
                        'do_sample': False,
                        'eos_token_id': tokenizer.eos_token_id,
                        'use_cache': True,
                        'return_dict_in_generate': True,
                    }
                    if attention_mask is not None:
                        gen_kwargs['attention_mask'] = attention_mask
                    if past_key_values is not None:
                        gen_kwargs['past_key_values'] = past_key_values

                    outputs = model.generate(**gen_kwargs)
                    # update past_key_values for next incremental call
                    if hasattr(outputs, 'past_key_values'):
                        past_key_values = outputs.past_key_values
                    # extract generated tokens (after the incremental input)
                    # outputs.sequences holds the full sequence(s)
                    generated_tokens = outputs.sequences[0, -1:]
                    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    if len(decoded) > 0:
                        model_choice = decoded[0].upper()
                    else:
                        model_choice = ''
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

            prompt_path = save_prompt_file(full_prompt, game_id, row['trial_num_block'])

            history.append({
                "game": game_id,
                "trial": row['trial_num_block'],
                "type": "free",
                "choice": model_choice,
                "reward": reward,
                "prompt_path": prompt_path,
                "horizon": horizon
            })

            # prepare last_prompt_text for the next iteration (reflect new history)
            last_prompt_text = build_generate_prompt(game_id, i + 1, pd.DataFrame(history))

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

    # Device and precision setup for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if torch.cuda.is_available():
        try:
            model.half()
        except Exception:
            # Some models/custom modules may not support half() safely
            pass
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
        history = simulate_participant(participant_data, pipe, model, tokenizer, device)
        # Save results
        history.to_csv(out_path, index=False)
        # Cleanup: delete model and clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
