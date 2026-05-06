import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_models
import pandas as pd
import torch
import gc
import json
from multi_cue_prompt import build_generate_centaur_prompt, build_generate_llama_prompt
import hashlib
import gzip

DATA_IN = 'data/in/timeline_lowercase.csv'
MODEL = 'centaur-8B-adapter'  # Change this to the desired model name
LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
RUN_TEST_MODE = True if '8B' in MODEL else False  # Set to True to only run on small 8B models for faster testing
ENGLISH_OPTIONS_LOW = [
    'extremely low', 'very low', 'low', 'somewhat low', 'normal',
    'somewhat high', 'high', 'very high', 'extremely high'
]
ENGLISH_OPTIONS_ABB= ['EL', 'VL', 'L', 'SL', 'N', 'SH', 'H', 'VH', 'EH']
GENERATE_ONE_TOKEN = False  # If True, we will only take the first token of the model's response as the choice. This is important for models that might generate more than one token (e.g., "EL (extremely low)")
if DATA_IN.endswith('lowercase.csv'):
    ENGLISH_OPTIONS = ENGLISH_OPTIONS_LOW
    print("Using lowercase English options for tokenization")
    DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
else:
    ENGLISH_OPTIONS = ENGLISH_OPTIONS_ABB
    print("Using abbreviated English options for tokenization")
    GENERATE_ONE_TOKEN = True  # For abbreviated options, we should only take the first token to avoid issues with extra tokens being generated
    DATA_FOLDER_OUT = f'data/out/generative_abbreviated/{MODEL}/singles'


PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
NUMERIC_OPTIONS = ['10', '20', '30', '40', '50', '60', '70', '80', '90']

def save_prompt(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    path = os.path.join(PROMPT_DIR, f"{h}.txt.gz")
    if not os.path.exists(path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(prompt)
    return h

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


def simulate_participant(timeline_df, wrapper, participant_id,test_mode=False):
    """
    Open-loop simulation: model generates Caldionine estimates trial-by-trial
    using the participant's timeline and all past model responses as context.

    Returns:
        results: list of dicts per trial with trial index, cues, model choice,
                 correct criterium, and the prompt used.
    """
    choice_options = choice_options_for_participant(timeline_df)
    completed_rows = []
    results = []

    for trial_num, (_, row) in enumerate(timeline_df.iterrows()):
        #  if test_mode=True, only run the first 50 trial to quickly check the code works and inspect the prompt and model response
            if test_mode and trial_num >= 50:
                break
            # Build incremental timeline: past completed rows + current row (no response yet)
            if completed_rows:
                past_df = pd.DataFrame(completed_rows)
                trial_timeline = pd.concat([past_df, row.to_frame().T], ignore_index=True)
            else:
                trial_timeline = row.to_frame().T.reset_index(drop=True)

            if LLM_TYPE == 'centaur':
                prompt = build_generate_centaur_prompt(trial_timeline)
            else:
                prompt = build_generate_llama_prompt(trial_timeline)
            if GENERATE_ONE_TOKEN:
                choice = wrapper.generate(prompt, choice_options=choice_options, max_new_tokens=1,processor_type="original")
                print("generating with max_new_tokens=1 due to GENERATE_ONE_TOKEN=True")
                choice = choice.strip()
                step_dists = None
            else:
                #choice, step_dists = wrapper.generate(prompt, choice_options=choice_options)
                choice = wrapper.generate(prompt, choice_options=choice_options, processor_type="prefix_tree")
                #print("generating with scores (multi-token) due to GENERATE_ONE_TOKEN=False")
                choice = choice.strip().lower()  # Normalize choice for comparison

            if choice_options is not None and choice not in choice_options:
                print(f"⚠️ Trial {trial_num + 1}: invalid choice '{choice}'")

            # Store model response so future trials can use it as context
            completed_row = row.copy()
            completed_row['response_raw'] = choice if choice is not None else ''
            completed_rows.append(completed_row)

            print(f'Trial {trial_num + 1}: choice={choice}, criterium={row["Criterium_English"]}')
            results.append({
                'participant_id': participant_id,
                'trial': trial_num + 1,
                'block': row['Block'],
                'condition': row['condition'],
                'cue1': row['Cue1_English'],
                'cue2': row['Cue2_English'],
                'criterium': row['Criterium_English'],
                'choice': choice,
                'prompt_hash': save_prompt(prompt),
                #'step_distributions': json.dumps(step_dists) if step_dists is not None else None,
            })

    return results

def main():
    os.makedirs(DATA_FOLDER_OUT, exist_ok=True)
    os.makedirs(PROMPT_DIR, exist_ok=True)

    timeline = pd.read_csv(DATA_IN)
    wrapper = get_models.ModelWrapper(MODEL, use_unsloth=True, temperature=1.0)
    torch.cuda.empty_cache()

    participants = timeline['Fp'].unique()
    if RUN_TEST_MODE:
        # run first 1 participants with condition 1,3 (English options) and first 1 participants with condition 2,4 (Numeric options)
        participants = timeline[timeline['condition'].isin([1, 3])]['Fp'].unique()[:1]
        participants = list(participants) + list(timeline[timeline['condition'].isin([2, 4])]['Fp'].unique()[:1])
        print(f"Running test mode only: participants {participants}")
    all_model_results = []

    for participant_id in participants:
        out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id}.csv'
        if os.path.exists(out_path):
            print(f"File {out_path} already exists. Skipping participant {participant_id}.")
            all_model_results.append({'participant_id': participant_id})
            continue

        participant_timeline = timeline[timeline['Fp'] == participant_id].reset_index(drop=True)
        print(f'\nSimulating participant {participant_id} ({len(participant_timeline)} trials)')
        gc.collect()
        torch.cuda.empty_cache()

        results = simulate_participant(participant_timeline, wrapper, participant_id=participant_id, test_mode=RUN_TEST_MODE)
        result_df = pd.DataFrame(results)
        result_df.to_csv(out_path, index=False)
        print(f'Results saved to {out_path}')
        all_model_results.append({'participant_id': participant_id})
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = f'{DATA_FOLDER_OUT}/all_participants_summary.csv'
    pd.DataFrame(all_model_results).to_csv(summary_path, index=False)
    print(f'Summary saved to {summary_path}')

if __name__ == "__main__":
    main()
