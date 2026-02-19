import get_models
import re
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import os

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'centaur-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'

def build_rl_prompt(past_trials: list) -> str:
    """Builds the prompt for the current trial with past trial data."""
    recent_trials = past_trials
    instructions = [
        "In this task, you have to repeatedly choose between two slot machines labeled U and P.",
        "You can choose a slot machine by pressing its corresponding key.",
        "When you select one of the machines, you will win 1 or 0 points.",
        "Your goal is to choose the slot machines that will give you the most points.",
        "You will receive feedback about the outcome after making a choice.",
        "The environment may change unpredictably; past success does not guarantee future results.",
        "You will play 1 game in total, consisting of 100 trials.\n",
        "Game 1:"
    ]
    
    prompt = "\n".join(instructions)
    # join instruction with \n
    # Add history of past trials to the prompt
    for past_trial in recent_trials:
        prompt += f"You press <<{past_trial['choice']}>> and get {past_trial['reward']} points.\n"

    # Add the current choice prompt
    prompt += f"You press <<"
    return prompt

def predict_participant(df_participant, model, tokenizer):
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
            "reward": row['reward'],
            "cumulative_reward": df_participant.iloc[:trial+1]['reward'].sum()
        })
    
    prompt = build_rl_prompt(past_trials)

    # 2. Tokenize ONCE and keep the BatchEncoding object
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)

    with torch.no_grad():
        # 3. Single Forward Pass
        outputs = model(input_ids)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

        per_trial_results = []
        
        # 4. Find choice positions using Regex
        choice_pattern = r'You press <<([^>]+)>>'
        matches = list(re.finditer(choice_pattern, prompt))

        for choice_idx, match in enumerate(matches):
            choice_char = match.group(1)
            char_start = match.start(1)

            # Fast mapping from character index to token index
            token_idx = encoding.char_to_token(0, char_start)

            if token_idx is None:
                continue

            # LOGIC: Logits at [token_idx - 1] predict the token at [token_idx]
            target_logits = logits[token_idx - 1]
            
            # Probability calculations
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)
            probs = torch.exp(log_probs)
            
            actual_token_id = input_ids[0, token_idx]
            nll = -log_probs[actual_token_id].item()
            #6. Get Top-2 for analysis
            top2_probs, top2_indices = torch.topk(log_probs, 2)
            top2_tokens = tokenizer.convert_ids_to_tokens(top2_indices)
            top2_probs = top2_probs.exp().tolist()

            per_trial_results.append({
                'trial_index': choice_idx,
                'ground_truth': choice_char,
                'nll': nll,
                'top2': list(zip(top2_tokens, top2_probs))
            })

    # Summary
    valid_trial_nlls = [r['nll'] for r in per_trial_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_trial_nlls) / len(valid_trial_nlls) if valid_trial_nlls else float('inf')

    print(f"✅ Simulation complete")
    print(f"🎯 Overall NLL: {overall_nll:.4f}")

    return per_trial_results, overall_nll,prompt


def main():

    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline = pd.read_csv(DATA_IN_TEST)
    timeline['choice'] = timeline['choice'].map({0: 'U', 1: 'P'})
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
