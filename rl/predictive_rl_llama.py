import get_models
import re
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
import os

DATA_IN_TEST = 'data/in/test_data.csv'

MODEL = 'llama-70B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/predictive/{MODEL}/singles'
def build_prediction_prompt(past_trials: list,total_trials) -> str:
    """
    Formats the entire game history as a continuous dialogue.
    Each trial is an assistant turn, allowing for NLL extraction
    at every 'choice' token in a single forward pass.
    """
    # 1. System Prompt (The Rules)
    system_msg = [
        "In this task, you have to repeatedly choose between two slot machines labeled U and P."
        "When you select one of the machines, you will win 1 or 0 points."
        "Your goal is to choose the slot machines that will give you the most points."
        "You will receive feedback about the outcome after making a choice."
        "The environment may change unpredictably, and past success does not guarantee future results. You’ll need to adapt to these changes to keep finding the better machine."
        "You will play 1 game in total, consisting of 100 trials."
        "Respond with ONLY the character 'U' or 'P'."
    ]
    system_text = "".join(system_msg)
    transcript = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_text}<|eot_id|>"
    # 2. THE FIX: Add the initial User prompt to start the game
    transcript += f"<|start_header_id|>user<|end_header_id|>\nGame 1."
    # 3. Iterative Assistant/User turns
    for i, trial in enumerate(past_trials):
        # The choice is the Assistant's action
        choice = trial['choice']  # 'U' or 'P'
        reward = trial['reward']  # 1 or 0
        #print(choice)

        # We wrap the choice in the assistant header so the model "owns" it
        transcript += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{choice}<|eot_id|>"

        # We provide the reward in a User block (as feedback from the environment)
        # We only add this if it's not the very last trial (unless you want to evaluate the next)
        if i < len(past_trials) - 1:
            transcript += (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"-> {reward} points."
            )

    return transcript

def predict_participant(df_participant: pd.DataFrame, model, tokenizer):
    """Simulates a participant with log-likelihood tracking, NLL calculations, and top-2 token probabilities"""
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

    prompt = build_prediction_prompt(past_trials,total_trials)
    #print(f"Prompt:\n{prompt}")

    # Tokenize the full participant prompt once
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768)
    inputs = inputs.to(model.device) # Correct way to move to device while keeping BatchEncoding type

    per_trial_results = []

    with torch.no_grad():
        # 3. Single forward pass
        outputs = model(**inputs)
        # Shift logits: logits[i] predicts input_ids[i+1]
        logits = outputs.logits[0]

        # 4. Find choice positions using Regex
        # We look for the Choice (U or P) immediately followed by the <|eot_id|>
        # This matches the specific Assistant turns in our transcript
        # Regex to find 'I' or 'H' only when they follow the assistant header
        choice_pattern = r"(?<=<\|start_header_id\|>assistant<\|end_header_id\|>\n)([UP])(?=<\|eot_id\|>)"
        matches = list(re.finditer(choice_pattern, prompt))

        for choice_idx, match in enumerate(matches):
            choice_char = match.group(1)
            char_start = match.start(1)

            # 5. Map character index to token index
            token_idx = inputs.char_to_token(char_start)

            if token_idx is None or token_idx == 0:
                continue

            # The logit that predicts input_ids[token_idx] is at [token_idx - 1]
            target_logits = logits[token_idx - 1]
            log_probs = torch.nn.functional.log_softmax(target_logits, dim=-1)

            # Get NLL for the actual choice
            target_id = inputs['input_ids'][0, token_idx] # Corrected access to input_ids
            nll = -log_probs[target_id].item()

            # 6. Get Top-2 for analysis
            top2_probs, top2_indices = torch.topk(log_probs, 2)
            top2_tokens = tokenizer.convert_ids_to_tokens(top2_indices)
            top2_probs = top2_probs.exp().tolist()

            per_trial_results.append({
                'trial_index': choice_idx + 1,
                'ground_truth': choice_char,
                'nll': nll,
                'top2': list(zip(top2_tokens, top2_probs))
            })


    # Compute summary statistics
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


    letter_token_ids = {
    "U": tokenizer("U", add_special_tokens=False)['input_ids'][0],
    "P": tokenizer("P", add_special_tokens=False)['input_ids'][0],
}


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
