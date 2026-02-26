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
