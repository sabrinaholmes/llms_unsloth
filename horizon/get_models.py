import os
import glob
from unsloth import FastLanguageModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import random, torch

from huggingface_hub import whoami, HfApi,HfFolder,login

access_token = HfFolder.get_token()
login(token=access_token)

MODEL_PATHS = {
    'centaur-70B': 'marcelbinz/Llama-3.1-Centaur-70B',
    'centaur-70B-adapter': 'marcelbinz/Llama-3.1-Centaur-70B-adapter',
    'centaur-8B': 'marcelbinz/Llama-3.1-Centaur-8B',
    'llama-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-70B-adapter': 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit',
    'llama-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-3-8B': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama-3-70B': 'meta-llama/Meta-Llama-3-70B-Instruct' # Added for completeness
}


def get_model_no_pipe(name):
    if name not in MODEL_PATHS:
        raise ValueError(f"Model name '{name}' not recognized. Available models: {list(MODEL_PATHS.keys())}")
    path = MODEL_PATHS[name]

    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=DTYPE
    )

    model.eval()
    model._past = None  # Reset past key values if any
    tokenizer = AutoTokenizer.from_pretrained(path)

    return model, tokenizer

def get_model_no_pipe_unsloth(name):
    if name not in MODEL_PATHS:
        raise ValueError(f"Model name '{name}' not recognized. Available models: {list(MODEL_PATHS.keys())}")
    path = MODEL_PATHS[name]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=32768,
        dtype=None,
        load_in_4bit=True,  # Same as original
    )
    # Ensure deterministic inference behavior (disable dropout etc.)
    try:
        model.eval()
    except Exception:
        pass
    return model, tokenizer

def create_text_generation_pipeline(model, tokenizer, temperature=1.0, max_new_tokens=1):
    """
    Creates a text-generation pipeline with the given model and tokenizer.

    Args:
        model: The preloaded model for text generation.
        tokenizer: The corresponding tokenizer.
        temperature (float): Sampling temperature for generation (default: 1.0).
        max_new_tokens (int): Maximum number of tokens to generate (default: 1024).

    Returns:
        A transformers pipeline object for text generation.
    """
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        pad_token_id=0,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def generate(prompt: str, pipe) -> str:
    """Generates a response from the model using the provided prompt.

    Args:
        prompt (str): The input prompt for the model.
        pipe: The text generation pipeline.

    Returns:
        str: The generated text response from the model.
    """
    return pipe(prompt)[0]['generated_text'][len(prompt):]
