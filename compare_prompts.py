import os
import sys
import pandas as pd
import re
import datasets

def normalize(s):
    return s.strip().replace('\r\n', '\n').replace('\r', '\n')


def load_hf_prompts(experiment_name='wu2018generalisation'):
    df_hf = pd.read_json("hf://datasets/marcelbinz/Psych-101-test/prompts_testing_t1.jsonl", lines=True)
    # Replace the ö characters with regex wildcards
    safe_name = re.sub(r'[^\x00-\x7F]+', lambda m: '.' * len(m.group()), experiment_name)
    df_exp = df_hf[df_hf['experiment'].str.contains(safe_name, na=False, regex=True)]
    return set(normalize(p) for p in df_exp['text'].dropna())


def is_prompt_in_test_set(prompt: str, experiment_name: str = 'wu2018generalisation') -> bool:
    hf_prompts = load_hf_prompts(experiment_name)
    print(f"checking experiment: {experiment_name}, {len(hf_prompts)} prompts in test set")
    local = normalize(prompt)
    if local in hf_prompts:
        return True

    print("\n--- Prompt NOT found in test set. Diagnosing closest match ---")
    hf_list = list(hf_prompts)

    # Find best candidate: prefer one sharing the same last 100 chars, else longest common suffix
    local_tail = local[-100:]
    candidate = next((p for p in hf_list if p[-100:] == local_tail), None)
    if candidate is None:
        # Fall back: pick the HF prompt with the longest common prefix
        candidate = max(hf_list, key=lambda p: len(os.path.commonprefix([local, p])))

    print(f"Local length: {len(local)}, Best HF candidate length: {len(candidate)}")

    # Find first differing character
    for i, (a, b) in enumerate(zip(local, candidate)):
        if a != b:
            print(f"First diff at index {i}:")
            print(f"  Local: {repr(local[max(0, i-50):i+50])}")
            print(f"  HF:    {repr(candidate[max(0, i-50):i+50])}")
            break
    else:
        # No char diff — lengths differ
        if len(local) == len(candidate):
            print("Prompts are identical after normalization (unexpected).")
        else:
            shorter, longer = (local, candidate) if len(local) < len(candidate) else (candidate, local)
            label_s = "Local" if len(local) < len(candidate) else "HF"
            label_l = "HF" if len(local) < len(candidate) else "Local"
            print(f"{label_s} is shorter. Extra content in {label_l}:")
            print(repr(longer[len(shorter):len(shorter)+200]))

    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_prompts.py <prompt>")
        sys.exit(1)

    prompt = sys.argv[1]
    result = is_prompt_in_test_set(prompt)
    print(result)