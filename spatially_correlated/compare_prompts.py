import sys
import pandas as pd


def normalize(s):
    return s.strip().replace('\r\n', '\n').replace('\r', '\n')


def load_hf_prompts():
    df_hf = pd.read_json("hf://datasets/marcelbinz/Psych-101-test/prompts_testing_t1.jsonl", lines=True)
    df_wu = df_hf[df_hf['experiment'].str.contains('wu2018generalisation', na=False)]
    return set(normalize(p) for p in df_wu['text'].dropna())


def is_prompt_in_test_set(prompt: str) -> bool:
    hf_prompts = load_hf_prompts()
    return normalize(prompt) in hf_prompts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_prompts.py <prompt>")
        sys.exit(1)

    prompt = sys.argv[1]
    result = is_prompt_in_test_set(prompt)
    print(result)
