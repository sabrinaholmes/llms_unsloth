import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_models
import re
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from multi_cue_prompt import build_predict_centaur_prompt, build_predict_llama_prompt
from compare_prompts import is_prompt_in_test_set


DATA_IN_TEST = 'data/in/test/full_test_data_multi_cue.csv'
MODEL = 'llama-70B-adapter'
DATA_FOLDER_OUT = f'data/out/predictive_steps/{MODEL}/singles'
LLM_TYPE = 'llama' if 'llama' in MODEL else 'centaur'
RUN_TEST_SET_ONLY = True if LLM_TYPE == 'centaur' else False
EXPERIMENT_NAME = 'collsiöö2023MCPL'
RUN_VERBAL_ONLY = False
NORMALISE = False  # divide multi-token NLL by answer length for comparability

ENGLISH_OPTIONS = [
    'extremely low', 'very low', 'low', 'somewhat low', 'normal',
    'somewhat high', 'high', 'very high', 'extremely high'
]

NUMERIC_OPTIONS = ['10', '20', '30', '40', '50', '60', '70', '80', '90']


def build_option_token_seqs(tokenizer, options, context=''):
    """
    Build {option_str: [token_id, ...]} for every option, tokenized exactly
    as they appear in the prompt.

    context: text immediately preceding the answer in the prompt,
             e.g. '<<' for Centaur or '\n' for Llama. Using the actual
             context ensures token IDs match what char_to_token returns,
             avoiding a leading-space mismatch that causes inf NLL.
    """
    option_token_seqs = {}
    for option in options:
        enc = tokenizer(context + option, add_special_tokens=False)
        start = enc.char_to_token(len(context))
        option_token_seqs[option] = enc['input_ids'][start:]
    return option_token_seqs


def build_trie(option_token_seqs):
    """
    Build prefix trie: tuple(prefix_token_ids) -> set(valid_next_token_ids).

    Encodes which tokens are valid at each position given what came before.
    Used to mask the vocabulary at each step to only allowed continuations.

      ()              -> {ext, very, som, normal, high, low}   # step 1
      (ext,)          -> {remely}                               # step 2
      (som,)          -> {ewhat}                                # step 2
      (very,)         -> {high, low}                            # step 2
      (ext, remely)   -> {high, low}                            # step 3
      (som, ewhat)    -> {high, low}                            # step 3
    """
    trie = {}
    for token_seq in option_token_seqs.values():
        for i in range(len(token_seq)):
            prefix = tuple(token_seq[:i])
            trie.setdefault(prefix, set()).add(token_seq[i])
    return trie


def compute_english_nll(all_logits, token_idx, ground_truth_option,
                        option_token_seqs, trie, tokenizer):
    """
    Compute masked NLL for a multi-token English ground truth option.

    Single forward pass is sufficient here because we only evaluate the
    ground truth sequence — logits at each position are already conditioned
    on the actual preceding ground truth tokens in the prompt, which is
    exactly the correct conditioning for NLL evaluation.

    At each token position k:
      - Take logits[token_idx - 1 + k]  (predicts token k of the answer)
      - Mask to only the tokens valid at trie depth k given actual prefix
      - Softmax over masked logits → restricted probability distribution
      - NLL += -log(prob of actual ground truth token k)

    Also records the masked distribution at each step for inspection.

    Args:
        all_logits:           tensor [seq_len, vocab_size]
        token_idx:            index of first answer token in the prompt
        ground_truth_option:  e.g. 'extremely_high'
        option_token_seqs:    {option_str: [token_id, ...]}
        trie:                 {tuple(prefix): set(valid_next_ids)}
        tokenizer:            for decoding tokens in step_distributions

    Returns:
        nll:               float
        step_distributions: list of dicts, one per token position:
                              'step'   : int (1-indexed)
                              'prefix' : list of token strings seen so far
                              'vocab'  : {token_str: prob} masked distribution
                              'top5'   : top-5 (token_str, prob)
    """
    if ground_truth_option not in option_token_seqs:
        print(f"  ⚠️  '{ground_truth_option}' not in option set")
        return float('inf'), []

    token_seq = option_token_seqs[ground_truth_option]
    nll = 0.0
    step_distributions = []

    for k, actual_tok in enumerate(token_seq):
        prefix = tuple(token_seq[:k])   # actual ground truth tokens so far
        logits_k = all_logits[token_idx - 1 + k].clone()

        # Mask to valid tokens at this trie depth
        valid_ids = list(trie.get(prefix, set()))
        mask = torch.full_like(logits_k, float('-inf'))
        for tid in valid_ids:
            mask[tid] = 0.0
        log_probs_k = F.log_softmax(logits_k + mask, dim=-1)
        probs_k     = log_probs_k.exp()

        step_nll = -log_probs_k[actual_tok].item()
        nll += step_nll

        sorted_vocab = sorted(
            [(tokenizer.convert_ids_to_tokens([tid])[0], probs_k[tid].item())
             for tid in valid_ids],
            key=lambda x: -x[1]
        )

        step_distributions.append({
            'step':        k + 1,
            'prefix':      tokenizer.convert_ids_to_tokens(list(prefix)),
            'prob':        probs_k[actual_tok].item(),
            'sorted_vocab': sorted_vocab,
        })

    if NORMALISE and len(token_seq) > 0:
        nll = nll / len(token_seq)

    return nll, step_distributions


def predict_participant(participant_df, model, tokenizer, exp_mixed=False, limited_train=False):
    """
    Single forward pass participant simulation.

    English conditions (1/3):
      - Trie restricts vocabulary at each token position to valid continuations
      - NLL = sum of -log(masked_prob) at each ground truth token position
      - Correctly conditioned because ground truth tokens sit in the prompt

    Numeric conditions (2/4):
      - Mask logits to {10,20,...,90}, softmax, NLL of ground truth token
    """
    all_results = []

    if LLM_TYPE == 'centaur':
        trigger_pattern = r'<<([^>]+)>>'
    else:
        trigger_pattern = r'<\|start_header_id\|>assistant<\|end_header_id\|>\n(.*?)<\|eot_id\|>'

    participant_id = participant_df['Fp'].iloc[0]
    full_prompt = (
        build_predict_centaur_prompt(participant_df, exp_mixed, limited_train)
        if LLM_TYPE == 'centaur'
        else build_predict_llama_prompt(participant_df)
    )

    encoding  = tokenizer(full_prompt, return_tensors="pt", truncation=True)
    input_ids = encoding['input_ids'].to(model.device)

    # Pre-compute once per participant — tokenize options in their actual prompt context
    # so token IDs match what the model sees (no leading-space mismatch)
    context = '<<' if LLM_TYPE == 'centaur' else '\n'
    option_token_seqs = build_option_token_seqs(tokenizer, ENGLISH_OPTIONS, context=context)
    trie = build_trie(option_token_seqs)

    print("Option token sequences:")
    for opt, seq in option_token_seqs.items():
        print(f"  {opt}: {tokenizer.convert_ids_to_tokens(seq)}")

    numeric_token_ids = [
        tid for tid in tokenizer.convert_tokens_to_ids(NUMERIC_OPTIONS)
        if tid != tokenizer.unk_token_id
    ]

    in_test = is_prompt_in_test_set(full_prompt, experiment_name=EXPERIMENT_NAME)
    print(f"✅ Participant {participant_id} — in test set: {in_test}")

    with torch.no_grad():
        all_logits = model(input_ids).logits[0]   # [seq_len, vocab_size]

    for trial_idx, match in enumerate(re.finditer(trigger_pattern, full_prompt)):
        choice_char = match.group(1)
        char_pos    = match.start(1)

        trial_condition = participant_df.iloc[trial_idx]['condition']
        is_english      = trial_condition in [1, 3]

        token_idx = encoding.char_to_token(0, char_pos)
        if token_idx is None:
            continue

        if is_english:
            nll, step_distributions = compute_english_nll(
                all_logits, token_idx, choice_char,
                option_token_seqs, trie, tokenizer
            )

            print(f"\n  Trial {trial_idx + 1} | ground truth: {choice_char} | NLL: {nll:.4f}")
            for sd in step_distributions:
                prefix_label = 'start' if not sd['prefix'] else str(sd['prefix'])
                print(f"    Step {sd['step']} (after {prefix_label}): {sd['sorted_vocab']}")

            all_results.append({
                "participant_id":    participant_id,
                "trial_index":       trial_idx + 1,
                "ground_truth":      choice_char,
                "nll":               nll,
                "step_distributions": step_distributions,
            })

        else:
            # Numeric — single token, mask + softmax
            logits = all_logits[token_idx - 1].clone()
            mask   = torch.full_like(logits, float('-inf'))
            for tid in numeric_token_ids:
                mask[tid] = 0.0
            log_probs = F.log_softmax(logits + mask, dim=-1)

            actual_token_id = input_ids[0, token_idx]
            nll = -log_probs[actual_token_id].item()

            sorted_vocab = sorted(
                [(opt, log_probs[tid].exp().item())
                 for opt, tid in zip(NUMERIC_OPTIONS, numeric_token_ids)],
                key=lambda x: -x[1]
            )
            all_results.append({
                "participant_id":    participant_id,
                "trial_index":       trial_idx + 1,
                "ground_truth":      choice_char,
                "nll":               nll,
                "step_distributions": [{
                    'step':        1,
                    'prefix':      [],
                    'prob':        log_probs[actual_token_id].exp().item(),
                    'sorted_vocab': sorted_vocab,
                }],
            })

    valid_nlls  = [r['nll'] for r in all_results if r['nll'] != float('inf')]
    overall_nll = sum(valid_nlls) / len(valid_nlls) if valid_nlls else float('inf')
    print(f"\n✅ Simulation complete — Overall NLL: {overall_nll:.4f}")

    return all_results, overall_nll, full_prompt


def main():
    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)

    model, tokenizer = get_models.get_model_no_pipe_unsloth(MODEL)
    timeline      = pd.read_csv(DATA_IN_TEST)
    exp_mixed     = False
    limited_train = False
    all_model_results = []

    for exp in timeline['exp'].unique():
        print(f"\n🔍 Processing experiment: {exp}")
        exp_df = timeline[timeline['exp'] == exp]
        print(f"Found {len(exp_df['participant'].unique())} participants")

        for p in exp_df['participant'].unique():
            if exp == 2: exp_mixed     = True
            if exp == 3: limited_train = True; exp_mixed = False

            print(f"\n🧠 Simulating participant {p}")
            out_path       = f'{DATA_FOLDER_OUT}/participant_{p}_exp{exp}.csv'
            out_path_steps = f'{DATA_FOLDER_OUT}/participant_{p}_exp{exp}_steps.csv'

            if os.path.exists(out_path):
                print("Already done. Skipping...")
                continue

            model_data = exp_df[exp_df['participant'] == p]
            fp = exp_df['Fp'].iloc[0]

            if RUN_TEST_SET_ONLY:
                prompt_check = build_predict_centaur_prompt(model_data, exp_mixed, limited_train)
                if not is_prompt_in_test_set(prompt_check, experiment_name=EXPERIMENT_NAME):
                    print("Not in test set. Skipping...")
                    continue

            results, overall_nll, prompt = predict_participant(
                model_data, model, tokenizer, exp_mixed, limited_train
            )

            # Main results CSV — one row per trial, prob_step_N columns for each answer token
            rows = []
            for r in results:
                row = {k: v for k, v in r.items() if k != 'step_distributions'}
                for i, sd in enumerate(r.get('step_distributions') or []):
                    row[f'prob_step_{i + 1}'] = sd.get('prob')
                rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)

            # Step distributions CSV — one row per (trial, step) with full sorted distribution
            step_rows = []
            for r in results:
                for sd in (r.get('step_distributions') or []):
                    prefix_str = 'start' if not sd['prefix'] else str(sd['prefix'])
                    step_rows.append({
                        'participant_id': r['participant_id'],
                        'trial_index':    r['trial_index'],
                        'ground_truth':   r['ground_truth'],
                        'step':           sd['step'],
                        'prefix':         prefix_str,
                        'gt_prob':        sd.get('prob'),
                        'distribution':   str(sd['sorted_vocab']),
                    })
            if step_rows:
                pd.DataFrame(step_rows).to_csv(out_path_steps, index=False)
                print(f"Step distributions saved to {out_path_steps}")

            print(f"Results saved to {out_path}")
            all_model_results.append({
                'model_id': fp, 'overall_nll': overall_nll, 'prompt': prompt
            })

    pd.DataFrame(all_model_results).to_csv(
        f'{DATA_FOLDER_OUT}/all_models_summary.csv', index=False
    )
    print("Summary saved.")


if __name__ == "__main__":
    main()