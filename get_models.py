import os
import glob
from unsloth import FastLanguageModel
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm
import random, torch

from huggingface_hub import whoami, HfApi,HfFolder,login

class RawLogitsCaptureProcessor(LogitsProcessor):
    """Captures raw logits before any other processors modify them."""
    def __init__(self):
        self.raw_scores: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.raw_scores.append(scores[0].clone())  # save (vocab_size,) per step
        return scores


class AllowlistLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: list[int], device):
        self.mask = torch.full((len(tokenizer),), float("-inf"), device=device)
        self.mask[allowed_token_ids] = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        return scores + self.mask


class PrefixTreeLogitsProcessor(LogitsProcessor):
    """
    Constrains generation to a finite set of multi-token options via a prefix tree.
    At each step, only tokens that continue a valid option prefix are allowed.
    After a complete option is generated, only EOS is allowed (forcing generation to stop).
    """
    def __init__(self, allowed_options: list[str], tokenizer, device):
        vocab_size = len(tokenizer)
        eos_id = tokenizer.eos_token_id

        # Build prefix tree: prefix_tuple -> set of valid next token IDs
        # Each option is added twice: with and without leading space, so both " low" and "low" (different token IDs) are valid paths to the same option.
        prefix_tree: dict[tuple, set] = {}
        for option in allowed_options:
            # Add both spaced (" low") and unspaced ("low") encodings so the model
            # can take either token path to the same option.
            for encoding in (' ' + option, option):
                token_ids = tokenizer.encode(encoding, add_special_tokens=False)
                for i in range(len(token_ids)):
                    prefix = tuple(token_ids[:i])
                    prefix_tree.setdefault(prefix, set()).add(token_ids[i])
                # After full option: only EOS allowed -> generation stops
                prefix_tree.setdefault(tuple(token_ids), set()).add(eos_id)

        # Pre-compute mask tensors (avoids per-call allocation)
        self.masks: dict[tuple, torch.Tensor] = {}
        for prefix, allowed_ids in prefix_tree.items():
            mask = torch.full((vocab_size,), float('-inf'), device=device)
            for tid in allowed_ids:
                mask[tid] = 0.0
            self.masks[prefix] = mask

        self.prompt_length: int | None = None

    def set_prompt_length(self, length: int):
        self.prompt_length = length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_length is None:
            generated = ()
        else:
            generated = tuple(input_ids[0, self.prompt_length:].tolist())
        mask = self.masks.get(generated)
        if mask is None:
            return scores  # Unknown prefix — allow everything (safety fallback)
        return scores + mask.unsqueeze(0)

access_token = HfFolder.get_token()
login(token=access_token)

MODEL_PATHS = {
    'centaur-70B': 'marcelbinz/Llama-3.1-Centaur-70B',
    'centaur-70B-adapter': 'marcelbinz/Llama-3.1-Centaur-70B-adapter',
    'centaur-8B': 'marcelbinz/Llama-3.1-Centaur-8B',
    'centaur-8B-adapter':'marcelbinz/Llama-3.1-Minitaur-8B-adapter',
    'llama-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
    'llama-70B-adapter': 'unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit',
    'llama-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama-8B-adapter': 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    'llama-8B-base-adapter': 'unsloth/Llama-3.1-8B-unsloth-bnb-4bit',
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


def generate(prompt: str, pipe: transformers.pipeline, tokenizer: AutoTokenizer, choice_options: list[str] = None, max_new_tokens: int = 1, processor_type: str = 'prefix_tree') -> str:
    """Generates a response from the model using the provided prompt.

    Args:
        processor_type: 'prefix_tree' (default) for multi-token options with stateful prefix
                        masking, or 'allowlist' for single-token allowlist masking.
    """
    logits_processor = None
    if choice_options:
        device = next(pipe.model.parameters()).device
        if processor_type == 'prefix_tree':
            processor = PrefixTreeLogitsProcessor(choice_options, tokenizer, device)
            prompt_length = pipe.tokenizer(prompt, return_tensors='pt')['input_ids'].shape[1]
            processor.set_prompt_length(prompt_length)
            max_new_tokens = max(
                max(
                    len(tokenizer.encode(' ' + opt, add_special_tokens=False)),
                    len(tokenizer.encode(opt, add_special_tokens=False)),
                )
                for opt in choice_options
            )
        else:  # 'allowlist'
            allowed_ids = pipe.tokenizer.convert_tokens_to_ids(choice_options)
            allowed_ids = [i for i in allowed_ids if i != pipe.tokenizer.unk_token_id]
            processor = AllowlistLogitsProcessor(allowed_ids, device=device)
        logits_processor = LogitsProcessorList([processor])
    return pipe(
        prompt,
        logits_processor=logits_processor,
        max_new_tokens=max_new_tokens,
    )[0]['generated_text'][len(prompt):]


class ModelWrapper:
    def __init__(self, name: str, use_unsloth: bool = False, temperature: float = 1.0):
        if use_unsloth:
            self.model, self.tokenizer = get_model_no_pipe_unsloth(name)
            FastLanguageModel.for_inference(self.model)
        else:
            self.model, self.tokenizer = get_model_no_pipe(name)
        self.model._past = None
        self.temperature = temperature
        self.pipe = create_text_generation_pipeline(self.model, self.tokenizer, temperature)

    def generate(self, prompt: str, choice_options: list[str] = None, max_new_tokens: int = 1, processor_type: str = 'prefix_tree') -> str:
        # AllowlistLogitsProcessor reads the global `tokenizer`, so set it before instantiating
        global tokenizer
        tokenizer = self.tokenizer
        return generate(prompt, self.pipe, self.tokenizer, choice_options, max_new_tokens, processor_type)

    def generate_with_scores(
        self,
        prompt: str,
        choice_options: list[str] = None,
        processor_type: str = 'prefix_tree',
    ) -> tuple[str, list[dict]]:
        """Like generate(), but also returns per-step probability distributions.

        Returns:
            (generated_text, step_distributions) where step_distributions is a list
            of dicts (one per generated token):
                {'token_id': int, 'token': str, 'probs': {token_str: float}}
            probs contains only the valid (non-masked) tokens at that step.
        """
        global tokenizer
        tokenizer = self.tokenizer

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors='pt').to(device)
        prompt_length = inputs['input_ids'].shape[1]

        logits_processor = None
        capture_processor = None
        max_new_tokens = 1
        if choice_options:
            capture_processor = RawLogitsCaptureProcessor()
            if processor_type == 'prefix_tree':
                processor = PrefixTreeLogitsProcessor(choice_options, self.tokenizer, device)
                processor.set_prompt_length(prompt_length)
                max_new_tokens = max(
                    max(
                        len(self.tokenizer.encode(' ' + opt, add_special_tokens=False)),
                        len(self.tokenizer.encode(opt, add_special_tokens=False)),
                    )
                    for opt in choice_options
                )
            else:
                allowed_ids = self.tokenizer.convert_tokens_to_ids(choice_options)
                allowed_ids = [i for i in allowed_ids if i != self.tokenizer.unk_token_id]
                processor = AllowlistLogitsProcessor(allowed_ids, device=device)
            logits_processor = LogitsProcessorList([capture_processor, processor])

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=True,
                temperature=self.temperature,
            )
        # output.scores: tuple of (batch=1, vocab_size) tensors, one per step, AFTER logits processors
        # output.sequences: (1, prompt_len + gen_len) token ids

        generated_ids = output.sequences[0, prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        step_distributions = []
        for step_idx, step_scores in enumerate(output.scores):
            logits = step_scores[0]  # shape: (vocab_size,)
            valid_mask = logits > float('-inf')
            valid_logits = logits[valid_mask]
            probs = torch.softmax(valid_logits, dim=-1).cpu().tolist()
            valid_ids = valid_mask.nonzero(as_tuple=True)[0].tolist()
            token_id = generated_ids[step_idx].item()

            raw_top_k = None
            if capture_processor is not None and step_idx < len(capture_processor.raw_scores):
                raw_logits = capture_processor.raw_scores[step_idx]
                raw_probs = torch.softmax(raw_logits, dim=-1)
                k = min(20, raw_probs.size(0))
                raw_top_probs, raw_top_ids = torch.topk(raw_probs, k)
                raw_top_k = {
                    self.tokenizer.convert_ids_to_tokens([tid])[0]: p
                    for tid, p in zip(raw_top_ids.tolist(), raw_top_probs.cpu().tolist())
                }

                token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
                id_to_prob = dict(zip(valid_ids, probs))

                print(f"\n[Step {step_idx}] Generated token: '{token_str}'")
                print("  RAW top-10 (pre-mask):")
                for tok, p in list(raw_top_k.items())[:10]:
                    print(f"    {tok:25s} {p:.6f}")
                print("  CONSTRAINED (post-mask):")
                masked_probs = dict(zip(
                    [self.tokenizer.convert_ids_to_tokens([tid])[0] for tid in valid_ids],
                    probs,
                ))
                for tok, p in sorted(masked_probs.items(), key=lambda x: -x[1]):
                    print(f"    {tok:25s} {p:.6f}")

                if choice_options:
                    print("  OPTION-LEVEL (first token):")
                    groups: dict = {}
                    for opt in choice_options:
                        first_ids = []
                        for encoding in (' ' + opt, opt):
                            tids = self.tokenizer.encode(encoding, add_special_tokens=False)
                            if tids:
                                first_ids.append(tids[0])
                        key = frozenset(first_ids)
                        groups.setdefault(key, []).append(opt)
                    for first_ids_set, opts in sorted(
                        groups.items(),
                        key=lambda x: -sum(id_to_prob.get(t, 0.0) for t in x[0]),
                    ):
                        p = sum(id_to_prob.get(t, 0.0) for t in first_ids_set)
                        tok_strs = ' / '.join(self.tokenizer.convert_ids_to_tokens(list(first_ids_set)))
                        opts_str = ', '.join(f"'{o}'" for o in opts)
                        print(f"    {tok_strs:35s} → {opts_str}: {p:.6f}")

            step_distributions.append({
                'token_id': token_id,
                'token': self.tokenizer.convert_ids_to_tokens([token_id])[0],
                'probs': {
                    self.tokenizer.convert_ids_to_tokens([tid])[0]: p
                    for tid, p in zip(valid_ids, probs)
                },
                'raw_top_k': raw_top_k,
            })

        return generated_text, step_distributions
