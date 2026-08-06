def define_choice_options_from_df(df):
    """ define which choice options were made by participants """
    return df['choice'].unique().tolist()

def system_message(choice_options=None, llm_type='llama'):
    """Construct the system message with dynamic choice options.

    For `centaur` LLMs, insert the instruction about pressing the corresponding
    key as the 4th sentence (right after the sentence about points).
    """
    sentences = [
        f"In this task, you have to repeatedly choose between two slot machines labeled {choice_options[0]} and {choice_options[1]}.",
        "You can choose a slot machine by pressing its corresponding key.",
        "When you select one of the machines, you will win 1 or 0 points.",
        "Your goal is to choose the slot machines that will give you the most points.",
        "You will receive feedback about the outcome after making a choice.",
        "The environment may change unpredictably; past success does not guarantee future results.",
        "You will play 1 game in total, consisting of 100 trials.",
    ]


    if llm_type == 'llama':
        sentences.append(f"Respond with exactly ONE character: {choice_options[0]} or {choice_options[1]}. "
                         "Reply with that single character only — do not include quotes, spaces, punctuation, extra words, or newlines.")

    return sentences

def build_llama_prompt(past_trials: list,choice_options: list) -> str:
    """
    Builds a multi-turn llama-style prompt where each past choice is an 'assistant' message
    and each reward is a 'user' message.
    """
    system_msg=system_message(choice_options=choice_options, llm_type='llama')
    system_text = "\n".join(system_msg)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_text}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\nGame 1."

    if not past_trials:
        prompt += "No trials completed yet."
    else:
        for trial in past_trials:
            prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{trial['choice_mapped']}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n{trial['reward']} points."

    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


def build_centaur_prompt(past_trials: list, choice_options: list) -> str:
    """Builds the centaur-style prompt for the current trial with past trial data."""
    system_msg=system_message(choice_options=choice_options, llm_type='centaur')
    prompt = "\n".join(system_msg)
    prompt += "\nGame 1."
    for past_trial in past_trials:
        prompt += f"You press <<{past_trial['choice_mapped']}>> and get {past_trial['reward']} points.\n"

    prompt += f"You press <<"
    return prompt

def build_llama_prompt_no_rewards(past_trials: list, choice_options: list) -> str:
    """Builds a llama-style prompt without including reward information."""
    system_msg=system_message(choice_options=choice_options, llm_type='llama')
    system_text = "\n".join(system_msg)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_text}<|eot_id|>"
    prompt += f"<|start_header_id|>user<|end_header_id|>\nGame 1."

    if not past_trials:
        prompt += "No trials completed yet."
    else:
        for trial in past_trials:
            prompt += f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{trial['choice_mapped']}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n"

    prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt


def build_centaur_prompt_no_rewards(past_trials: list, choice_options: list) -> str:
    """Builds a centaur-style prompt without including reward information."""
    system_msg=system_message(choice_options=choice_options, llm_type='centaur')
    prompt = "\n".join(system_msg)
    prompt += "\nGame 1."
    for past_trial in past_trials:
        prompt += f"You press <<{past_trial['choice_mapped']}>>.\n"

    prompt += f"You press <<"
    return prompt