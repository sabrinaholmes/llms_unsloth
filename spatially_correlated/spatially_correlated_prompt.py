import json
def system_message(llm_type='llama',task_type='accumulation'):
    """Construct the system message with dynamic choice options.

    For `centaur` LLMs, insert the instruction about pressing the corresponding
    key as the 4th sentence (right after the sentence about points).
    """
    sentences = [
        "You will be presented with a series of 16 different environments to explore.",
        "In each trial, you can select an option between numbers 1 and 30 by pressing the corresponding key.",
        "By selecting any of these options, you will earn points associated with each unique option.",
        "Imagine these options 1 through 30 as lying next to each other in an ordered line;",
        "options closer to each other tend to have similar rewards as rewards tend to cluster together.",
        "For each environment, you will be able to make either 5 or 10 choices.",
        "When you made all your choices in a given environment, you will start making choices in the next unexplored environment.",
        "The rewards underlying the different options are different in each environment so you will learn them anew for each environment.",
        "Each environment starts with the value of a single option revealed.",
        "When you choose the number corresponding to a different option, you will be told the value of that option and receive those points.",
        "Previously revealed options, including the starting option, can also be reselected, although there may be small changes in the point value.",
    ]

    if llm_type == 'llama':
        sentences.append(
            "Respond with exactly ONE character. "
            "Reply with single character only — do not include quotes, spaces, punctuation, extra words, or newlines."
        )
    if llm_type == 'centaur' and task_type == 'accumulation':
        # insert task objective at the end of the system message for centaur
        sentences.append(
            "It is your task to gain as many points as possible across all 16 environments.")
    if llm_type == 'centaur' and task_type == 'maximization':
        sentences.append(
            "It is your task to to learn where the largest reward is in each of the 16 environments. ")
    return sentences


def build_prediction_centaur_prompt(timeline_df) -> str:
    """Builds the centaur-style prompt for the current trial with past trial data."""
    if timeline_df['scenario'].iloc[0] == 0:
        task_type = 'accumulation'
    elif timeline_df['scenario'].iloc[0] == 1:
        task_type = 'maximization'
    system_msg = system_message(llm_type='centaur', task_type=task_type)
    prompt = "\n".join(system_msg)
    # Parse search history from the participant's row
    search_history = timeline_df['searchHistory'].iloc[0]
    if isinstance(search_history, str):
        search_history = json.loads(search_history)

    num_envs = len(search_history['xcollect'])
    print(f"Number of environments in search history: {num_envs}")
    for env in range(num_envs):
        prompt += f"\n\nEnvironment number {env + 1}:\n"
        # Determine number of choices per environment from horizon
        x = search_history['xcollect'][env]
        y = search_history['ycollectScaled'][env]
        horizon=len(x)-1
        # First element is the initial revealed option
        prompt += f"The value of option {x[0]} is {y[0]}. You have {horizon} choices to make in this environment.\n"

        # Remaining elements are the participant's actual choices
        for trial in range(1, len(x)):
            prompt += f"You press <<{x[trial]}>> and get {y[trial]} points.\n"

    return prompt


def build_generate_prompt_centaur(completed_envs, current_env_history, scenario, horizon):
    """
    Build prompt for open-loop centaur generation.

    Args:
        completed_envs: list of dicts with keys 'x' (options) and 'y' (rewards);
                        x[0]/y[0] is the initial reveal, x[1:]/y[1:] are choices.
        current_env_history: dict with 'x' and 'y' lists for the in-progress environment;
                             x[0]/y[0] = initial reveal, x[1:]/y[1:] = choices so far.
        scenario: 0 = accumulation, 1 = maximization.
        horizon: number of choices the model will make in the current environment.

    Returns:
        Prompt string ending with 'You press <<' to elicit the next choice.
    """
    task_type = 'accumulation' if scenario == 0 else 'maximization'
    sentences = system_message(llm_type='centaur', task_type=task_type)
    prompt = '\n'.join(sentences)

    # Completed environments
    for env_idx, env in enumerate(completed_envs):
        x, y = env['x'], env['y']
        env_horizon = len(x) - 1
        prompt += f'\n\nEnvironment number {env_idx + 1}:\n'
        prompt += f'The value of option {x[0]} is {y[0]}. You have {env_horizon} choices to make in this environment.\n'
        for i in range(1, len(x)):
            prompt += f'You press <<{x[i]}>> and get {y[i]} points.\n'

    # Current in-progress environment
    env_num = len(completed_envs) + 1
    x, y = current_env_history['x'], current_env_history['y']
    prompt += f'\n\nEnvironment number {env_num}:\n'
    prompt += f'The value of option {x[0]} is {y[0]}. You have {horizon} choices to make in this environment.\n'
    for i in range(1, len(x)):
        prompt += f'You press <<{x[i]}>> and get {y[i]} points.\n'

    # Elicit next choice
    prompt += 'You press <<'
    return prompt