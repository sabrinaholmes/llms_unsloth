def define_choice_options_from_df(df):
    """ define which choice options were made by participants """
    import ast
    value = df['choice_options'].iloc[0]
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)

def system_message(choice_options=None, llm_type='llama'):
    """Construct the system message with dynamic choice options.

    For `centaur` LLMs, insert the instruction about pressing the corresponding
    key as the 4th sentence (right after the sentence about points).
    """
    sentences = [
        f"You are participating in multiple games involving two slot machines, labeled {choice_options[0]} and {choice_options[1]}.",
        "The two slot machines are different across different games.",
        "Each time you choose a slot machine, you get some points.",
        "You choose a slot machine by pressing the corresponding key.",
        "Each slot machine tends to pay out about the same amount of points on average.",
        "Your goal is to choose the slot machines that will give you the most points across the experiment.",
        "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose.",
        "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials."
    ]

    if llm_type == 'llama':
        sentences.append(f"Respond with exactly ONE character: {choice_options[0]} or {choice_options[1]}. "
                         "Reply with that single character only — do not include quotes, spaces, punctuation, extra words, or newlines.")

    return sentences


def format_single_game_llama(game_id, game_df, is_current_game=False):
    """Format a single game's block for the llama-style prompt.

    Game 1 opens a fresh user turn. Games 2+ append into the previous game's
    last open user reward turn (no redundant user header). Completed games
    leave their final user reward turn open so the next game can continue it.
    """
    forced_df = game_df[game_df['type'] == 'forced'].sort_values('trial')
    forced_lines = [
        f"{r['mapped_choice']}: {r['reward']} points."
        for _, r in forced_df.iterrows()
    ]
    forced_text = "\n".join(forced_lines)
    total_trials = game_df['trial'].max()

    if game_id == 1:
        # Open a fresh user turn
        block = f"\n<|start_header_id|>user<|end_header_id|>\nGame {game_id}. There are {total_trials} trials in this game.\n{forced_text}<|eot_id|>"
    else:
        # Continue the previous game's open user reward turn, then close it
        block = f"\n\nGame {game_id}. There are {total_trials} trials in this game.\n{forced_text}<|eot_id|>"

    free_df = game_df[game_df['type'] == 'free'].sort_values('trial')
    free_rows = list(free_df.iterrows())
    for i, (_, row) in enumerate(free_rows):
        is_last = (i == len(free_rows) - 1)
        block += f"<|start_header_id|>assistant<|end_header_id|>\n{row['mapped_choice']}<|eot_id|>"
        if is_last and not is_current_game:
            # Leave open so the next game can append into this user turn
            block += f"<|start_header_id|>user<|end_header_id|>\n{row['reward']} points."
        else:
            block += f"<|start_header_id|>user<|end_header_id|>\n{row['reward']} points.<|eot_id|>"

    return block


def format_single_game(game_id, game_df, llm_type='centaur', is_current_game=False, current_trial=None, trial_col='trial'):
    """Dispatch to the appropriate per-game formatter for `llm_type`.

    - For `centaur`, uses `format_single_game_centaur` which produces natural-language
      instructions and respects `is_current_game`/`current_trial`.
    - For `llama`, uses `format_single_game_llama` which produces the token-marked
      blocks used by the llama prompt format and also respects current-game filtering.
    """
    if llm_type == 'centaur':
        return format_single_game_centaur(game_id, game_df, is_current_game=is_current_game, current_trial=current_trial, trial_col=trial_col)
    elif llm_type == 'llama':
        # For llama, format_single_game_llama currently includes all free trials.
        # If we need to limit to trials before the current decision, emulate that here.
        if is_current_game:
            # make a copy with filtered free trials
            gdf = game_df.copy()
            gdf = gdf[(gdf['type'] == 'forced') | ((gdf['type'] == 'free') & (gdf[trial_col] < current_trial))]
            return format_single_game_llama(game_id, gdf, is_current_game=True)
        return format_single_game_llama(game_id, game_df, is_current_game=False)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")


# --- Centaur-style prompt processing ---
def format_single_game_centaur(game_id, game_df, is_current_game=False, current_trial=None,trial_col=None):
    """Formats a single game's header and trials into a block of text."""
    total_trials = game_df[trial_col].max()
    # Header for the game
    lines = [f"\n\nGame {game_id}. There are {total_trials} trials in this game."]
    
    # 1. Add Forced Trials (always included)
    forced = game_df[game_df["type"] == "forced"]
    #print(f"columns: {game_df.columns}")
    for _, row in forced.iterrows():
        lines.append(f"You are instructed to press {row['mapped_choice']} and get {row['reward']} points.")
    
    # 2. Add Free Trials
    # If it's the current game, only add trials BEFORE the current decision
    if is_current_game:
        free = game_df[(game_df["type"] == "free") & (game_df[trial_col] < current_trial)]
    else:
        free = game_df[game_df["type"] == "free"]
        
    for _, row in free.iterrows():
        lines.append(f"You press <<{row['mapped_choice']}>> and get {row['reward']} points.")
        
    return "\n".join(lines)

def build_multi_game_prompt(block_df, current_game_id, current_trial, trial_col='trial',choice_options=None,llm_type='centaur',eval=False):
    """Constructs the full prompt history across all games in a block."""
    # Initial Instructions (Only show this once at the very top)
    system_msg = system_message(llm_type=llm_type, choice_options=choice_options)
    if llm_type =='llama':
        # For llama, join the system message sentences into a single string with newlines, since the llama prompt format expects a single string for the system message.
        system_msg = "\n".join(system_msg)
        full_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>"
    else:
    # `system_message` returns a list of sentences; join into a string
        full_prompt = "\n".join(system_msg)

    # 1. Add all COMPLETED games
    past_games_ids = sorted(block_df[block_df['game'] < current_game_id]['game'].unique())
    for g_id in past_games_ids:
        game_data = block_df[block_df['game'] == g_id]
        full_prompt += format_single_game(g_id, game_data, llm_type=llm_type, trial_col=trial_col)

    # 2. Add the CURRENT game history (only trials before current decision)
    current_game_data = block_df[block_df['game'] == current_game_id]
    full_prompt += format_single_game(
        current_game_id,
        current_game_data,
        llm_type=llm_type,
        is_current_game=True,
        current_trial=current_trial,
        trial_col=trial_col
    )
    if eval:
        #print(f"trigger will be added at the end of this prompt: \n{full_prompt[-500:]}")
        if llm_type == 'centaur':
            full_prompt+="\nYou press <<"
            #print(f"full prompt with trigger added: \n{full_prompt[-500:]}")
        elif llm_type == 'llama':
            full_prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    return full_prompt