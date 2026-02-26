def define_choice_options_from_df(df):
    """ define which choice options were made by participants """
    return df['choice'].unique().tolist()

def system_message(choice_options=None, llm_type='llama'):
    """Construct the system message with dynamic choice options.

    For `centaur` LLMs, insert the instruction about pressing the corresponding
    key as the 4th sentence (right after the sentence about points).
    """
    sentences = [
        f"You are participating in multiple games involving two slot machines, labeled {choice_options[0]} and {choice_options[1]}.",
        "The two slot machines are different across different games.",
        "Each time you choose a slot machine, you get some points.",
        "Each slot machine tends to pay out about the same amount of points on average.",
        "Your goal is to choose the slot machines that will give you the most points across the experiment.",
        "The first 4 trials in each game are INSTRUCTED trials where you will be told which slot machine to choose.",
        "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials."
    ]

    if llm_type == 'centaur':
        # Insert the centaur-specific instruction as the 4th sentence
        sentences.insert(3, "You choose a slot machine by pressing the corresponding key.")

    if llm_type == 'llama':
        sentences.append(f"Respond with exactly ONE character: {choice_options[0]} or {choice_options[1]}. "
                         "Reply with that single character only — do not include quotes, spaces, punctuation, extra words, or newlines.")

    return sentences


def build_full_prompt(df_participant,choice_options=None):
    """Return a full concatenated prompt from a participant dataframe.

    This matches the format used by `horizon_llama.py` (system header, forced blocks,
    then interleaved assistant/user turns). It isolates assistant choices with a
    separating space before the eot marker so tokenizers don't join the token
    with the marker.
    """
    if choice_options is None:
        choice_options = define_choice_options_from_df(df_participant)
    if len(choice_options) < 2:
        raise ValueError(
            "`df_participant['choice']` must contain at least two unique choice options; "
            f"found {len(choice_options)}: {choice_options}"
        )
    system_msg_array = system_message(choice_options=choice_options, llm_type='llama')
    system_msg = " ".join(system_msg_array)
    block_transcript = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>"

    df_sorted = df_participant.sort_values(['game', 'trial'])

    for game_id, game_df in df_sorted.groupby('game'):
        block_transcript += format_single_game_llama(game_id, game_df)
    block_transcript += "\n <|start_header_id|>assistant<|end_header_id|>\n"
    return block_transcript


def format_single_game_llama(game_id, game_df):
    """Format a single game's block for the llama-style `build_full_prompt`.

    This mirrors the original inline logic in `build_full_prompt`: it emits
    the user header with instructed (forced) trials followed by interleaved
    assistant/user turns for free trials.
    If `is_current_game` is True, only include free trials before `current_trial`.
    """
    def _get_free_df(gdf, is_current_game=False, current_trial=None, trial_col='trial'):
        if is_current_game:
            return gdf[(gdf['type'] == 'free') & (gdf[trial_col] < current_trial)].sort_values('trial')
        return gdf[gdf['type'] == 'free'].sort_values('trial')

    forced_df = game_df[game_df['type'] == 'forced'].sort_values('trial')
    forced_lines = [f"{r['choice']}->{r['reward']} points" for _, r in forced_df.iterrows()]
    forced_text = "\n".join(forced_lines)
    total_trials = game_df['trial'].max()

    block = (
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Game {game_id}. There are {total_trials} trials in this game.\n"
        f"(INSTRUCTED TRIALS):{forced_text}<|eot_id|>"
    )
    # Default: include all free trials
    free_df = game_df[game_df['type'] == 'free'].sort_values('trial')
    for _, row in free_df.iterrows():
        block += f"<|start_header_id|>assistant<|end_header_id|>\n{row['choice']}<|eot_id|>"
        block += f"<|start_header_id|>user<|end_header_id|>\n->{row['reward']} points.<|eot_id|>"
    

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
            return format_single_game_llama(game_id, gdf)
        return format_single_game_llama(game_id, game_df)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")


# --- Centaur-style prompt processing ---
def format_single_game_centaur(game_id, game_df, is_current_game=False, current_trial=None,trial_col=None):
    """Formats a single game's header and trials into a block of text."""
    total_trials = game_df[trial_col].max()
    
    # Header for the game
    lines = [f"Game {game_id}. There are {total_trials} trials in this game."]
    
    # 1. Add Forced Trials (always included)
    forced = game_df[game_df["type"] == "forced"]
    for _, row in forced.iterrows():
        lines.append(f"You are instructed to press {row['choice']} and get {row['reward']} points.")
    
    # 2. Add Free Trials
    # If it's the current game, only add trials BEFORE the current decision
    if is_current_game:
        free = game_df[(game_df["type"] == "free") & (game_df[trial_col] < current_trial)]
    else:
        free = game_df[game_df["type"] == "free"]
        
    for _, row in free.iterrows():
        lines.append(f"You press <<{row['choice']}>> and get {row['reward']} points.")
        
    return "\n".join(lines)

def build_multi_game_prompt(block_df, current_game_id, current_trial, trial_col='trial',choice_options=None,llm_type='centaur'):
    """Constructs the full prompt history across all games in a block."""
    # Initial Instructions (Only show this once at the very top)
    system_msg = system_message(llm_type=llm_type, choice_options=choice_options)
    # `system_message` returns a list of sentences; join into a string
    full_prompt = "\n".join(system_msg) if isinstance(system_msg, list) else str(system_msg)

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

    if llm_type == 'llama':
        # 3. Add the final trigger for llama (isolated with a newline and space to prevent token merging)
        full_prompt += "\n <|start_header_id|>assistant<|end_header_id|>\n"
    if llm_type == 'centaur':
        # 3. Add the final trigger for centaur (no extra space needed)
        full_prompt += "\nYou press <<"
    return full_prompt