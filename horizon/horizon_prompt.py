import os

def build_full_prompt(df_participant):
    """Return a full concatenated prompt from a participant dataframe.

    This matches the format used by `horizon_llama.py` (system header, forced blocks,
    then interleaved assistant/user turns). It isolates assistant choices with a
    separating space before the eot marker so tokenizers don't join the token
    with the marker.
    """
    system_msg = (
        "You are participating in multiple games involving two slot machines, labeled I and H."
        "The two slot machines are different across different games."
        "Each time you choose a slot machine, you get some points."
        "Each slot machine tends to pay out about the same amount of points on average."
        "Your goal is to choose the slot machines that will give you the most points across the experiment."
        "The first 4 trials in each game are instructed trials where you will be told which slot machine to choose."
        "After these instructed trials, you will have the freedom to choose for either 1 or 6 trials."
        "Respond with ONLY 'I' or 'H'."
    )

    block_transcript = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>"

    df_sorted = df_participant.sort_values(['game', 'trial'])

    for game_id, game_df in df_sorted.groupby('game'):
        forced_df = game_df[game_df['type'] == 'forced'].sort_values('trial')
        forced_lines = [f"{r['choice']}->{r['reward']} points" for _, r in forced_df.iterrows()]
        forced_text = "\n".join(forced_lines)

        block_transcript += (
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"Game {game_id}(Instructed):{forced_text}<|eot_id|>"
        )

        free_df = game_df[game_df['type'] == 'free'].sort_values('trial')
        for _, row in free_df.iterrows():
            # isolate assistant choice and add a space before the marker
            block_transcript += (
                f"<|start_header_id|>assistant<|end_header_id|>\n{row['choice']} <|eot_id|>"
            )
            block_transcript += (
                f"<|start_header_id|>user<|end_header_id|>\n->{row['reward']} points.<|eot_id|>"
            )

    return block_transcript
