import json
import pandas as pd
def system_message(task_type='verbal',exp_mixed=False,):
    sentences = [
        "Your task is to estimate the blood concentration of the hormone Caldionine based on information about the amount of two other hormones, Progladine and Amalydine, in multiple individuals' urine.",
        # ← removed hardcoded numerical lines here
        "Your goal is to estimate the concentration of Caldionine correctly.",
        "You will receive feedback about the actual concentration after making your estimate.",
        "This feedback will stop at some point.\n"
    ]
    if exp_mixed:
        if task_type == 'verbal':
            sentences.insert(1, "Both Progladine and Amalydine can take five values (1, 2, 3, 4, 5).\nCaldionine can take nine values (extremely low, very low, low, somewhat low, normal, somewhat high, high, very high, extremely high).")
        elif task_type == 'numerical':
            sentences.insert(1, "Both Progladine and Amalydine can take five values (very little, a little, average, a lot, very much).\nCaldionine can take nine values (10, 20, 30, 40, 50, 60, 70, 80, 90).")
    else:
        if task_type == 'verbal':
            sentences.insert(1, "Both Progladine and Amalydine can take five values (very little, a little, average, a lot, very much).\nCaldionine can take nine values (extremely low, very low, low, somewhat low, normal, somewhat high, high, very high, extremely high).")
        elif task_type == 'numerical':
            sentences.insert(1, "Both Progladine and Amalydine can take five values (1, 2, 3, 4, 5).\nCaldionine can take nine values (10, 20, 30, 40, 50, 60, 70, 80, 90).")

    return sentences

def build_predict_centaur_prompt(timeline: pd.DataFrame,exp_mixed=False,limited_train=False) -> str:
    """Builds the multi-cue prediction prompt from all trials in the timeline.

    Iterates through each trial. For condition 1 or 3 (English cues/responses),
    and condition 2 or 4 (numeric cues/responses), appends the error-feedback
    line using the values already stored in each column.
    Response is wrapped in <<...>> for downstream NLL evaluation.
    """
    participant_condition=timeline['condition'].iloc[0]
    print(f"Building prompt for participant with condition {participant_condition}")
    system_msg = system_message(task_type='verbal' if participant_condition in [1,3] else 'numerical',exp_mixed=exp_mixed)
    prompt = "\n".join(system_msg)+"\n"
    for _, row in timeline.iterrows():
        #if cues,response,criterium are strings, then lowercase them for better readability
        cue1 = row['Cue1_English'].lower() if isinstance(row['Cue1_English'], str) else row['Cue1_English']
        cue2 = row['Cue2_English'].lower() if isinstance(row['Cue2_English'], str) else row['Cue2_English']
        raw_response = row['response_raw'].lower() if isinstance(row['response_raw'], str) else row['response_raw']
        criterium = row['Criterium_English'].lower() if isinstance(row['Criterium_English'], str) else row['Criterium_English']
        condition = row['condition']
        block=row['Block']
        prompt += (
                f"Progladine: {cue1}. Amalydine: {cue2}. "
                f"You say that the Caldionine concentration is <<{raw_response}>>."
            )
        if limited_train:
            train_blocks=6
        else:            
            train_blocks=10

        
        if block<=train_blocks:
            if raw_response == criterium:
                prompt += f" That is correct. The correct concentration of Caldionine is indeed {criterium}.\n"
            else:
                prompt += f" That is incorrect. The correct concentration of Caldionine is {criterium}.\n"
        else:
            prompt += " \n"
    return prompt

def build_generate_centaur_prompt(timeline: pd.DataFrame) -> str:
    """Open-loop prompt for generation: past trials with feedback, current trial ends with <<.

    All completed trials (iloc[:-1]) are appended with their responses and feedback.
    The last row (current trial) contributes only its cues, ending with << so the
    model can generate its answer.
    """
    participant_condition = timeline['condition'].iloc[0]
    task_type = 'verbal' if participant_condition in [1, 3] else 'numerical'
    system_msg = system_message(llm_type='centaur', task_type=task_type)
    prompt = "\n".join(system_msg) + "\n"

    past_trials = timeline.iloc[:-1]
    current = timeline.iloc[-1]

    for _, row in past_trials.iterrows():
        cue1 = row['Cue1_English'].lower() if isinstance(row['Cue1_English'], str) else row['Cue1_English']
        cue2 = row['Cue2_English'].lower() if isinstance(row['Cue2_English'], str) else row['Cue2_English']
        raw_response = row['response_raw'].lower() if isinstance(row['response_raw'], str) else row['response_raw']
        criterium = row['Criterium_English'].lower() if isinstance(row['Criterium_English'], str) else row['Criterium_English']
        block = row['Block']

        prompt += (
            f"Progladine: {cue1}. Amalydine: {cue2}. "
            f"You say that the Caldionine concentration is <<{raw_response}>>."
        )
        if block <= 10:
            if raw_response == criterium:
                prompt += f" That is correct. The correct concentration of Caldionine is indeed {criterium}.\n"
            else:
                prompt += f" That is incorrect. The correct concentration of Caldionine is {criterium}.\n"
        else:
            prompt += " \n"

    cur_cue1 = current['Cue1_English'].lower() if isinstance(current['Cue1_English'], str) else current['Cue1_English']
    cur_cue2 = current['Cue2_English'].lower() if isinstance(current['Cue2_English'], str) else current['Cue2_English']
    prompt += f"Progladine: {cur_cue1}. Amalydine: {cur_cue2}. You say that the Caldionine concentration is <<"
    return prompt


def build_predict_llama_prompt(timeline: pd.DataFrame) -> str:
    """Builds the multi-cue prediction prompt from all trials in the timeline.

    Iterates through each trial. For condition 1 or 3 (English cues/responses),
    and condition 2 or 4 (numeric cues/responses), appends the error-feedback
    line using the values already stored in each column.
    Response is wrapped in <<...>> for downstream NLL evaluation.
    """
    participant_condition=timeline['condition'].iloc[0]
    system_msg = system_message(task_type='verbal' if participant_condition in [1,3] else 'numerical')
    prompt = "\n".join(system_msg)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{prompt}<|eot_id|>\n"
    for _, row in timeline.iterrows():
        #if cues,response,criterium are strings, then lowercase them for better readability
        cue1 = row['Cue1_English'].lower() if isinstance(row['Cue1_English'], str) else row['Cue1_English']
        cue2 = row['Cue2_English'].lower() if isinstance(row['Cue2_English'], str) else row['Cue2_English']
        raw_response = row['response_raw'].lower() if isinstance(row['response_raw'], str) else row['response_raw']
        criterium = row['Criterium_English'].lower() if isinstance(row['Criterium_English'], str) else row['Criterium_English']
        condition = row['condition']
        block=row['Block']
        prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Progladine: {cue1}. Amalydine: {cue2}.<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n{raw_response}<|eot_id|>\n"
            )
        if block<=10:
            if raw_response == criterium:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n" f"Correct. The correct concentration of Caldionine is indeed {criterium}.\n"
            else:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n" f"Incorrect. The correct concentration of Caldionine is {criterium}.\n"
        else:
            prompt += " \n"
    return prompt


def build_generate_llama_prompt(timeline: pd.DataFrame) -> str:
    """Open-loop prompt for generation: past trials with feedback, current trial ends with <<.

    All completed trials (iloc[:-1]) are appended with their responses and feedback.
    The last row (current trial) contributes only its cues, ending with << so the
    model can generate its answer.
    """
    participant_condition = timeline['condition'].iloc[0]
    task_type = 'verbal' if participant_condition in [1, 3] else 'numerical'
    system_msg = system_message(llm_type='llama', task_type=task_type)
    prompt = "\n".join(system_msg)
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{prompt}<|eot_id|>\n"

    past_trials = timeline.iloc[:-1]
    current = timeline.iloc[-1]

    for _, row in past_trials.iterrows():
        cue1 = row['Cue1_English'].lower() if isinstance(row['Cue1_English'], str) else row['Cue1_English']
        cue2 = row['Cue2_English'].lower() if isinstance(row['Cue2_English'], str) else row['Cue2_English']
        raw_response = row['response_raw'].lower() if isinstance(row['response_raw'], str) else row['response_raw']
        criterium = row['Criterium_English'].lower() if isinstance(row['Criterium_English'], str) else row['Criterium_English']
        block = row['Block']

        prompt += (
                f"<|start_header_id|>user<|end_header_id|>\n"
                f"Progladine: {cue1}. Amalydine: {cue2}.<|eot_id|>\n"
                f"<|start_header_id|>assistant<|end_header_id|>\n{raw_response}.<|eot_id|>\n"
            )
        if block <= 10:
            if raw_response == criterium:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n" f"Correct. The correct concentration of Caldionine is indeed {criterium}.\n"
            else:
                prompt += f"<|start_header_id|>user<|end_header_id|>\n" f"Incorrect. The correct concentration of Caldionine is {criterium}.\n"
        else:
            prompt += " \n"

    cur_cue1 = current['Cue1_English'].lower() if isinstance(current['Cue1_English'], str) else current['Cue1_English']
    cur_cue2 = current['Cue2_English'].lower() if isinstance(current['Cue2_English'], str) else current['Cue2_English']
    prompt += f"<|start_header_id|>user<|end_header_id|>\n" f"Progladine: {cur_cue1}. Amalydine: {cur_cue2}<|eot_id|>\n"
    prompt+= f"<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt