import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import get_models
import pandas as pd
import numpy as np
import random
import re
import torch
import os
import gc
import hashlib
import gzip
from spatially_correlated_prompt import build_generate_prompt_centaur,build_generate_prompt_llama


DATA_IN_ = 'data/in/scaled_kernels'
MODEL = 'llama-8B-adapter'  # Change this to the desired model name
DATA_FOLDER_OUT = f'data/out/generative/{MODEL}/singles'
PROMPT_DIR = os.path.join(DATA_FOLDER_OUT, "prompts")
NUMBER_OF_PARTICIPANTS = 81
NUMBER_OF_PARTICIPANTS_PER_SCENARIO_1 = 45  # For scenario 1 (accumulation), we have 45 participants, which is more than half of the total participants. The remaining 36 participants will be assigned to scenario 2 (maximization).
NUMBER_OF_PARTICIPANTS_PER_SCENARIO_2 = NUMBER_OF_PARTICIPANTS - NUMBER_OF_PARTICIPANTS_PER_SCENARIO_1
RUN_TEST_MODE = True if '8B' in MODEL else False  # Set to True to only run on small number of simulations for test

LLM_TYPE='llama' if 'llama' in MODEL else 'centaur'
choice_options = [str(i) for i in range(1, 31)]  # Model generates just the number after 'You press <<'
def save_prompt(prompt: str) -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()
    path = os.path.join(PROMPT_DIR, f"{h}.txt.gz")
    if not os.path.exists(path):
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(prompt)
    return h

def simulate_participant(timeline_df, wrapper, participant_id, scenario=0, n_envs=16, kernel_type='rough', seed=None):
    """
    Open-loop centaur simulation: model generates choices step-by-step
    using kernel reward landscapes.

    Args:
        timeline_df: kernel1_flat or kernel2_flat — rows are environments,
                     columns: env, tile_0..tile_29 (float 0-1), scale (int 65-85).
                     Also accepts pre-scaled versions (tile values as ints).
        pipe: transformers text-generation pipeline.
        tokenizer: The corresponding tokenizer.
        participant_id: unique participant index used as default RNG seed.
        scenario: 0 = accumulation, 1 = maximization.
        horizon: choices per environment; if None, randomly 5 or 10 per env.
        n_envs: number of environments to simulate (default 16).
        kernel_type: type of kernel (rough or smooth).
        seed: explicit RNG seed; defaults to participant_id.

    Returns:
        results: list of dicts per trial with env_idx, env_id, trial, choice, reward.
    """
    random_seed = seed if seed is not None else participant_id
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    envs = timeline_df['env'].unique()
    selected_envs = random.sample(list(envs), n_envs)
    print(f'Selected environments: {selected_envs}')
    #assign horizon 5 to half of the environments and 10 to the other half if horizon is None and save assignments in a dictionary
    half = n_envs // 2
    horizon_assignments = [5] * half + [10] * (n_envs - half)
    random.shuffle(horizon_assignments)
    horizon_dict = dict(zip(selected_envs, horizon_assignments))
    completed_envs = []
    results = []

    for env_idx, env_id in enumerate(selected_envs):
        env_row = timeline_df[timeline_df['env'] == env_id].iloc[0]
        env_horizon = horizon_dict[env_id]

        # Random initial revealed option
        init_option = random.randint(1, 30)
        # Get reward for the initial revealed option from the kernel
        def get_reward(option):
            base_reward = env_row[f'tile_{option}']
            # add some normally distributed noise to the reward with mean 0 and std 1
            return int(base_reward + np.random.normal(0, 1))
        init_reward = get_reward(init_option)
        current_env = {'x': [init_option], 'y': [init_reward]}
        print(f'\nEnv {env_idx + 1} (id={env_id}): initial option={init_option}, reward={init_reward}, horizon={env_horizon}')

        for trial in range(env_horizon):
            if LLM_TYPE == 'centaur':
                prompt = build_generate_prompt_centaur(
                    completed_envs, current_env, scenario, env_horizon
                )
            else:
                prompt = build_generate_prompt_llama(
                    completed_envs, current_env, scenario, env_horizon
                )
            choice = wrapper.generate(prompt, choice_options=choice_options,processor_type='prefix_tree')
            if choice is not None:
                choice = choice.strip()  # Remove any leading/trailing whitespace
            if choice is not None and choice != 'None':
                choice=int(choice)
                if choice not in range(1, 31):
                    print(f"⚠️ Invalid choice '{choice}' generated")
                    choice = None
                    reward = 0
                else:
                    reward = get_reward(choice)
            else:
                choice = None
                reward = 0
            print(f'Trial {trial + 1}: raw response={choice}')
            
            current_env['x'].append(choice)
            current_env['y'].append(reward)

            results.append({
                'env_idx': env_idx + 1,
                'env_id': env_id,
                'trial': trial + 1,
                'choice': choice,
                'scenario': scenario,
                'kernel': kernel_type,
                'init_option': init_option,
                'horizon': env_horizon,
                'init_reward': init_reward,
                'reward': reward,
                'prompt_hash': save_prompt(prompt),
                'random_seed': random_seed,
            })
            print(f'  Choice={choice}, Reward={reward}, Number of invalid choices so far: {sum(1 for r in results if r["choice"] is None)}')

        completed_envs.append(current_env)

    return results

def main():
    
    if not os.path.exists(DATA_FOLDER_OUT):
        os.makedirs(DATA_FOLDER_OUT)
    if not os.path.exists(PROMPT_DIR):
        os.makedirs(PROMPT_DIR)
    #load kernels for rough(kernel1_flat) and smooth(kernel2_flat) reward landscapes
    kernel1_df = pd.read_csv(f'{DATA_IN_}/kernel1_scaled.csv')
    kernel2_df = pd.read_csv(f'{DATA_IN_}/kernel2_scaled.csv')
    wrapper = get_models.ModelWrapper(MODEL, use_unsloth=True, temperature=1.0)
    torch.cuda.empty_cache()
    all_model_results = []
    for scenario in [0, 1]:
        if RUN_TEST_MODE:
            print(f"Running test mode for scenario {scenario} only with 2 participants (2 per kernel type)")
            NUMBER_OF_PARTICIPANTS_PER_SCENARIO_1 = 2
            NUMBER_OF_PARTICIPANTS_PER_SCENARIO_2 = 2

        num_participants= NUMBER_OF_PARTICIPANTS_PER_SCENARIO_1 if scenario == 0 else NUMBER_OF_PARTICIPANTS_PER_SCENARIO_2
        # for each scenario(maximization or accumulation), sample half rough and half smooth kernels for the participants
        #number of participants per kernel type
        num_participants_per_kernel = num_participants // 2
        # if num_participants is odd, assign the extra participant to the smooth kernel
        if num_participants % 2 != 0:
            num_participants_per_kernel += 1

        num_participants_kernel1 = num_participants - num_participants_per_kernel  # rough
        num_participants_kernel2 = num_participants_per_kernel  # smooth

        kernel_assignments = (
            [(kernel2_df, 'smooth')] * num_participants_kernel2 +
            [(kernel1_df, 'rough')] * num_participants_kernel1
        )

        for kernel_df, kernel_type in kernel_assignments:
            participant_id = len(all_model_results)
            out_path = f'{DATA_FOLDER_OUT}/participant_{participant_id}_scenario_{scenario}_kernel_{kernel_type}.csv'
            if os.path.exists(out_path):
                print(f"File {out_path} already exists. Skipping participant {participant_id}.")
                all_model_results.append({'participant_id': participant_id, 'scenario': scenario, 'kernel': kernel_type})
                continue
            print(f'\nSimulating participant {participant_id} (scenario={scenario}, kernel={kernel_type})')
            gc.collect()
            torch.cuda.empty_cache()
            results = simulate_participant(kernel_df, wrapper, participant_id=participant_id, scenario=scenario, kernel_type=kernel_type)
            result_df = pd.DataFrame(results)
            result_df.to_csv(out_path, index=False)
            print(f'Results saved to {out_path}')
            all_model_results.append({'participant_id': participant_id, 'scenario': scenario, 'kernel': kernel_type})
            gc.collect()
            torch.cuda.empty_cache()

    all_model_df = pd.DataFrame(all_model_results)
    summary_path = f'{DATA_FOLDER_OUT}/all_participants_summary.csv'
    all_model_df.to_csv(summary_path, index=False)
    print(f'Summary saved to {summary_path}')

if __name__ == "__main__":
    main()
