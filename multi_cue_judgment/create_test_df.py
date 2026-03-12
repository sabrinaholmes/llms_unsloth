import argparse
import pandas as pd
import os

DATA_ORIGINAL= 'data/in/combined_df.csv'
OUTPUT_PATH = 'data/in/test/test_df_exp1.csv'

def find_test_participants(experiment_name: str):
    df_hf = pd.read_json("hf://datasets/marcelbinz/Psych-101-test/prompts_testing_t1.jsonl", lines=True)
    df_exp = df_hf[df_hf['experiment'].str.contains(experiment_name, na=False)]
    print(df_exp.head())
    test_participants = set(df_exp['participant'].dropna())
    print(f"test participant ids{test_participants}")
    return test_participants

def create_test_df(experiment_name: str, output_path: str):
    test_participants = find_test_participants(experiment_name)
    df_original = pd.read_csv(DATA_ORIGINAL)
    ids_in_original = set(df_original['Fp'].unique())
    # to test id add +101 since id start from 1 in the original data but participant ids in the test set start from 0
    test_participants = {int(pid) + 101 for pid in test_participants}
    df_test = df_original[df_original['Fp'].isin(test_participants)]
    #print(f"Participants in test set: {test_participants}")
    #print(f"Participants in original data: {ids_in_original}")
    df_test.to_csv(output_path, index=False)
    print(f"Test DataFrame created with {len(df_test)} rows and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test DataFrame for spatially correlated task")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment to filter participants")
    args = parser.parse_args()
    #if output directory does not exist, create it
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    create_test_df(args.experiment_name, OUTPUT_PATH)
