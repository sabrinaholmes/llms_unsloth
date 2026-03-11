import pandas as pd
import json
import numpy as np
import os

def read_original_kernel(PATH):
    """Reads the original kernel JSON file and converts it to a DataFrame."""
    kernel_df = pd.read_json(PATH)
    return kernel_df

def stack_and_scale(kernel_df, seed=42):
    """Stacks the kernel DataFrame and scales the rewards."""
    # Stack the search history
    stacked = kernel_df.stack().reset_index()
    stacked.columns = ['tile_idx', 'env', 'data']
    stacked[['y', 'x']] = pd.DataFrame(stacked['data'].tolist())

    kernel_flat = stacked.pivot(index='env', columns='x', values='y')
    kernel_flat.columns = [f'tile_{int(x+1)}' for x in kernel_flat.columns]
    kernel_flat = kernel_flat.reset_index()  # keeps 'env' as a column
    # Set the random seed for reproducibility and generate random scales between 65 and 85
    np.random.seed(seed)
    kernel_flat['scale'] = np.random.randint(65, 86, size=len(kernel_flat))
    #start env numbering at 1 instead of 0
    kernel_flat['env'] = kernel_flat['env']+1
    # Scale the rewards
    tile_cols = [f'tile_{i+1}' for i in range(30)]
    kernel_flat_scaled = kernel_flat.copy()
    kernel_flat_scaled[tile_cols] = (kernel_flat[tile_cols].multiply(kernel_flat['scale'], axis=0)).astype(int)
    kernel_flat_scaled.head()

    return kernel_flat_scaled
def save_scaled_kernel(kernel_flat_scaled, output_path):
    """Saves the scaled kernel DataFrame to a CSV file."""
    kernel_flat_scaled.to_csv(output_path, index=False)

def main():
    # define path
    PATH = 'data/in/original_kernels'
    for f in os.listdir(PATH):
        if f.endswith('.json'):
            print(f"Processing {f}...")
            kernel_df = read_original_kernel(f"{PATH}/{f}")
            kernel_flat_scaled = stack_and_scale(kernel_df)
            output_path = f"data/in/scaled_kernels/{f.replace('.json', '_scaled.csv')}"
            if not os.path.exists("data/in/scaled_kernels"):
                os.makedirs("data/in/scaled_kernels")
            save_scaled_kernel(kernel_flat_scaled, output_path)
            print(f"Saved scaled kernel to {output_path}")

if __name__ == "__main__":
    main()