#!/bin/bash

#SBATCH --job-name=rl_llama
#SBATCH -t 00:20:00                  # Estimated time, adapt to your needs
#SBATCH --mail-type=all              # Send mail when job begins and ends
#SBATCH -p kisski                    # The partition
#SBATCH -G A100:1                    # Request 1 GPUs
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


module load miniforge3
module load gcc
module load cuda
module load python/3.11.9
# Activate the environment using an absolute path so sbatch finds it regardless of CWD
source $HOME/.project/dir.project/unsloth_env/bin/activate
echo "VENV environment 'unsloth_env' activated."
# Resolve the script directory and switch to it so file paths are relative to this script

# Read the token from token.txt
TOKEN=$(cat token.txt)

# Export the token as an environment variable
export HF_TOKEN="$TOKEN"

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
python predictive_rl_llama.py