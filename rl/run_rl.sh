#!/bin/bash
#SBATCH --job-name=llama-70B-predictive-rl
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- 1. Fix Storage Paths (Prevents No Space Left Error) ---
export HF_HOME="/share/users/student/s/snamazova/.cache/huggingface"
export TMPDIR="/share/users/student/s/snamazova/tmp"
export PIP_CACHE_DIR="/share/users/student/s/snamazova/.cache/pip"

# Ensure directories exist before the script starts
mkdir -p $HF_HOME $TMPDIR $PIP_CACHE_DIR logs

# Debugging: Print the current working directory and environment
echo "Current working directory: $(pwd)"
# 1. Load the tool from Spack
spack load miniconda3

# 2. Source the conda profile so the 'activate' command works
# Note: The path below is the standard one for Spack-installed Miniconda
source $(spack location -i miniconda3)/etc/profile.d/conda.sh

# 3. Now you can activate and run
conda activate unsloth_env
echo "Conda environment 'unsloth_env' activated."

srun python predictive_rl_llama.py