#!/bin/bash

#SBATCH --job-name=gen_horizon_centaur
#SBATCH -t 04:00:00                  # Estimated time, adapt to your needs
#SBATCH --mail-type=all              # Send mail when job begins and ends
#SBATCH -p kisski-h100                   # The partition
#SBATCH -G H100:1                    # Request 1 GPUs
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

# Set cluster proxy so compute nodes use the institution proxy for internet access
# (ensure www-cache.gwdg.de:3128 is reachable from the compute nodes)
export http_proxy="http://www-cache.gwdg.de:3128"
export https_proxy="http://www-cache.gwdg.de:3128"
export ftp_proxy="http://www-cache.gwdg.de:3128"
# Uppercase variants for programs that read those
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export FTP_PROXY="$ftp_proxy"
# Don't proxy local addresses
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"
python generate_horizon.py