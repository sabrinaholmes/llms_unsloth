#!/bin/bash
#SBATCH --job-name=predictive_horizon_llama
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:A100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# --- 1. Fix Storage Paths (Prevents No Space Left Error) ---
export HF_HOME="/share/users/student/s/snamazova/.cache/huggingface"
export TMPDIR="/share/users/student/s/snamazova/tmp"
export PIP_CACHE_DIR="/share/users/student/s/snamazova/.cache/pip"

# Ensure directories exist before the script starts
mkdir -p $HF_HOME $TMPDIR $PIP_CACHE_DIR logs

srun python predictive_horizon_llama.py
# Debugging: Print the current working directory and a short PATH snapshot
echo "Current working directory: $(pwd)"
echo "PATH=$PATH"

# Try to initialize conda in a robust way:
# 1) If spack is available, use it to locate miniconda
# 2) Else, check common conda installation locations
# 3) If none found, dump diagnostics to `logs/` for post-mortem

CONDA_SH=""
if command -v spack >/dev/null 2>&1; then
	echo "spack found, attempting to load miniconda3"
	spack load miniconda3 || true
	if spack location -i miniconda3 >/dev/null 2>&1; then
		CONDA_SH="$(spack location -i miniconda3)/etc/profile.d/conda.sh"
	fi
fi

if [ -z "$CONDA_SH" ]; then
	if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
		CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
	elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
		CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
	elif [ -f "/share/conda/etc/profile.d/conda.sh" ]; then
		CONDA_SH="/share/conda/etc/profile.d/conda.sh"
	fi
fi

if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
	echo "Sourcing: $CONDA_SH"
	# shellcheck disable=SC1090
	source "$CONDA_SH"
	if command -v conda >/dev/null 2>&1; then
		conda activate unsloth_env || echo "Warning: conda activate failed"
		echo "Conda environment 'unsloth_env' activated."
	else
		echo "conda executable not found after sourcing $CONDA_SH"
	fi
else
	echo "No conda initialization found. Writing diagnostics to logs/job_env_${SLURM_JOB_ID:-unknown}.txt"
	env > logs/job_env_${SLURM_JOB_ID:-unknown}.txt
	echo "PATH=$PATH" >> logs/job_env_${SLURM_JOB_ID:-unknown}.txt
	echo "Look for conda under $HOME/miniconda3 or /opt/conda or install on compute nodes."
fi

# Prefer any available python; fall back to explicit paths if needed
if command -v python >/dev/null 2>&1; then
	PYTHON_CMD=python
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_CMD=python3
else
	PYTHON_CMD=""
fi

if [ -n "$PYTHON_CMD" ]; then
	echo "Using $PYTHON_CMD to run the job"
	srun $PYTHON_CMD predictive_horizon_llama.py
else
	echo "Error: python not found on compute node. See logs/job_env_${SLURM_JOB_ID:-unknown}.txt for PATH and environment." >&2
	exit 1
fi