#!/bin/bash
#SBATCH --array=1-100
# Avoid cluttering the root directory with log files:
#SBATCH --output=%A/%a/cout.txt
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8gb
#SBATCH --time=70:00:00
#SBATCH --qos=scavenger
#SBATCH --export=ALL

# This script assumes that you set up imitation in your NAS home directory and
# installed it in a venv located in the imitation directory.

# Call this script with the <algo> <env_named_config> parameters to be passed to the
# tune.py script.

source "/nas/ucb/$(whoami)/imitation/venv/bin/activate"

# Note: we run each worker in a separate working directory to avoid race
# conditions when writing sacred outputs to the same folder.
mkdir -p "$SLURM_ARRAY_JOB_ID"/"$SLURM_ARRAY_TASK_ID"
cd "$SLURM_ARRAY_JOB_ID"/"$SLURM_ARRAY_TASK_ID" || exit

srun python ../../tune.py --num_trials 400 -j ../"$1"_"$2".log "$1" "$2"