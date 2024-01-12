#!/bin/bash
#SBATCH --array=1-10
# Avoid cluttering the root directory with log files:
#SBATCH --output=slurm/%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8gb
#SBATCH --time=70:00:00
#SBATCH --qos=scavenger

# This script assumes that you set up imitation in your NAS home directory and
# installed it in a venv located in the imitation directory.

# Call this script with the <algo> <env_named_config> parameters to be passed to the
# tune.py script.

cd "/nas/ucb/$(whoami)/" || exit
source imitation/venv/bin/activate

# Note: we run each worker in a separate working directory to avoid race
# conditions when writing sacred outputs to the same folder.
mkdir workdir_"$1"_"$2"_"$SLURM_ARRAY_TASK_ID"
cd workdir_"$1"_"$2"_"$SLURM_ARRAY_TASK_ID" || exit

srun python ../imitation/tuning/tune.py -j ../"$1"_"$2".log "$1" "$2"
