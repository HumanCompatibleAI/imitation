#!/bin/bash
#SBATCH --array=1-5
# Avoid cluttering the root directory with log files:
#SBATCH --output=%x/%a/sbatch_cout.txt
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8gb
#SBATCH --time=70:00:00
#SBATCH --qos=scavenger
#SBATCH --export=ALL

# DESCRIPTION:
# Reruns the top trials from a previous hyperparameter sweep.

# PREREQUISITES:
# A folder with a hyperparameter sweep as started by tune_on_slurm.sh.

# USAGE:
# sbatch rerun_on_slurm <tune_folder> <top-k>
#
# Picks the top-k trial from the optuna study in <tune_folder> and reruns them with
# the same hyperparameters but different seeds.

# OUTPUT:
# Creates a subfolder in the given tune_folder for each worker:
# <tune_folder>/reruns/top_<top-k>/<seed>
# The output of each worker is written to a cout.txt.


source "/nas/ucb/$(whoami)/imitation/venv/bin/activate"

if [ -z $2 ]; then
  top_k=1
else
  top_k=$2
fi

worker_dir="$1/reruns/top_$top_k/$SLURM_ARRAY_TASK_ID/"

if [ -f "$worker_dir/cout.txt" ]; then
  exit 1
else
  # Note: we run each worker in a separate working directory to avoid race
  # conditions when writing sacred outputs to the same folder.
  mkdir -p "$worker_dir"
fi

cd "$worker_dir" || exit

srun --output="$worker_dir/cout.txt" python ../../rerun_on_slurm.py "$1/optuna_study.log" --top_k "$top_k" --seed "$SLURM_ARRAY_TASK_ID"
