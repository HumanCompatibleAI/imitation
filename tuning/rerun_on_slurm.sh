#!/bin/bash
#SBATCH --array=1-5
# Avoid cluttering the root directory with log files:
#SBATCH --output=%x/reruns/%a/sbatch_cout.txt
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
# sbatch --job-name=<name of previous tuning job> rerun_on_slurm.sh
#
# Picks the best trial from the optuna study in <tune_folder> and reruns them with
# the same hyperparameters but different seeds.

# OUTPUT:
# Creates a sub-folder in the given tune_folder for each worker:
# <tune_folder>/reruns/<seed>
# The output of each worker is written to a cout.txt.


source "/nas/ucb/$(whoami)/imitation/venv/bin/activate"

worker_dir="$SLURM_JOB_NAME/reruns/$SLURM_ARRAY_TASK_ID/"

if [ -f "$worker_dir/cout.txt" ]; then
  # This indicates that there is already a worker running in that directory.
  # So we better abort!
  echo "There is already a worker running in this directory. \
    Try different seeds by picking a different array range!"
  exit 1
else
  # Note: we run each worker in a separate working directory to avoid race
  # conditions when writing sacred outputs to the same folder.
  mkdir -p "$worker_dir"
fi

cd "$worker_dir" || exit

srun --output="cout.txt" python ../../../rerun_best_trial.py "../../optuna_study.log" --seed "$SLURM_ARRAY_TASK_ID"
