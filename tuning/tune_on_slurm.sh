#!/bin/bash
#SBATCH --array=1-100
# Avoid cluttering the root directory with log files:
#SBATCH --output=%x/%a/sbatch_cout.txt
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8gb
#SBATCH --time=70:00:00
#SBATCH --qos=scavenger
#SBATCH --export=ALL

# DESCRIPTION:
# This script is used to tune the hyperparameters of an algorithm on a given
# environment in parallel on a SLURM cluster with 400 trials and 100 workers.

# PREREQUISITES:
# This script assumes that you set up imitation in your NAS home directory and
# installed it in a venv located in the imitation directory.
# /nas/ucb/(your username)/imitation/venv/
# Do this by running the following commands:
# cd /nas/ucb/(your username)/
# git clone https://github.com/HumanCompatibleAI/imitation.git
# srun python3 -m venv venv
# source venv/bin/activate
# srun pip install -e .
# It is important to set up the venv using srun to ensure that the venv is working
# properly on the compute nodes.

# USAGE:
# Run this script with sbatch and pass it the algorithm and the environment
# named-config. For example, to tune PC on CartPole, run:
# sbatch --job-name=tuning_pc_on_cartpole tune_on_slurm.sh pc cartpole
# To change the number of workers, change the --array parameter above
# or pass the --array argument to sbatch.
# To change the number of trials, change the --num_trials parameter below.
# Supported are all algorithms and environments that are supported by the tune.py
# Run tune.py --help for more information.

# OUTPUT:
# This script creates a folder with the name of the SLURM job a numbered sub-folder for
# each worker: <SLURM_JOB_NAME>/<SLURM_ARRAY_TASK_ID>
# The main folder contains the optuna journal .log for synchronizing the workers.
# It is suitable to place this log on a nfs drive shared among all workers.
# Each worker is executed within it's own sub-folder to ensure that their outputs
# do not conflict with each other.
# The output of each worker is written to a cout.txt.
# The output of the sbatch command is written to sbatch_cout.txt.

# CONTINUING A TUNING RUN:
# Often it is desirable to continue an existing job or add more workers to it while it
# is running. Just run run this batch job again but change the --array parameter to
# ensure that the new workers do not conflict with the old ones. E.g. if you first ran
# the batch script with --array=1-100 (the default), a subsequent run should be launched
# with the --array=101-150 (for another 50 workers). For this you do not need to modify
# this file. You can pass it to sbatch to override.

source "/nas/ucb/$(whoami)/imitation/venv/bin/activate"

if [ -f "$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/cout.txt" ]; then
  # Note: this will just be written to sbatch_cout.txt and not to cout.txt to avoid
  # overriding existing cout.txt files. Unfortunately sbatch won't print this for us
  # so it is not very useful information.
  echo "The study folder for $SLURM_JOB_NAME already contains a folder for job $SLURM_ARRAY_TASK_ID!"
  echo "Are you trying to continue on an existing study? Then adapt the sbatch array range!"
  echo "E.g. if the highest folder number in $SLURM_JOB_NAME/ is 100 and you want to continue the study with another 50 runners, start this script using `sbatch --job-name=$SLURM_JOB_NAME --array=101-50 tune_on_slurm.sh $1 $2`"
  exit 1
else
  # Note: we run each worker in a separate working directory to avoid race
  # conditions when writing sacred outputs to the same folder.
  mkdir -p "$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID"
fi

cd "$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID" || exit

srun --output=cout.txt python ../../tune.py --num_trials 400 -j ../optuna_study.log "$1" "$2"
