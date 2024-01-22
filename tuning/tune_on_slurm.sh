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
# To change the number of workers, change the --array parameter above.
# To change the number of trials, change the --num_trials parameter below.
# Supported are all algorithms and environments that are supported by the tune.py
# Run tune.py --help for more information.

# OUTPUT:
# This script creates a folder with the name of the SLURM job ID and a numbered
# subfolder for each worker: <SLURM_JOB_ID>/<SLURM_ARRAY_TASK_ID>
# The main folder contains the optuna journal for synchronizing the workers.
# Each worker is executed within it's own subfolder to ensure that their outputs
# do not conflict with each other. The output of each worker is written to a cout.txt.

source "/nas/ucb/$(whoami)/imitation/venv/bin/activate"

# Note: we run each worker in a separate working directory to avoid race
# conditions when writing sacred outputs to the same folder.
mkdir -p "$SLURM_ARRAY_JOB_ID"/"$SLURM_ARRAY_TASK_ID"
cd "$SLURM_ARRAY_JOB_ID"/"$SLURM_ARRAY_TASK_ID" || exit

srun python ../../tune.py --num_trials 400 -j ../"$1"_"$2".log "$1" "$2"