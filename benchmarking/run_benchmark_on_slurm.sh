#!/bin/bash
#SBATCH --array=1-10
# Avoid cluttering the root directory with log files:
#SBATCH --output=slurm/%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=8gb
#SBATCH --time=70:00:00
#SBATCH --qos=scavenger

# This script will run an imitation algorithm on an environment for 10 seeds.

# This script assumes that you set up imitation in your NAS home directory and
# installed it in a venv located in the imitation directory.

# Call this script with <script> <algo> <env>. Where
#  <scripts> is either 'train_imitation' (then algo must be 'bc' or 'dagger') or
#  'train_adversarial' (then algo must be 'gail' or 'airl').
#  The env can be any of 'seals_ant', 'seals_half_cheetah', 'seals_hopper',
#  'seals_swimmer',  'seals_walker'

cd /nas/ucb/$(whoami)/imitation
source venv/bin/activate
srun python -m imitation.scripts.$1 $2 with $2_$3 seed=$SLURM_ARRAY_TASK_ID -F benchmark_runs
