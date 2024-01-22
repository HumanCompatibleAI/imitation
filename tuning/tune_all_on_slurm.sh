#!/usr/bin/env bash

sbatch --job-name=tuning_pc_on_cartpole tune_on_slurm.sh pc cartpole
sbatch --job-name=tuning_pc_on_seals_ant tune_on_slurm.sh pc seals_ant
sbatch --job-name=tuning_pc_on_seals_half_cheetah tune_on_slurm.sh pc seals_half_cheetah
sbatch --job-name=tuning_pc_on_seals_hopper tune_on_slurm.sh pc seals_hopper
sbatch --job-name=tuning_pc_on_seals_swimmer tune_on_slurm.sh pc seals_swimmer
sbatch --job-name=tuning_pc_on_seals_walker tune_on_slurm.sh pc seals_walker
sbatch --job-name=tuning_pc_on_seals_humanoid tune_on_slurm.sh pc seals_humanoid
sbatch --job-name=tuning_pc_on_seals_cartpole tune_on_slurm.sh pc seals_cartpole
sbatch --job-name=tuning_pc_on_pendulum tune_on_slurm.sh pc pendulum
sbatch --job-name=tuning_pc_on_seals_mountain_car tune_on_slurm.sh pc seals_mountain_car