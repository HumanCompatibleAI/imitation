import json
import logging
import os
import random
import subprocess
import sys
from typing import Dict
import optuna
from imitation.util import networks, util
from run_train_imitation import LOGDIR

from stable_baselines3.sac import policies as SACPolicies


def get_sqil_results(config, environment, named_configs=[]):
    config_str = json.dumps(config)

    signature = str(random.getrandbits(128))
    # Run the separate script
    log_path = os.path.join(LOGDIR, signature)  # Define your log directory here
    subprocess.run(["python", "run_train_imitation.py", "--config", config_str, "--log_path", log_path, "--named_configs", json.dumps({"named_configs" : named_configs + [environment]})], check=True)

    # Read the results
    with open(log_path, 'r') as f:
        results = json.load(f)

    print(results)
    return results["imit_stats"]["return_mean"]

def objective_cartpole(trial, environment):
    # TODO: Add more hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 3e-6, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    config = {"rl.rl_kwargs": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }}
    
    return get_sqil_results(config, environment)

    # Convert config to CLI arguments or a temporary file

def objective_cheetah(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True) # too broad?
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512]) # might be too broad
    gradient_steps = trial.suggest_int("gradient_steps", 1, 20)
    tau = trial.suggest_float("tau", 0.001, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 0.999)

    config_updates = {
        "demonstrations.n_expert_demos": 50,
        "sqil.total_timesteps": 1e6,
        # "policy.policy_cls": SACPolicies.SACPolicy,

        "sqil.train_kwargs": {
            "progress_bar": False,
        },
        "rl.rl_kwargs": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_steps": gradient_steps,
            "tau": tau,
            "gamma": gamma,
            "learning_starts": 1000,
        }
    }

    named_configs = ["rl.sac", "policy.sac256"]


    return get_sqil_results(config_updates, "half_cheetah", named_configs)    


def perform_trial(environment):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, n_jobs=20)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")

def perform_cheetah_trial():
    # TODO figure out search type
    # TODO figure out logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cheetah_trial_1"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)

    study.optimize(objective_cheetah, n_trials=1, n_jobs=1)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")

if __name__ == "__main__":
    perform_cheetah_trial()
