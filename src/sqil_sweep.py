import functools
import json
import logging
import os
import random
import subprocess
import sys
from typing import Dict

import optuna
from stable_baselines3.sac import policies as SACPolicies

from imitation.util import networks, util
from run_train_imitation import LOGDIR

fast_configs = [
    "environment.fast",
    "demonstrations.fast",
    "policy_evaluation.fast",
    "fast",
]
environments = [
    ("cheetah", "seals/HalfCheetah-v1"),
    ("ant", "seals/Ant-v1"),
    # ("walker", "seals/Walker2d-v1"), # these currently lead to errors
    # ("hopper", "seals/Hopper-v1"),
    ("swimmer", "seals/Swimmer-v1"),
]


def get_sqil_results(config, named_configs=[]):
    config_str = json.dumps(config)

    signature = str(random.getrandbits(128))
    # Run the separate script
    log_path = os.path.join(LOGDIR, signature) 
    subprocess.run(
        [
            "python",
            "run_train_imitation.py",
            "--config",
            config_str,
            "--log_path",
            log_path,
            "--named_configs",
            json.dumps({"named_configs": named_configs}),
        ],
        check=True,
    )

    # Read the results
    with open(log_path, "r") as f:
        results = json.load(f)

    print(results)
    return results["imit_stats"]["return_mean"]


def objective(trial, environment):
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True
    )  # too broad?
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    gradient_steps = trial.suggest_int("gradient_steps", 1, 20)
    tau = trial.suggest_float("tau", 0.001, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.5, 0.999)

    config_updates = {
        "environment.gym_id": environment,
        "demonstrations.n_expert_demos": 50,
        "sqil.total_timesteps": 100_000,
        "sqil.train_kwargs": {
            "progress_bar": True,
            "log_interval": 100,
        },
        "rl.rl_kwargs": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_steps": gradient_steps,
            "tau": tau,
            "gamma": gamma,
            "learning_starts": 1000,
        },
        "expert.loader_kwargs": {
            "organization": "ernestum",
        },
    }

    named_configs = ["rl.sac", "policy.sac256"]

    return get_sqil_results(config_updates, named_configs)


def perform_trial(study_name, environment, n_trials, n_jobs):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )

    trial_objective = functools.partial(objective, environment=environment)

    study.optimize(trial_objective, n_trials=n_trials, n_jobs=n_jobs)

    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")

def show_study(study_name):
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    print("-" * 80)
    print("Study name: ", study_name)
    print(f"Best params: {study.best_params}")
    print(f"Best value: {study.best_value}")
    print(f"Num trials: {len(study.trials)}")



if __name__ == "__main__":
    n_trials, n_jobs = 40, 10

    for name, env in environments:
        # perform_trial(f"{name}_final", env, n_trials, n_jobs)
        show_study(f"{name}_final")
