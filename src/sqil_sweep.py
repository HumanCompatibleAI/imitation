import json
import os
import random
import subprocess
from typing import Dict
import optuna
from imitation.util import networks, util
from run_train_imitation import LOGDIR


def objective(trial):
    # TODO: Add more hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 3e-6, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])



    # Convert config to CLI arguments or a temporary file
    config_str = json.dumps(dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
    ))

    signature = str(random.getrandbits(128))
    # Run the separate script
    log_path = os.path.join(LOGDIR, signature)  # Define your log directory here
    subprocess.run(["python", "run_train_imitation.py", "--config", config_str, "--log_path", log_path], check=True)

    # Read the results
    with open(log_path, 'r') as f:
        results = json.load(f)

    return results["imit_stats"]["return_mean"]

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=1)

# print(f"Best params: {study.best_params}")
# print(f"Best value: {study.best_value}")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
