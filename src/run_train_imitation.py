import json
import argparse
import os
from typing import Dict
from imitation.util import util
from imitation.scripts import train_imitation 

TEST_DATA_PATH = util.parse_path("../tests/testdata")
CARTPOLE_TEST_DATA_PATH = TEST_DATA_PATH / "expert_models/cartpole_0/"
CARTPOLE_TEST_POLICY_PATH = CARTPOLE_TEST_DATA_PATH / "policies/final"
CARTPOLE_TEST_ROLLOUT_PATH = CARTPOLE_TEST_DATA_PATH / "rollouts/final.npz"
LOGDIR = "hp_sweep_logs"

ALGO_FAST_CONFIGS = {
    "adversarial": [
        "environment.fast",
        "demonstrations.fast",
        "rl.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "eval_policy": ["environment.fast", "fast"],
    "imitation": [
        "environment.fast",
        "demonstrations.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "preference_comparison": [
        "environment.fast",
        "rl.fast",
        "policy_evaluation.fast",
        "fast",
    ],
    "rl": ["environment.fast", "rl.fast", "fast"],
}

def generate_imitation_config(command_name: str) -> Dict:
    environment_named_config = "seals_cartpole"

    expert_config = dict(
        policy_type="ppo",
        loader_kwargs=dict(path=CARTPOLE_TEST_POLICY_PATH / "model.zip"),
    )
    # Note: we don't have a seals_cartpole expert in our testdata folder,
    # so we use the cartpole environment in this case.
    environment_named_config = "cartpole"

    return dict(
        command_name=command_name,
        named_configs=[environment_named_config] + ALGO_FAST_CONFIGS["imitation"], # added for iteration
        config_updates=dict(
            expert=expert_config,
            demonstrations=dict(path=CARTPOLE_TEST_ROLLOUT_PATH, n_expert_demos=10),
        ),
    )

def sqil_config():
    imitation_config = generate_imitation_config("sqil")

    return imitation_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    args = parser.parse_args()

    config = sqil_config()

    run = train_imitation.train_imitation_ex.run(**config)
    print(run.info)
    with open(args.log_path, 'w') as f:
        json.dump(run.info, f)

if __name__ == '__main__':
    main()

