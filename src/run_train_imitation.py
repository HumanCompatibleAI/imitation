import json
import argparse
import os
from typing import Dict, List, Optional
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

def generate_imitation_config(command_name: str, environment_named_config) -> Dict:
    expert_config = dict(
        policy_type="ppo",
        loader_kwargs=dict(path=CARTPOLE_TEST_POLICY_PATH / "model.zip"),
    )
    # Note: we don't have a seals_cartpole expert in our testdata folder,
    # so we use the cartpole environment in this case.
    # environment_named_config = "cartpole"

    return dict(
        command_name=command_name,
        named_configs=[],
        config_updates=dict(
            # expert=expert_config,
            # demonstrations=dict(path=CARTPOLE_TEST_ROLLOUT_PATH, n_expert_demos=10),
        ),
    )

def sqil_config(env_name):
    imitation_config = generate_imitation_config("sqil", env_name)

    return imitation_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--env_name', type=Optional[str], default=None)
    parser.add_argument('--named_configs', type=str, default={"named_configs": []})
    args = parser.parse_args()

    config = sqil_config(args.env_name)

    config["config_updates"].update(json.loads(args.config))
    named_configs = json.loads(args.named_configs)
    config["named_configs"] += named_configs["named_configs"]
    print("Config lajsdfklasjfd: ", config)

    run = train_imitation.train_imitation_ex.run(**config)
    print(run.result)
    with open(args.log_path, 'w') as f:
        json.dump(run.result, f)

if __name__ == '__main__':
    main()

