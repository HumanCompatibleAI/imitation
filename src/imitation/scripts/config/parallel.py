"""Config files for parallel experiments.

Parallel experiments are intended to be defined in Python rather than
via CLI. For example, a user should add a new
`@parallel_ex.named_config` to define a new parallel experiment.

Adding custom named configs is necessary because the CLI interface can't add
search spaces to the config like `"seed": tune.choice([0, 1, 2, 3])`.

For tuning hyperparameters of an algorithm on a given environment, override
the `base_named_configs` argument with the named config of the environment.
Ex: python -m imitation.scripts.parallel with example_gail \
    'base_named_configs=["logging.wandb_logging", "seals_half_cheetah"]'
"""

import numpy as np
import ray.tune as tune
import sacred

from imitation.util.util import make_unique_timestamp

parallel_ex = sacred.Experiment("parallel")


@parallel_ex.config
def config():
    sacred_ex_name = "train_rl"  # The experiment to parallelize
    init_kwargs = {}  # Keyword arguments to pass to ray.init()
    _uuid = make_unique_timestamp()
    run_name = f"DEFAULT_{_uuid}"  # CLI --name option. For analysis grouping.
    resources_per_trial = {}  # Argument to `tune.run`
    base_named_configs = []  # Background settings before search_space is applied
    base_config_updates = {}  # Background settings before search_space is applied
    search_space = {
        "named_configs": [],
        "config_updates": {},
    }  # `config` argument to `ray.tune.run(trainable, config)`

    local_dir = None  # `local_dir` arg for `ray.tune.run`
    upload_dir = None  # `upload_dir` arg for `ray.tune.run`
    experiment_checkpoint_path = ""
    eval_best_trial = False
    eval_trial_seeds = 5  # Number of seeds to search over by default
    num_samples = 1  # Number of samples per grid search configuration
    repeat = 1
    search_alg = "optuna"  # search algorithm to use
    experiment_checkpoint_path = ""  # Path to checkpoint of experiment to resume
    syncer = None  # Sacred syncer to use
    resume = False  # Whether to resume experiment from checkpoint


# Debug named configs


@parallel_ex.named_config
def generate_test_data():
    """Used by tests/generate_test_data.sh to generate tests/testdata/gather_tb/.

    "tests/testdata/gather_tb/" should contain 2 Tensorboard run directories,
    one for each of the trials in the search space below.
    """
    sacred_ex_name = "train_rl"
    run_name = "TEST"
    repeat = 1
    search_space = {
        "config_updates": {
            "rl": {
                "rl_kwargs": {
                    "learning_rate": tune.choice(
                        [3e-4 * x for x in (1 / 3, 1 / 2)],
                    ),
                },
            },
        },
    }
    base_named_configs = [
        "cartpole",
        "environment.fast",
        "policy_evaluation.fast",
        "rl.fast",
        "fast",
    ]
    base_config_updates = {
        "rollout_save_final": False,
    }


@parallel_ex.named_config
def example_cartpole_rl():
    sacred_ex_name = "train_rl"
    run_name = "example-cartpole"
    repeat = 2
    search_space = {
        "config_updates": {
            "rl": {
                "rl_kwargs": {
                    "learning_rate": tune.choice(np.logspace(3e-6, 1e-1, num=3)),
                    "nminibatches": tune.choice([16, 32, 64]),
                },
            },
        },
    }
    base_named_configs = ["cartpole"]
    resources_per_trial = dict(cpu=4)
