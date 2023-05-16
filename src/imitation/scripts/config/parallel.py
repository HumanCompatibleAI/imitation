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
from torch import nn

from imitation.algorithms import dagger
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


@parallel_ex.named_config
def example_rl():
    sacred_ex_name = "train_rl"
    run_name = "rl_tuning"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {"environment": {"num_vec": 1}}
    search_space = {
        "config_updates": {
            "rl": {
                "batch_size": tune.choice([512, 1024, 2048, 4096, 8192]),
                "rl_kwargs": {
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                    "batch_size": tune.choice([64, 128, 256, 512]),
                    "n_epochs": tune.choice([5, 10, 20]),
                },
            },
        },
    }
    num_samples = 100
    eval_best_trial = True
    eval_trial_seeds = 5
    repeat = 1
    resources_per_trial = dict(cpu=1)


@parallel_ex.named_config
def example_bc():
    sacred_ex_name = "train_imitation"
    run_name = "bc_tuning"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {
        "environment": {"num_vec": 1},
        "demonstrations": {"rollout_type": "ppo-huggingface"},
    }
    search_space = {
        "config_updates": {
            "bc": dict(
                batch_size=tune.choice([8, 16, 32, 64]),
                l2_weight=tune.loguniform(1e-6, 1e-2),  # L2 regularization weight
                optimizer_kwargs=dict(
                    lr=tune.loguniform(1e-5, 1e-2),
                ),
                train_kwargs=dict(
                    n_epochs=tune.choice([1, 5, 10, 20]),
                ),
            ),
        },
        "command_name": "bc",
    }
    num_samples = 64
    eval_best_trial = True
    eval_trial_seeds = 5
    repeat = 3
    resources_per_trial = dict(cpu=1)


@parallel_ex.named_config
def example_dagger():
    sacred_ex_name = "train_imitation"
    run_name = "dagger_tuning"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {
        "environment": {"num_vec": 1},
        "dagger": {"total_timesteps": 1e5},
        "bc_kwargs": {
            "batch_size": 16,
            "l2_weight": 1e-4,
            "optimizer_kwargs": {"lr": 1e-3},
        },
    }
    search_space = {
        "config_updates": {
            "bc_train_kwargs": dict(
                n_epochs=tune.choice([1, 5, 10]),
            ),
            "dagger": dict(
                beta_schedule=tune.choice(
                    [dagger.LinearBetaSchedule(i) for i in [1, 5, 15]]
                    + [dagger.ExponentialBetaSchedule(i) for i in [0.3, 0.5, 0.7]],
                ),
                rollout_round_min_episodes=tune.choice([3, 5, 10]),
            ),
        },
        "command_name": "dagger",
    }
    num_samples = 50
    repeat = 3
    eval_best_trial = True
    eval_trial_seeds = 5
    resources_per_trial = dict(cpu=1)


@parallel_ex.named_config
def example_gail():
    sacred_ex_name = "train_adversarial"
    run_name = "gail_tuning_hc"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {
        "environment": {"num_vec": 1},
        "total_timesteps": 1e7,
    }
    search_space = {
        "config_updates": {
            "algorithm_kwargs": dict(
                demo_batch_size=tune.choice([32, 128, 512, 2048, 8192]),
                n_disc_updates_per_round=tune.choice([8, 16]),
            ),
            "rl": {
                "batch_size": tune.choice([4096, 8192, 16384]),
                "rl_kwargs": {
                    "ent_coef": tune.loguniform(1e-7, 1e-3),
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                },
            },
            "algorithm_specific": {},
        },
        "command_name": "gail",
    }
    num_samples = 100
    eval_best_trial = True
    eval_trial_seeds = 5
    repeat = 3
    resources_per_trial = dict(cpu=1)


@parallel_ex.named_config
def example_airl():
    sacred_ex_name = "train_adversarial"
    run_name = "airl_tuning"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {
        "environment": {"num_vec": 1},
        "total_timesteps": 1e7,
    }
    search_space = {
        "config_updates": {
            "algorithm_kwargs": dict(
                demo_batch_size=tune.choice([32, 128, 512, 2048, 8192]),
                n_disc_updates_per_round=tune.choice([8, 16]),
            ),
            "rl": {
                "batch_size": tune.choice([4096, 8192, 16384]),
                "rl_kwargs": {
                    "ent_coef": tune.loguniform(1e-7, 1e-3),
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                },
            },
            "algorithm_specific": {},
        },
        "command_name": "airl",
    }
    num_samples = 100
    eval_best_trial = True
    eval_trial_seeds = 5
    repeat = 3
    resources_per_trial = dict(cpu=1)


@parallel_ex.named_config
def example_pc():
    sacred_ex_name = "train_preference_comparisons"
    run_name = "pc_tuning"
    base_named_configs = ["logging.wandb_logging"]
    base_config_updates = {
        "environment": {"num_vec": 1},
        "total_timesteps": 2e7,
        "total_comparisons": 5000,
        "query_schedule": "hyperbolic",
        "gatherer_kwargs": {"sample": True},
    }
    search_space = {
        "named_configs": tune.choice(
            [
                ["reward.normalize_output_disable"],
            ],
        ),
        "config_updates": {
            "train": {
                "policy_kwargs": {
                    "activation_fn": tune.choice(
                        [
                            nn.ReLU,
                        ],
                    ),
                },
            },
            "num_iterations": tune.choice([25, 50]),
            "initial_comparison_frac": tune.choice([0.1, 0.25]),
            "reward_trainer_kwargs": {
                "epochs": tune.choice([1, 3, 6]),
            },
            "rl": {
                "batch_size": tune.choice([512, 2048, 8192]),
                "rl_kwargs": {
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                    "ent_coef": tune.loguniform(1e-7, 1e-3),
                },
            },
        },
    }
    num_samples = 100
    eval_best_trial = True
    eval_trial_seeds = 5
    repeat = 3
    resources_per_trial = dict(cpu=1)
