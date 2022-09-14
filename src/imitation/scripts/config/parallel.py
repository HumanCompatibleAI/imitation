"""Config files for parallel experiments.

Parallel experiments are intended to be defined in Python rather than
via CLI. For example, a user should add a new
`@parallel_ex.named_config` to define a new parallel experiment.

Adding custom named configs is necessary because the CLI interface can't add
search spaces to the config like `"seed": tune.grid_search([0, 1, 2, 3])`.
"""

import numpy as np
import ray.tune as tune
import sacred

from imitation.algorithms.dagger import ExponentialBetaSchedule, LinearBetaSchedule
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
    n_seeds = 3  # Number of seeds to search over by default
    num_samples = 1  # Number of samples per grid search configuration


@parallel_ex.config
def seeds(n_seeds):
    search_space = {"config_updates": {"seed": tune.grid_search(list(range(n_seeds)))}}


@parallel_ex.named_config
def s3():
    upload_dir = "s3://shwang-chai/private"


# Debug named configs


@parallel_ex.named_config
def generate_test_data():
    """Used by tests/generate_test_data.sh to generate tests/testdata/gather_tb/.

    "tests/testdata/gather_tb/" should contain 2 Tensorboard run directories,
    one for each of the trials in the search space below.
    """
    sacred_ex_name = "train_rl"
    run_name = "TEST"
    n_seeds = 1
    search_space = {
        "config_updates": {
            "rl": {
                "rl_kwargs": {
                    "learning_rate": tune.grid_search(
                        [3e-4 * x for x in (1 / 3, 1 / 2)],
                    ),
                },
            },
        },
    }
    base_named_configs = ["cartpole", "common.fast", "train.fast", "rl.fast", "fast"]
    base_config_updates = {
        "rollout_save_final": False,
    }


@parallel_ex.named_config
def example_cartpole_rl():
    sacred_ex_name = "train_rl"
    run_name = "example-cartpole"
    n_seeds = 2
    search_space = {
        "config_updates": {
            "rl": {
                "rl_kwargs": {
                    "learning_rate": tune.grid_search(np.logspace(3e-6, 1e-1, num=3)),
                    "nminibatches": tune.grid_search([16, 32, 64]),
                },
            },
        },
    }
    base_named_configs = ["cartpole"]
    resources_per_trial = dict(cpu=4)


EASY_ENVS = ["cartpole", "pendulum", "mountain_car"]


@parallel_ex.named_config
def example_rl_easy():
    sacred_ex_name = "train_rl"
    run_name = "example-rl-easy"
    n_seeds = 2
    search_space = {
        "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
        "config_updates": {
            "rl": {
                "rl_kwargs": {
                    "learning_rate": tune.grid_search(np.logspace(3e-6, 1e-1, num=3)),
                    "nminibatches": tune.grid_search([16, 32, 64]),
                },
            },
        },
    }
    resources_per_trial = dict(cpu=4)


@parallel_ex.named_config
def example_gail_easy():
    sacred_ex_name = "train_adversarial"
    run_name = "example-gail-easy"
    n_seeds = 1
    search_space = {
        "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
        "config_updates": {
            "init_trainer_kwargs": {
                "rl": {
                    "rl_kwargs": {
                        "learning_rate": tune.grid_search(
                            np.logspace(3e-6, 1e-1, num=3),
                        ),
                        "nminibatches": tune.grid_search([16, 32, 64]),
                    },
                },
            },
        },
    }
    search_space = {
        "command_name": "gail",
    }


MY_ENVS = ["seals_ant", "seals_half_cheetah"]


@parallel_ex.named_config
def example_bc():
    sacred_ex_name = "train_imitation"
    run_name = "bc_tuning_seals_half_cheetah"
    n_seeds = 5
    base_named_configs = ["common.wandb_logging", "seals_half_cheetah"]
    base_config_updates = {
        "common": {"wandb": {"wandb_kwargs": {"project": "algorithm-benchmark"}}},
    }
    search_space = {
        "config_updates": {
            "bc_kwargs": dict(
                batch_size=tune.grid_search([16, 32, 64]),
                l2_weight=tune.grid_search([1e-4, 0]),  # L2 regularization weight
                optimizer_kwargs=dict(
                    lr=tune.grid_search([1e-3, 1e-4]),
                ),
            ),
            "bc_train_kwargs": dict(
                n_epochs=tune.grid_search([1, 4, 7]),
            ),
        },
        "command_name": "bc",
    }
    resources_per_trial = dict(cpu=2)


@parallel_ex.named_config
def example_dagger():
    sacred_ex_name = "train_imitation"
    run_name = "dagger_tuning"
    n_seeds = 5
    base_named_configs = ["common.wandb_logging"]
    base_config_updates = {
        "common": {"wandb": {"wandb_kwargs": {"project": "algorithm-benchmark"}}},
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
                n_epochs=tune.grid_search([4, 7, 10]),
            ),
            "dagger": dict(
                beta_schedule=tune.grid_search(
                    [LinearBetaSchedule(i) for i in [1, 5, 15]]
                    + [ExponentialBetaSchedule(i) for i in [0.3, 0.5, 0.7]],
                ),
                rollout_round_min_episodes=tune.grid_search([3, 5, 10]),
            ),
        },
        "command_name": "dagger",
    }
    resources_per_trial = dict(cpu=2, gpu=0.1)


@parallel_ex.named_config
def example_gail():
    sacred_ex_name = "train_adversarial"
    run_name = "gail-tuning"
    n_seeds = 5
    base_named_configs = ["common.wandb_logging"]
    base_config_updates = {
        "common": {"wandb": {"wandb_kwargs": {"project": "algorithm-benchmark"}}},
        "total_timesteps": 1e5,
    }
    search_space = {
        "named_configs": tune.grid_search([[env] for env in EASY_ENVS]),
        "config_updates": {
            "algorithm_kwargs": dict(
                demo_batch_size=tune.choice([512, 1024, 2048]),
                n_disc_updates_per_round=tune.choice([2, 4, 8]),
                gen_replay_buffer_capacity=tune.choice([512, 1024]),
                # gen_train_timesteps=0,
            ),
            "rl": {
                "batch_size": tune.choice([512, 1024, 2048]),
                "rl_kwargs": {"ent_coef": tune.loguniform(1e-18, 1e-3)},
            },
            "algorithm_specific": {},
        },
    }
    search_space = {
        "command_name": "gail",
    }
    resources_per_trial = dict(cpu=2)
