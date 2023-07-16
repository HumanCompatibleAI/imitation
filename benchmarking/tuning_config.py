"""Config files for tuning experiments."""

import ray.tune as tune
import sacred
from torch import nn

from imitation.algorithms import dagger
from imitation.scripts.parallel import parallel_ex

tuning_ex = sacred.Experiment("tuning", ingredients=[parallel_ex])


@tuning_ex.named_config
def example_rl():
    parallel_run_config = dict(
        sacred_ex_name="train_rl",
        run_name="rl_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={"environment": {"num_vec": 1}},
        search_space={
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
        },
        num_samples=100,
        repeat=1,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5


@tuning_ex.named_config
def example_bc():
    parallel_run_config = dict(
        sacred_ex_name="train_imitation",
        run_name="bc_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
        },
        search_space={
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
        },
        num_samples=64,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )

    num_eval_seeds = 5
    eval_best_trial_resource_multiplier = 1


@tuning_ex.named_config
def example_dagger():
    parallel_run_config = dict(
        sacred_ex_name="train_imitation",
        run_name="dagger_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
            "dagger": {"total_timesteps": 1e5},
            "bc": {
                "batch_size": 16,
                "l2_weight": 1e-4,
                "optimizer_kwargs": {"lr": 1e-3},
            },
        },
        search_space={
            "config_updates": {
                "bc": dict(
                    train_kwargs=dict(
                        n_epochs=tune.choice([1, 5, 10]),
                    ),
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
        },
        num_samples=50,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5


@tuning_ex.named_config
def example_gail():
    parallel_run_config = dict(
        sacred_ex_name="train_adversarial",
        run_name="gail_tuning_hc",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
            "total_timesteps": 1e7,
        },
        search_space={
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
        },
        num_samples=100,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5


@tuning_ex.named_config
def example_airl():
    parallel_run_config = dict(
        sacred_ex_name="train_adversarial",
        run_name="airl_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
            "total_timesteps": 1e7,
        },
        search_space={
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
        },
        num_samples=100,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5


@tuning_ex.named_config
def example_pc():
    parallel_run_config = dict(
        sacred_ex_name="train_preference_comparisons",
        run_name="pc_tuning",
        base_named_configs=["logging.wandb_logging"],
        base_config_updates={
            "environment": {"num_vec": 1},
            "demonstrations": {"source": "huggingface"},
            "total_timesteps": 2e7,
            "total_comparisons": 5000,
            "query_schedule": "hyperbolic",
            "gatherer_kwargs": {"sample": True},
        },
        search_space={
            "named_configs": [
                ["reward.normalize_output_disable"],
            ],
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
        },
        num_samples=100,
        repeat=3,
        resources_per_trial=dict(cpu=1),
    )
    num_eval_seeds = 5
