"""Config files for tuning experiments."""

import ray.tune as tune
import sacred

from imitation.algorithms import dagger as dagger_alg
from imitation.scripts.parallel import parallel_ex

tuning_ex = sacred.Experiment("tuning", ingredients=[parallel_ex])


@tuning_ex.named_config
def rl():
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
def bc():
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
def dagger():
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
                        [dagger_alg.LinearBetaSchedule(i) for i in [1, 5, 15]]
                        + [
                            dagger_alg.ExponentialBetaSchedule(i)
                            for i in [0.3, 0.5, 0.7]
                        ],
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
def gail():
    parallel_run_config = dict(
        sacred_ex_name="train_adversarial",
        run_name="gail_tuning",
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
def airl():
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
def pc():
    parallel_run_config = dict(
        sacred_ex_name="train_preference_comparisons",
        run_name="pc_tuning",
        base_named_configs=[],
        base_config_updates={
            "environment": {"num_vec": 1},
            "total_timesteps": 2e7,
            "total_comparisons": 1000,
            "active_selection": True,
        },
        search_space={
            "named_configs": ["reward.reward_ensemble"],
            "config_updates": {
                "active_selection_oversampling": tune.randint(1, 11),
                "comparison_queue_size": tune.randint(
                    1, 1001,
                ),  # upper bound determined by total_comparisons=1000
                "exploration_frac": tune.uniform(0.0, 0.5),
                "fragment_length": tune.randint(
                    1, 1001,
                ),  # trajectories are 1000 steps long
                "gatherer_kwargs": {
                    "temperature": tune.uniform(0.0, 2.0),
                    "discount_factor": tune.uniform(0.95, 1.0),
                    "sample": tune.choice([True, False]),
                },
                "initial_comparison_frac": tune.uniform(0.01, 1.0),
                "num_iterations": tune.randint(1, 51),
                "preference_model_kwargs": {
                    "noise_prob": tune.uniform(0.0, 0.1),
                    "discount_factor": tune.uniform(0.95, 1.0),
                },
                "query_schedule": tune.choice(
                    ["hyperbolic", "constant", "inverse_quadratic",]
                ),
                "trajectory_generator_kwargs": {
                    "switch_prob": tune.uniform(0.1, 1),
                    "random_prob": tune.uniform(0.1, 0.9),
                },
                "transition_oversampling": tune.uniform(0.9, 2.0),
                "reward_trainer_kwargs": {
                    "epochs": tune.randint(1, 11),
                },
                "rl": {
                    "rl_kwargs": {
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
