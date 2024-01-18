"""Definitions for search spaces used when tuning hyperparameters.

To add a new search space, add a new entry to the `objectives_by_algo` dict.
The key should be the name of the algorithm, and the value should be a RunSacredAsTrial
object that specifies what sacred experiment to run and how to sample hyperparameters.

Note: you could specify multiple search spaces for the same algorithm. Just make sure
to give them different names, and then specify the name when running the tuning script.
For example, to use different spaces for different classes of environments, you could
have a "pc-classic-control" and a "pc-mujoco" search space.
"""

import dataclasses
from typing import Callable, List, Mapping, Any, Dict

import optuna
import sacred

import imitation.scripts.train_preference_comparisons


@dataclasses.dataclass
class RunSacredAsTrial:
    """Runs a sacred experiment as an optuna trial.

    Assumes that the sacred experiment returns a dict with a key 'imit_stats' that
    contains a dict with a key 'monitor_return_mean'.
    """

    """The sacred experiment to run."""
    sacred_ex: sacred.Experiment

    """A function that returns a list of named configs to pass to sacred.run."""
    suggest_named_configs: Callable[[optuna.Trial], List[str]]

    """A function that returns a dict of config updates to pass to sacred.run."""
    suggest_config_updates: Callable[[optuna.Trial], Mapping[str, Any]]

    def __call__(
            self,
            trial: optuna.Trial,
            run_options: Dict,
            extra_named_configs: List[str]
    ) -> float:
        """Run the sacred experiment and return the performance.

        Args:
            trial: The optuna trial to sample hyperparameters for.
            run_options: Options to pass to sacred.run(options=).
            extra_named_configs: Additional named configs to pass to sacred.run.
        """

        config_updates = self.suggest_config_updates(trial)
        named_configs = self.suggest_named_configs(trial) + extra_named_configs

        trial.set_user_attr("config_updates", config_updates)
        trial.set_user_attr("named_configs", named_configs)

        experiment: sacred.Experiment = self.sacred_ex
        result = experiment.run(
            config_updates=config_updates,
            named_configs=named_configs,
            options=run_options,
        )
        if result.status != "COMPLETED":
            raise RuntimeError(
                f"Trial failed with {result.fail_trace()} and status {result.status}."
            )
        return result.result['imit_stats']['monitor_return_mean']


"""A mapping from algorithm names to functions that run the algorithm as an optuna trial."""
objectives_by_algo = dict(
    pc=RunSacredAsTrial(
        sacred_ex=imitation.scripts.train_preference_comparisons.train_preference_comparisons_ex,
        suggest_named_configs=lambda _: ["reward.reward_ensemble"],
        suggest_config_updates=lambda trial: {
            "seed": trial.number,
            "environment": {"num_vec": 8},
            "total_timesteps": 2e7,
            "total_comparisons": 1000,
            "active_selection": True,
            "active_selection_oversampling": trial.suggest_int("active_selection_oversampling", 1, 11),
            "comparison_queue_size": trial.suggest_int("comparison_queue_size", 1, 1001),  # upper bound determined by total_comparisons=1000
            "exploration_frac": trial.suggest_float("exploration_frac", 0.0, 0.5),
            "fragment_length": trial.suggest_int("fragment_length", 1, 1001),  # trajectories are 1000 steps long
            "gatherer_kwargs": {
                "temperature": trial.suggest_float("gatherer_temperature", 0.0, 2.0),
                "discount_factor": trial.suggest_float("gatherer_discount_factor", 0.95, 1.0),
                "sample": trial.suggest_categorical("gatherer_sample", [True, False]),
            },
            "initial_epoch_multiplier": trial.suggest_float("initial_epoch_multiplier", 1, 200.0),
            "initial_comparison_frac": trial.suggest_float("initial_comparison_frac", 0.01, 1.0),
            "num_iterations": trial.suggest_int("num_iterations", 1, 51),
            "preference_model_kwargs": {
                "noise_prob": trial.suggest_float("preference_model_noise_prob", 0.0, 0.1),
                "discount_factor": trial.suggest_float("preference_model_discount_factor", 0.95, 1.0),
            },
            "query_schedule": trial.suggest_categorical("query_schedule", ["hyperbolic", "constant", "inverse_quadratic"]),
            "trajectory_generator_kwargs": {
                "switch_prob": trial.suggest_float("tr_gen_switch_prob", 0.1, 1),
                "random_prob": trial.suggest_float("tr_gen_random_prob", 0.1, 0.9),
            },
            "transition_oversampling": trial.suggest_float("transition_oversampling", 0.9, 2.0),
            "reward_trainer_kwargs": {
                "epochs": trial.suggest_int("reward_trainer_epochs", 1, 11),
            },
            "rl": {
                "rl_kwargs": {
                    "ent_coef": trial.suggest_float("rl_ent_coef", 1e-7, 1e-3, log=True),
                },
            },
        },
    ),
)