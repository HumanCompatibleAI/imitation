import argparse
import dataclasses
from typing import List, Mapping, Any, Callable, Dict

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
            "environment": {"num_vec": 1},
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


def make_parser() -> argparse.ArgumentParser:
    example_usage = "python -m imitation.scripts.tune pc seals_swimmer"
    possible_named_configs = "\n".join(
        f"  - {algo}: {', '.join(objective.sacred_ex.named_configs.keys())}"
        for algo, objective in objectives_by_algo.items()
    )

    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for imitation learning algorithms.",
        epilog=f"Example usage:\n{example_usage}\n\nPossible named configs:\n{possible_named_configs}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "algo",
        type=str,
        default="pc",
        choices=objectives_by_algo.keys(),
        help="What algorithm to tune.",
    )
    parser.add_argument(
        "named_configs",
        type=str,
        nargs="+",
        default=[],
        help="Additional named configs to pass to the sacred experiment. "
             "Use this to select the environment to tune on.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Number of trials to run."
    )
    parser.add_argument(
        "-j",
        "--journal-log",
        type=str,
        default=None,
        help="A journal file to synchronize multiple instances of this script. "
             "Works on NFS storage."
    )
    return parser


def make_study(args: argparse.Namespace) -> optuna.Study:
    if args.journal_log is not None:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(args.journal_log)
        )
    else:
        storage = None

    return optuna.create_study(
        study_name=f"tuning_{args.algo}_with_{'_'.join(args.named_configs)}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )


def main():
    parser = make_parser()
    args = parser.parse_args()
    study = make_study(args)

    study.optimize(
        lambda trial: objectives_by_algo[args.algo](
            trial,
            run_options={"--name": study.study_name, "--file_storage": "sacred"},
            extra_named_configs=args.named_configs
        ),
        callbacks=[optuna.study.MaxTrialsCallback(args.num_trials)]
    )


if __name__ == '__main__':
    main()
