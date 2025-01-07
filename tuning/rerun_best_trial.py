"""Script to re-run the best trials from a previous hyperparameter tuning run."""
import argparse
import random

import hp_search_spaces
import optuna
import sacred


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-run the best trial from a previous tuning run.",
        epilog="Example usage:\npython rerun_best_trials.py tuning_run.json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=hp_search_spaces.objectives_by_algo.keys(),
        help="The algorithm that has been tuned. "
        "Can usually be deduced from the study name.",
    )
    parser.add_argument(
        "journal_log",
        type=str,
        help="The optuna journal file of the previous tuning run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="The seed to use for the re-run. A random seed is used by default.",
    )
    return parser


def infer_algo_name(study: optuna.Study) -> str:
    """Infer the algo name from the study name.

    Assumes that the study name is of the form "tuning_{algo}_with_{named_configs}".

    Args:
        study: The optuna study.

    Returns:
        algo name
    """
    assert study.study_name.startswith("tuning_")
    assert "_with_" in study.study_name
    return study.study_name[len("tuning_") :].split("_with_")[0]


def main():
    parser = make_parser()
    args = parser.parse_args()
    study: optuna.Study = optuna.load_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(args.journal_log),
        ),
        # in our case, we have one journal file per study so the study name can be
        # inferred
        study_name=None,
    )
    trial = study.best_trial

    algo_name = args.algo or infer_algo_name(study)
    sacred_experiment: sacred.Experiment = hp_search_spaces.objectives_by_algo[
        algo_name
    ].sacred_ex

    config_updates = trial.user_attrs["config_updates"].copy()
    config_updates["seed"] = args.seed
    result = sacred_experiment.run(
        config_updates=config_updates,
        named_configs=trial.user_attrs["named_configs"],
        options={"--name": study.study_name, "--file_storage": "sacred"},
    )
    if result.status != "COMPLETED":
        raise RuntimeError(
            f"Trial failed with {result.fail_trace()} and status {result.status}.",
        )


if __name__ == "__main__":
    main()
