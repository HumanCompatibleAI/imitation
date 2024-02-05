"""Script to re-run the best trials from a previous hyperparameter tuning run."""
import argparse
import random
from typing import List, Tuple

import optuna
import sacred

import hp_search_spaces


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=
        "Re-run the best trials from a previous tuning run.",
        epilog=f"Example usage:\n"
               f"python rerun_best_trials.py tuning_run.json\n",
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
        "--top-k",
        type=int,
        default=1,
        help="Chooses the kth best trial to re-run."
    )
    parser.add_argument(
        "journal_log",
        type=str,
        help="The optuna journal file of the previous tuning run."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="The seed to use for the re-run. A random seed is used by default."
    )
    return parser


def infer_algo_name(study: optuna.Study) -> Tuple[str, List[str]]:
    """Infer the algo name from the study name.

    Assumes that the study name is of the form "tuning_{algo}_with_{named_configs}".
    """
    assert study.study_name.startswith("tuning_")
    assert "_with_" in study.study_name
    return study.study_name[len("tuning_"):].split("_with_")[0]


def get_top_k_trial(study: optuna.Study, k: int) -> optuna.trial.Trial:
    if k <= 0:
        raise ValueError(f"--top-k must be positive, but is {k}.")
    finished_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(finished_trials) == 0:
        raise ValueError("No trials have completed.")
    if len(finished_trials) < k:
        raise ValueError(
            f"Only {len(finished_trials)} trials have completed, but --top-k is {k}."
        )

    return sorted(
        finished_trials,
        key=lambda t: t.value, reverse=True,
    )[k-1]


def main():
    parser = make_parser()
    args = parser.parse_args()
    study: optuna.Study = optuna.load_study(
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(args.journal_log)
        ),
        # in our case, we have one journal file per study so the study name can be
        # inferred
        study_name=None,
    )
    trial = get_top_k_trial(study, args.top_k)

    print(trial.value, trial.params)

    algo_name = args.algo or infer_algo_name(study)
    sacred_experiment: sacred.Experiment = hp_search_spaces.objectives_by_algo[algo_name].sacred_ex

    config_updates = trial.user_attrs["config_updates"].copy()
    config_updates["seed"] = args.seed
    result = sacred_experiment.run(
        config_updates=config_updates,
        named_configs=trial.user_attrs["named_configs"],
        options={"--name": study.study_name, "--file_storage": "sacred"},

    )
    if result.status != "COMPLETED":
        raise RuntimeError(
            f"Trial failed with {result.fail_trace()} and status {result.status}."
        )


if __name__ == '__main__':
    main()
