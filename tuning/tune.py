"""Script to tune hyperparameters for imitation learning algorithms using optuna."""
import argparse

import optuna

from hp_search_spaces import objectives_by_algo


def make_parser() -> argparse.ArgumentParser:
    example_usage = "python tune.py pc seals_swimmer"
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
