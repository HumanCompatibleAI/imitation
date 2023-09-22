"""Train a reward model using preference comparisons.

Can be used as a CLI script, or the `train_preference_comparisons` function
can be called directly.
"""

import functools
import pathlib
from typing import Any, Mapping, Optional, Type, Union

import numpy as np
import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import type_aliases

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
from imitation.algorithms import preference_comparisons
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward
from imitation.scripts.ingredients import rl as rl_common
from imitation.util import video_wrapper


def save_model(
    agent_trainer: preference_comparisons.AgentTrainer,
    save_path: pathlib.Path,
):
    """Save the model as `model.zip`."""
    policies_serialize.save_stable_model(
        output_dir=save_path / "policy",
        model=agent_trainer.algorithm,
    )


def save_checkpoint(
    trainer: preference_comparisons.PreferenceComparisons,
    save_path: pathlib.Path,
    allow_save_policy: Optional[bool],
):
    """Save reward model and optionally policy."""
    save_path.mkdir(parents=True, exist_ok=True)
    th.save(trainer.model, save_path / "reward_net.pt")
    if allow_save_policy:
        # Note: We should only save the model as model.zip if `trajectory_generator`
        # contains one. Currently we are slightly over-conservative, by requiring
        # that an AgentTrainer be used if we're saving the policy.
        assert isinstance(
            trainer.trajectory_generator,
            preference_comparisons.AgentTrainer,
        )
        save_model(trainer.trajectory_generator, save_path)
    else:
        trainer.logger.warn(
            "trainer.trajectory_generator doesn't contain a policy to save.",
        )


@train_preference_comparisons_ex.main
def train_preference_comparisons(
    total_timesteps: int,
    total_comparisons: int,
    num_iterations: int,
    comparison_queue_size: Optional[int],
    fragment_length: int,
    transition_oversampling: float,
    initial_comparison_frac: float,
    exploration_frac: float,
    trajectory_path: Optional[str],
    trajectory_generator_kwargs: Mapping[str, Any],
    save_preferences: bool,
    agent_path: Optional[str],
    preference_model_kwargs: Mapping[str, Any],
    reward_trainer_kwargs: Mapping[str, Any],
    gatherer_cls: Type[preference_comparisons.PreferenceGatherer],
    gatherer_kwargs: Mapping[str, Any],
    active_selection: bool,
    active_selection_oversampling: int,
    uncertainty_on: str,
    fragmenter_kwargs: Mapping[str, Any],
    allow_variable_horizon: bool,
    checkpoint_interval: int,
    query_schedule: Union[str, type_aliases.Schedule],
    video_log_dir: Optional[str],
    _rnd: np.random.Generator,
) -> Mapping[str, Any]:
    """Train a reward model using preference comparisons.

    Args:
        total_timesteps: number of environment interaction steps
        total_comparisons: number of preferences to gather in total
        num_iterations: number of times to train the agent against the reward model
            and then train the reward model against newly gathered preferences.
        comparison_queue_size: the maximum number of comparisons to keep in the
            queue for training the reward model. If None, the queue will grow
            without bound as new comparisons are added.
        fragment_length: number of timesteps per fragment that is used to elicit
            preferences
        transition_oversampling: factor by which to oversample transitions before
            creating fragments. Since fragments are sampled with replacement,
            this is usually chosen > 1 to avoid having the same transition
            in too many fragments.
        initial_comparison_frac: fraction of total_comparisons that will be
            sampled before the rest of training begins (using the randomly initialized
            agent). This can be used to pretrain the reward model before the agent
            is trained on the learned reward.
        exploration_frac: fraction of trajectory samples that will be created using
            partially random actions, rather than the current policy. Might be helpful
            if the learned policy explores too little and gets stuck with a wrong
            reward.
        trajectory_path: either None, in which case an agent will be trained
            and used to sample trajectories on the fly, or a path to a pickled
            sequence of TrajectoryWithRew to be trained on.
        trajectory_generator_kwargs: kwargs to pass to the trajectory generator.
        save_preferences: if True, store the final dataset of preferences to disk.
        agent_path: if given, initialize the agent using this stored policy
            rather than randomly.
        preference_model_kwargs: passed to PreferenceModel
        reward_trainer_kwargs: passed to BasicRewardTrainer or EnsembleRewardTrainer
        gatherer_cls: type of PreferenceGatherer to use (defaults to SyntheticGatherer)
        gatherer_kwargs: passed to the PreferenceGatherer specified by gatherer_cls
        active_selection: use active selection fragmenter instead of random fragmenter
        active_selection_oversampling: factor by which to oversample random fragments
            from the base fragmenter of active selection.
            this is usually chosen > 1 to allow the active selection algorithm to pick
            fragment pairs with highest uncertainty. = 1 implies no active selection.
        uncertainty_on: passed to ActiveSelectionFragmenter
        fragmenter_kwargs: passed to RandomFragmenter
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.
        checkpoint_interval: Save the reward model and policy models (if
            trajectory_generator contains a policy) every `checkpoint_interval`
            iterations and after training is complete. If 0, then only save weights
            after training is complete. If <0, then don't save weights at all.
        query_schedule: one of ("constant", "hyperbolic", "inverse_quadratic").
            A function indicating how the total number of preference queries should
            be allocated to each iteration. "hyperbolic" and "inverse_quadratic"
            apportion fewer queries to later iterations when the policy is assumed
            to be better and more stable.
        video_log_dir: If set, save videos to this directory.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Rollout statistics from trained policy.

    Raises:
        ValueError: Inconsistency between config and deserialized policy normalization.
    """
    # This allows to specify total_timesteps, total_comparisons etc. in scientific
    # notation, which is interpreted as a float by python.
    total_timesteps = int(total_timesteps)
    total_comparisons = int(total_comparisons)
    num_iterations = int(num_iterations)
    comparison_queue_size = (
        int(comparison_queue_size) if comparison_queue_size is not None else None
    )
    fragment_length = int(fragment_length)
    active_selection_oversampling = int(active_selection_oversampling)
    checkpoint_interval = int(checkpoint_interval)

    custom_logger, log_dir = logging_ingredient.setup_logging()

    post_wrappers = (
        [
            video_wrapper.video_wrapper_factory(
                pathlib.Path(video_log_dir),
                single_video=False,
            ),
        ]
        if video_log_dir
        else None
    )

    with environment.make_venv(post_wrappers=post_wrappers) as venv:
        reward_net = reward.make_reward_net(venv)
        relabel_reward_fn = functools.partial(
            reward_net.predict_processed,
            update_stats=False,
        )
        if agent_path is None:
            agent = rl_common.make_rl_algo(venv, relabel_reward_fn=relabel_reward_fn)
        else:
            agent = rl_common.load_rl_algo_from_path(
                agent_path=agent_path,
                venv=venv,
                relabel_reward_fn=relabel_reward_fn,
            )

        if trajectory_path is None:
            # Setting the logger here is not necessary (PreferenceComparisons takes care
            # of it automatically) but it avoids creating unnecessary loggers.
            agent_trainer = preference_comparisons.AgentTrainer(
                algorithm=agent,
                reward_fn=reward_net,
                venv=venv,
                exploration_frac=exploration_frac,
                rng=_rnd,
                custom_logger=custom_logger,
                **trajectory_generator_kwargs,
            )
            # Stable Baselines will automatically occupy GPU 0 if it is available.
            # Let's use the same device as the SB3 agent for the reward model.
            reward_net = reward_net.to(agent_trainer.algorithm.device)
            trajectory_generator: preference_comparisons.TrajectoryGenerator = (
                agent_trainer
            )
        else:
            if exploration_frac > 0:
                raise ValueError(
                    "exploration_frac can't be set when a trajectory dataset is used",
                )
            trajectory_generator = preference_comparisons.TrajectoryDataset(
                trajectories=data_serialize.load_with_rewards(
                    trajectory_path,
                ),
                rng=_rnd,
                custom_logger=custom_logger,
                **trajectory_generator_kwargs,
            )

        fragmenter: preference_comparisons.Fragmenter = (
            preference_comparisons.RandomFragmenter(
                **fragmenter_kwargs,
                rng=_rnd,
                custom_logger=custom_logger,
            )
        )
        preference_model = preference_comparisons.PreferenceModel(
            **preference_model_kwargs,
            model=reward_net,
        )
        if active_selection:
            fragmenter = preference_comparisons.ActiveSelectionFragmenter(
                preference_model=preference_model,
                base_fragmenter=fragmenter,
                fragment_sample_factor=active_selection_oversampling,
                uncertainty_on=uncertainty_on,
                custom_logger=custom_logger,
            )
        gatherer = gatherer_cls(
            **gatherer_kwargs,
            rng=_rnd,
            custom_logger=custom_logger,
        )

        loss = preference_comparisons.CrossEntropyRewardLoss()

        reward_trainer = preference_comparisons._make_reward_trainer(
            preference_model,
            loss,
            _rnd,
            reward_trainer_kwargs,
        )

        main_trainer = preference_comparisons.PreferenceComparisons(
            trajectory_generator,
            reward_net,
            num_iterations=num_iterations,
            fragmenter=fragmenter,
            preference_gatherer=gatherer,
            reward_trainer=reward_trainer,
            comparison_queue_size=comparison_queue_size,
            fragment_length=fragment_length,
            transition_oversampling=transition_oversampling,
            initial_comparison_frac=initial_comparison_frac,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
            query_schedule=query_schedule,
        )

        def save_callback(iteration_num):
            if checkpoint_interval > 0 and iteration_num % checkpoint_interval == 0:
                save_checkpoint(
                    trainer=main_trainer,
                    save_path=log_dir / "checkpoints" / f"{iteration_num:04d}",
                    allow_save_policy=bool(trajectory_path is None),
                )

        results = main_trainer.train(
            total_timesteps,
            total_comparisons,
            callback=save_callback,
        )

        # Storing and evaluating policy only useful if we generated trajectory data
        if bool(trajectory_path is None):
            results = dict(results)
            results["rollout"] = policy_evaluation.eval_policy(agent, venv)

    if save_preferences:
        main_trainer.dataset.save(log_dir / "preferences.pkl")

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save_checkpoint(
            trainer=main_trainer,
            save_path=log_dir / "checkpoints" / "final",
            allow_save_policy=bool(trajectory_path is None),
        )

    return results


def main_console():
    observer_path = (
        pathlib.Path.cwd() / "output" / "sacred" / "train_preference_comparisons"
    )
    observer = FileStorageObserver(observer_path)
    train_preference_comparisons_ex.observers.append(observer)
    train_preference_comparisons_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
