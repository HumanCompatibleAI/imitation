"""Train a reward model using preference comparisons.

Can be used as a CLI script, or the `train_preference_comparisons` function
can be called directly.
"""

import os
import pathlib
import pickle
from typing import Any, Mapping, Optional

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import vec_env

from imitation.algorithms import preference_comparisons
from imitation.policies import serialize
from imitation.rewards import reward_nets
from imitation.scripts.common import common, reward
from imitation.scripts.common import rl as rl_common
from imitation.scripts.common import train, wb
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)


@train_preference_comparisons_ex.main
def train_preference_comparisons(
    _run,
    _seed: int,
    normalize: bool,
    normalize_kwargs: Mapping[str, Any],
    total_timesteps: int,
    total_comparisons: int,
    comparisons_per_iteration: int,
    fragment_length: int,
    transition_oversampling: float,
    trajectory_path: Optional[str],
    save_preferences: bool,
    agent_path: Optional[str],
    reward_trainer_kwargs: Mapping[str, Any],
    gatherer_kwargs: Mapping[str, Any],
    rl: Mapping[str, Any],
    allow_variable_horizon: bool,
) -> Mapping[str, Any]:
    """Train a reward model using preference comparisons.

    Args:
        _seed: Random seed.
        normalize: If True, then rescale observations and reward.
        normalize_kwargs: kwargs for `VecNormalize`.
        total_timesteps: number of environment interaction steps
        total_comparisons: number of preferences to gather in total
        comparisons_per_iteration: number of preferences to gather at once (before
            switching back to agent training). This doesn't impact the total number
            of comparisons that are gathered, only the frequency of switching
            between preference gathering and agent training.
        fragment_length: number of timesteps per fragment that is used to elicit
            preferences
        transition_oversampling: factor by which to oversample transitions before
            creating fragments. Since fragments are sampled with replacement,
            this is usually chosen > 1 to avoid having the same transition
            in too many fragments.
        trajectory_path: either None, in which case an agent will be trained
            and used to sample trajectories on the fly, or a path to a pickled
            sequence of TrajectoryWithRew to be trained on
        save_preferences: if True, store the final dataset of preferences to disk.
        agent_path: if given, initialize the agent using this stored policy
            rather than randomly.
        reward_trainer_kwargs: passed to CrossEntropyRewardTrainer
        gatherer_kwargs: passed to SyntheticGatherer
        rl: parameters for RL training, used for restoring agents.
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.

    Returns:
        Rollout statistics from trained policy.

    Raises:
        FileNotFoundError: Path corresponding to saved policy missing.
        ValueError: Inconsistency between config and deserialized policy normalization.
    """
    custom_logger, log_dir = common.setup_logging()
    custom_logger = wb.setup_wandb_writer(custom_logger=custom_logger, log_dir=log_dir)
    venv = common.make_venv()

    vec_normalize = None
    reward_net = reward.make_reward_net(venv)
    if reward_net is None:
        reward_net = reward_nets.BasicRewardNet(
            venv.observation_space,
            venv.action_space,
        )
    if agent_path is None:
        agent = rl_common.make_rl_algo(venv)
    else:
        # TODO(ejnnr): this is pretty similar to the logic in policies/serialize.py
        # but I did make a few small changes that make it a bit tricky to actually
        # factor this out into a helper function. Still, sharing at least part of the
        # code would probably be good.
        policy_dir = pathlib.Path(agent_path)
        if not policy_dir.is_dir():
            raise FileNotFoundError(
                f"agent_path={agent_path} needs to be a directory containing model.zip "
                "and optionally vec_normalize.pkl.",
            )

        model_path = policy_dir / "model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Could not find policy at expected location {model_path}",
            )

        agent = rl["rl_cls"].load(
            model_path,
            env=venv,
            seed=_seed,
            **rl["rl_kwargs"],
        )
        custom_logger.info(f"Warm starting agent from '{model_path}'")

        normalize_path = policy_dir / "vec_normalize.pkl"
        try:
            with open(normalize_path, "rb") as f:
                vec_normalize = pickle.load(f)
        except FileNotFoundError:
            # We did not use VecNormalize during training, skip
            pass
        else:
            if not normalize:
                raise ValueError(
                    "normalize=False but the loaded policy has "
                    "associated normalization stats.",
                )
            # TODO(ejnnr): this check is hacky, what if we change the default config?
            if normalize_kwargs != {"norm_reward": False}:
                # We could adjust settings manually but that's very brittle
                # if SB3 changes any of the VecNormalize internals
                print(normalize_kwargs)
                raise ValueError(
                    "Setting normalize_kwargs is not supported "
                    "when loading an existing VecNormalize.",
                )
            vec_normalize.training = True
            # TODO(ejnnr): We should figure out at some point if reward normalization
            # is useful for preference comparisons but I haven't tried it yet. We'd also
            # have to decide where to normalize rewards; setting norm_reward=True here
            # would normalize the rewards that the reward model sees. This would
            # probably translate to some degree to its output (i.e. the rewards for
            # the agent). Alternatively, we could just train the reward model on
            # unnormalized rewards and then normalize its output before giving it
            # to the agent (which would also work for human feedback).
            vec_normalize.norm_reward = False
            vec_normalize.set_venv(venv)
            # Note: the following line must come after the previous set_venv line!
            # Otherwise, we get recursion errors
            venv = vec_normalize
            agent.set_env(venv)
            custom_logger.info(f"Loaded VecNormalize from '{normalize_path}'")

    if normalize and vec_normalize is None:
        # if no stats have been loaded, create a new VecNormalize wrapper
        venv = vec_normalize = vec_env.VecNormalize(venv, **normalize_kwargs)
        agent.set_env(venv)

    if trajectory_path is None:
        # Setting the logger here is not really necessary (PreferenceComparisons
        # takes care of that automatically) but it avoids creating unnecessary loggers
        trajectory_generator = preference_comparisons.AgentTrainer(
            agent,
            reward_net,
            custom_logger=custom_logger,
        )
    else:
        trajectory_generator = preference_comparisons.TrajectoryDataset(
            trajectory_path,
            _seed,
            custom_logger,
        )

    fragmenter = preference_comparisons.RandomFragmenter(
        seed=_seed,
        custom_logger=custom_logger,
    )
    gatherer = preference_comparisons.SyntheticGatherer(
        **gatherer_kwargs,
        seed=_seed,
        custom_logger=custom_logger,
    )
    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        model=reward_net,
        **reward_trainer_kwargs,
        custom_logger=custom_logger,
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparisons_per_iteration=comparisons_per_iteration,
        fragment_length=fragment_length,
        transition_oversampling=transition_oversampling,
        custom_logger=custom_logger,
        allow_variable_horizon=allow_variable_horizon,
        seed=_seed,
    )
    results = main_trainer.train(total_timesteps, total_comparisons)

    th.save(reward_net, os.path.join(log_dir, "final_reward_net.pt"))

    if save_preferences:
        main_trainer.dataset.save(os.path.join(log_dir, "preferences.pkl"))

    # Storing and evaluating the policy only makes sense if we actually used it
    if trajectory_path is None:
        serialize.save_stable_model(
            os.path.join(log_dir, "final_policy"),
            agent,
            vec_normalize,
        )
        results = dict(results)
        results["rollout"] = train.eval_policy(agent, venv)

    return results


def main_console():
    observer = FileStorageObserver(
        os.path.join("output", "sacred", "train_preference_comparisons"),
    )
    train_preference_comparisons_ex.observers.append(observer)
    train_preference_comparisons_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
