"""Common configuration elements for training imitation algorithms."""

import logging
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import sacred
from stable_baselines3.common import base_class, vec_env

from imitation.data import rollout, types
from imitation.policies import base
from imitation.util import logger as imit_logger
from imitation.util import sacred as sacred_util
from imitation.util import util

train_ingredient = sacred.Ingredient("train")
logger = logging.getLogger(__name__)


@train_ingredient.config
def config():
    # Logging
    log_root = None
    log_dir = None
    log_format_strs = ["tensorboard", "stdout"]

    # Environment config
    env_name = "seals/CartPole-v0"  # environment to train on
    num_vec = 8  # number of environments in VecEnv
    parallel = True  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps = None  # Set to positive int to limit episode horizons
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    # Demonstrations
    data_dir = "data/"
    rollout_path = None  # path to file containing rollouts
    n_expert_demos = None  # Num demos used. None uses every demo possible

    # Trainig
    # TODO(adam): does this need to be here or could it be in rl and separately for bc?
    # Is there any script we want to not take a policy?
    # Hmm, MCE IRL doesn't need this -- it can just ignore it though.
    # (Perhaps have that script warn if something non-default is set...?)
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {}

    # Evaluation
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return

    # TODO(adam): should we separate config into a separate file?
    # or just disable F841 on this whole file?
    _ = locals()  # quieten flake8
    del _


@train_ingredient.config
def defaults(data_dir, env_name, rollout_path):
    # If rollout_path not set explicitly, then guess it based on environment name.
    if rollout_path is None:
        rollout_hint = env_name.split("-")[0].replace("/", "_").lower()
        rollout_path = os.path.join(
            data_dir,
            "expert_models",
            f"{rollout_hint}_0",
            "rollouts",
            "final.pkl",
        )
        del rollout_hint

    _ = locals()  # quieten flake8
    del _


@train_ingredient.config_hook
def hook(config, command_name, logger):
    del logger
    updates = {}
    if config["train"]["log_dir"] is None:
        env_sanitized = config["train"]["env_name"].replace("/", "_")
        log_dir = os.path.join(
            "output",
            command_name,
            env_sanitized,
            util.make_unique_timestamp(),
        )
        updates = dict(log_dir=log_dir)
    return updates


@train_ingredient.named_config
def fast():
    n_expert_demos = 1
    n_episodes_eval = 1
    max_episode_steps = 5
    parallel = False  # easier to debug with everything in one process
    num_vec = 2
    _ = locals()  # quieten flake8
    del _


@train_ingredient.capture
def setup_logging(
    _run,
    log_dir: str,
    log_format_strs: Sequence[str],
) -> Tuple[imit_logger.HierarchicalLogger, str]:
    """Builds the imitation logger.

    Args:
        log_dir: The directory to log to.
        log_format_strs: The types of formats to log to.

    Returns:
        The configured imitation logger and `log_dir`.
        Returning `log_dir` avoids the caller needing to capture this value.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger.info("Logging to %s", log_dir)
    sacred_util.build_sacred_symlink(log_dir, _run)
    custom_logger = imit_logger.configure(log_dir, log_format_strs)
    return custom_logger, log_dir


@train_ingredient.capture
def make_venv(
    _seed,
    env_name: str,
    num_vec: int,
    parallel: bool,
    log_dir: str,
    max_episode_steps: int,
    env_make_kwargs: Mapping[str, Any],
) -> vec_env.VecEnv:
    """Builds the vector environment.

     Args:
        env_name: The environment to train in.
        num_vec: Number of `gym.Env` to vectorize.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Logs episode return statistics to a subdirectory 'monitor`.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.

    Returns:
        The constructed vector environment.
    """
    return util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        max_episode_steps=max_episode_steps,
        log_dir=log_dir,
        env_make_kwargs=env_make_kwargs,
    )


@train_ingredient.capture
def load_expert_trajs(
    rollout_path: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = types.load(rollout_path)
    logger.info(f"Loaded {len(expert_trajs)} expert trajectories from '{rollout_path}'")
    if n_expert_demos is not None:
        if len(expert_trajs) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available via {rollout_path}.",
            )
        expert_trajs = expert_trajs[:n_expert_demos]
        logger.info(f"Truncated to {n_expert_demos} expert trajectories")
    return expert_trajs


@train_ingredient.capture
def eval_policy(
    rl_algo: base_class.BaseAlgorithm,
    venv: vec_env.VecEnv,
    n_episodes_eval: int,
) -> Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Args:
        rl_algo: Algorithm to evaluate.
        venv: Environment to evaluate on.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.

    """
    sample_until_eval = rollout.make_min_episodes(n_episodes_eval)
    trajs = rollout.generate_trajectories(
        rl_algo,
        venv,
        sample_until=sample_until_eval,
    )
    return rollout.rollout_stats(trajs)
