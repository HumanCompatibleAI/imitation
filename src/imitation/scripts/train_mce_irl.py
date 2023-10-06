"""Train Finite-horizon tabular Maximum Causal Entropy IRL.

Can be used as a CLI script, or the `train_mce_irl` function
can be called directly.
"""

from functools import partial
import logging
import pathlib
import os.path as osp
from typing import Any, Mapping, Type


import numpy as np
import torch as th
from sacred.observers import FileStorageObserver
from seals import base_envs
from seals.diagnostics.cliff_world import CliffWorldEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from imitation.algorithms import mce_irl as mceirl_algorithm
from imitation.data import rollout
from imitation.scripts.config.train_mce_irl import train_mce_irl_ex
from imitation.scripts.ingredients import demonstrations
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward
from imitation.util import util

logger = logging.getLogger(__name__)


@train_mce_irl_ex.command
def train_mce_irl(
    mceirl: Mapping[str, Any],
    optimizer_cls: Type[th.optim.Optimizer],
    optimizer_kwargs: Mapping[str, Any],
    env_kwargs: Mapping[str, Any],
    num_vec: int,
    parallel: bool,
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    custom_logger, log_dir = logging_ingredient.setup_logging()
    expert_trajs = demonstrations.get_expert_trajectories()
    env_creator = partial(CliffWorldEnv, **env_kwargs)
    env = env_creator()

    env_fns = [lambda: base_envs.ExposePOMDPStateWrapper(env_creator())] * num_vec
    # This is just a vectorized environment because `generate_trajectories` expects one
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        state_venv = SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        state_venv = DummyVecEnv(env_fns)

    reward_net = reward.make_reward_net(state_venv)
    mceirl_trainer = mceirl_algorithm.MCEIRL(
        demonstrations=expert_trajs,
        env=env,
        reward_net=reward_net,
        rng=_rnd,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        discount=mceirl["discount"],
        linf_eps=mceirl["linf_eps"],
        grad_l2_eps=mceirl["grad_l2_eps"],
        log_interval=mceirl["log_interval"],
        custom_logger=custom_logger,
    )
    mceirl_trainer.train(max_iter=int(mceirl["max_iter"]))
    util.save_policy(mceirl_trainer.policy, policy_path=osp.join(log_dir, "final.th"))
    th.save(reward_net, osp.join(log_dir, "reward_net.pt"))
    imit_stats = policy_evaluation.eval_policy(mceirl_trainer.policy, state_venv)
    return {
        "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_mce_irl"
    observer = FileStorageObserver(observer_path)
    train_mce_irl_ex.observers.append(observer)
    train_mce_irl_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
