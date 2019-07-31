"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""
from typing import Optional

import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


def init_trainer(env_id: str,
                 rollout_glob: str,
                 *,
                 n_expert_demos: Optional[int] = None,
                 seed: int = 0,
                 log_dir: str = None,
                 use_gail: bool = False,
                 num_vec: int = 8,
                 parallel: bool = False,
                 max_n_files: int = 1,
                 scale: bool = True,
                 airl_entropy_weight: float = 1.0,
                 discrim_kwargs: bool = {},
                 reward_kwargs: bool = {},
                 trainer_kwargs: bool = {},
                 make_blank_policy_kwargs: bool = {},
                 ):
  """Builds a Trainer, ready to be trained on a vectorized environment
  and expert demonstrations.

  Args:
    env_id: The string id of a gym environment.
    rollout_glob: Argument for `imitation.util.rollout.load_trajectories`.
    n_expert_demos: The number of expert trajectories to actually use
        after loading them via `load_trajectories`.
        If None, then use all available trajectories.
        If `n_expert_demos` is an `int`, then use
        exactly `n_expert_demos` trajectories, erroring if there aren't
        enough trajectories. If there are surplus trajectories, then use the
        first `n_expert_demos` trajectories and drop the rest.
    seed: Random seed.
    log_dir: Directory for logging output.
    use_gail: If True, then train using GAIL. If False, then train
        using AIRL.
    num_vec: The number of vectorized environments.
    parallel: If True, then use SubprocVecEnv; otherwise, DummyVecEnv.
    max_n_files: If provided, then only load the most recent `max_n_files`
        files, as sorted by modification times.
    policy_dir: The directory containing the pickled experts for
        generating rollouts.
    scale: If True, then scale input Tensors to the interval [0, 1].
    airl_entropy_weight: Only applicable for AIRL. The `entropy_weight`
        argument of `DiscrimNetAIRL.__init__`.
    trainer_kwargs: Arguments for the Trainer constructor.
    reward_kwargs: Arguments for the `*RewardNet` constructor.
    discrim_kwargs: Arguments for the `DiscrimNet*` constructor.
    make_blank_policy_kwargs: Keyword arguments passed to `make_blank_policy`,
        used to initialize the trainer.
  """
  env = util.make_vec_env(env_id, num_vec, seed=seed, parallel=parallel,
                          log_dir=log_dir)
  gen_policy = util.init_rl(env, verbose=1,
                            **make_blank_policy_kwargs)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space,
                                         env.action_space,
                                         scale=scale,
                                         **discrim_kwargs)
  else:
    rn = BasicShapedRewardNet(env.observation_space,
                              env.action_space,
                              scale=scale,
                              **reward_kwargs)
    discrim = discrim_net.DiscrimNetAIRL(rn,
                                         entropy_weight=airl_entropy_weight,
                                         **discrim_kwargs)

  expert_demos = util.rollout.load_trajectories(rollout_glob,
                                                max_n_files=max_n_files)
  if n_expert_demos is not None:
      assert len(expert_demos) >= n_expert_demos
      expert_demos = expert_demos[:n_expert_demos]

  expert_rollouts = util.rollout.flatten_trajectories(expert_demos)[:3]
  trainer = Trainer(env, gen_policy, discrim, expert_rollouts,
                    **trainer_kwargs)
  return trainer
