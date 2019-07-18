"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""
import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


def init_trainer(env_id: str,
                 rollouts_glob: Optional[str] = None,
                 use_gail: bool = False,
                 num_vec: int = 8,
                 discrim_scale: bool = False,
                 max_n_files: int = 1,
                 discrim_kwargs: bool = {},
                 reward_kwargs: bool = {},
                 trainer_kwargs: bool = {},
                 make_blank_policy_kwargs: bool = {},
                 ):
  """Builds a Trainer, ready to be trained on a vectorized environment
  and expert demonstrations.

  Args:
    env_id: The string id of a gym environment.
    rollouts_glob: A glob that matches .npz files containing demonstration
      rollouts. If `rollouts_glob` is None, then use the default glob
      `f"data/rollouts/{env_id}_*.npz"`.
    use_gail: If True, then train using GAIL. If False, then train
        using AIRL.
    num_vec: The number of vectorized environments.
    policy_dir: The directory containing the pickled experts for
        generating rollouts.
    trainer_kwargs: Arguments for the Trainer constructor.
    reward_kwargs: Arguments for the `*RewardNet` constructor.
    discrim_kwargs: Arguments for the `DiscrimNet*` constructor.
    max_n_files: If provided, then only load the most recent `max_n_files`
        files, as sorted by modification times.
    make_blank_policy_kwargs: Keyword arguments passed to `make_blank_policy`,
        used to initialize the trainer.
  """
  env = util.make_vec_env(env_id, num_vec)
  gen_policy = util.make_blank_policy(env, verbose=1,
                                      **make_blank_policy_kwargs)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space,
                                         env.action_space,
                                         scale=discrim_scale,
                                         **discrim_kwargs)
  else:
    rn = BasicShapedRewardNet(env.observation_space, env.action_space,
                              scale=discrim_scale, **reward_kwargs)
    discrim = discrim_net.DiscrimNetAIRL(rn, **discrim_kwargs)

  expert_rollouts = util.rollout.load_transitions(
      rollouts_glob, env, max_n_files=max_n_files)
  trainer = Trainer(env, gen_policy, discrim, expert_rollouts,
                    **trainer_kwargs)
  return trainer
