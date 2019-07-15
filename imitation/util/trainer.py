"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""

import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


def init_trainer(env_id, seed=0, log_dir=None, use_gail=False,
                 use_random_expert=True,
                 num_vec=8, parallel=True, discrim_scale=False,
                 discrim_kwargs={}, reward_kwargs={}, trainer_kwargs={},
                 make_blank_policy_kwargs={}):
  """Builds a Trainer, ready to be trained on a vectorized environment
  and either expert rollout data or random rollout data.

  Args:
    env_id (str): The string id of a gym environment.
    seed (int): Random seed.
    log_dir (Optoinal[str]): Directory for logging output.
    use_gail (bool): If True, then train using GAIL. If False, then train
        using AIRL.
    policy_dir (str): The directory containing the pickled experts for
        generating rollouts. Only applicable if `use_random_expert` is True.
    use_random_expert (bool):
        If True, then use a blank (random) policy to generate rollouts.
        If False, then load an expert policy. Will crash if there is no expert
        policy in `policy_dir`.
    trainer_kwargs (dict): Aguments for the Trainer constructor.
    reward_kwargs (dict): Arguments for the `*RewardNet` constructor.
    discrim_kwargs (dict): Arguments for the `DiscrimNet*` constructor.
    make_blank_policy_kwargs: Keyword arguments passed to `make_blank_policy`,
        used to initialize the trainer.
  """
  env = util.make_vec_env(env_id, num_vec, seed=seed, parallel=parallel,
                          log_dir=log_dir)
  gen_policy = util.make_blank_policy(env, verbose=1,
                                      **make_blank_policy_kwargs)

  if use_random_expert:
    expert_policies = [gen_policy]
  else:
    expert_policies = util.load_policy(env_id)
    if expert_policies is None:
      raise ValueError(env)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space,
                                         env.action_space,
                                         scale=discrim_scale,
                                         **discrim_kwargs)
  else:
    rn = BasicShapedRewardNet(env.observation_space, env.action_space,
                              scale=discrim_scale, **reward_kwargs)
    discrim = discrim_net.DiscrimNetAIRL(rn, **discrim_kwargs)

  trainer = Trainer(env, gen_policy, discrim,
                    expert_policies=expert_policies, **trainer_kwargs)
  return trainer
