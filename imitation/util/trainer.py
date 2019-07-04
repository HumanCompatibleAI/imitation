"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""

import gin
import gin.tf
import tensorflow as tf

import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


@gin.configurable
def init_trainer(env_id, policy_dir, use_gail, use_random_expert=True,
                 **trainer_kwargs):
  """Build an Trainer, ready to be trained on a vectorized environment
  and either expert rollout data or random rollout data.

  Args:
    env_id (str): The string id of a gym environment.
    use_gail (bool): If True, then train using GAIL. If False, then train
        using AIRL.
    policy_dir (str): The directory containing the pickled experts for
        generating rollouts. Only applicable if `use_random_expert` is True.
    use_random_expert (bool):
        If True, then use a blank (random) policy to generate rollouts.
        If False, then load an expert policy. Will crash if DNE.
    **trainer_kwargs: Additional arguments For the Trainer constructor.
  """
  env = util.make_vec_env(env_id, 8)
  gen_policy = util.make_blank_policy(env, init_tensorboard=False)
  tf.logging.info("use_random_expert %s", use_random_expert)
  if use_random_expert:
    expert_policy = gen_policy
  else:
    expert_policy = util.load_policy(env, basedir=policy_dir)
    if expert_policy is None:
      raise ValueError(env)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space, env.action_space)
  else:
    rn = BasicShapedRewardNet(env.observation_space, env.action_space)
    discrim = discrim_net.DiscrimNetAIRL(rn)

  trainer = Trainer(env, gen_policy, discrim,
                    expert_policies=expert_policy, **trainer_kwargs)
  return trainer
