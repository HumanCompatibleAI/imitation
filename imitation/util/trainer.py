"""
Utility functions for manipulating AIRLTrainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.airl and imitation.util)
"""

import gin
import gin.tf
import tensorflow as tf

from imitation.airl import AIRLTrainer
import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
import imitation.util as util


@gin.configurable
def init_trainer(env_id, policy_dir, use_gail, use_random_expert=True,
                 **kwargs):
  """Build an AIRLTrainer, ready to be trained on a vectorized environment
  and either expert rollout data or random rollout data.

  Args:
    env_id (str): The string id of a gym environment.
    use_gail (bool): If True, then train using GAIL. If False, then train
        using AIRL.
    policy_dir (str): The directory containing the pickled experts for
        generating rollouts. Only applicable if `use_random_expert` is True.
    use_random_expert (bool):
        If True, then use a blank (random) policy to generate rollouts.
        If False, then load an expert policy. Will crash if there is no expert
        policy in `policy_dir`.
    **kwargs: Additional arguments For the AIRLTrainer constructor.
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
    discrim = discrim_net.DiscrimNetGAIL(env)
  else:
    rn = BasicShapedRewardNet(env)
    discrim = discrim_net.DiscrimNetAIRL(rn)

  trainer = AIRLTrainer(env, gen_policy, discrim,
                        expert_policies=expert_policy, **kwargs)
  return trainer
