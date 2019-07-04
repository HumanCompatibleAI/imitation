"""
Utility functions for manipulating Trainer.

(The primary reason these functions are here instead of in utils.py is to
prevent cyclic imports between imitation.trainer and imitation.util)
"""

import gin
import gin.tf
import stable_baselines

import imitation.discrim_net as discrim_net
from imitation.reward_net import BasicShapedRewardNet
from imitation.trainer import Trainer
import imitation.util as util


@gin.configurable
def init_trainer(env_id, policy_dir, use_gail, use_random_expert=True,
                 theta_units=[32, 32], phi_units=[32, 32],
                 discrim_scale=False, discrim_kwargs={}, trainer_kwargs={}):
  """Builds a Trainer, ready to be trained on a vectorized environment
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
    trainer_kwargs (dict): Aguments for the Trainer constructor.
    discrim_kwargs (dict): Arguments for the DiscrimNet* constructor.
  """
  env = util.make_vec_env(env_id, 8)
  gen_policy = stable_baselines.PPO2(util.FeedForward32Policy, env,
                                     verbose=1, tensorboard_log="output/",
                                     learning_rate=3e-3,
                                     nminibatches=32,
                                     noptepochs=10,
                                     n_steps=2048)

  if use_random_expert:
    expert_policies = [gen_policy]
  else:
    expert_policies = util.load_policy(env, basedir=policy_dir)
    if expert_policies is None:
      raise ValueError(env)

  if use_gail:
    discrim = discrim_net.DiscrimNetGAIL(env.observation_space,
                                         env.action_space,
                                         scale=discrim_scale,
                                         **discrim_kwargs)
  else:
    rn = BasicShapedRewardNet(env.observation_space, env.action_space,
                              theta_units=theta_units, phi_units=phi_units,
                              scale=discrim_scale)
    discrim = discrim_net.DiscrimNetAIRL(rn, **discrim_kwargs)

  trainer = Trainer(env, gen_policy, discrim,
                    expert_policies=expert_policies, **trainer_kwargs)
  return trainer
