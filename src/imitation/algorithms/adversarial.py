from functools import partial
import os.path as osp
from typing import Optional, Sequence
from warnings import warn

import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecNormalize
import tensorflow as tf
from tqdm import tqdm

from imitation import summaries
import imitation.rewards.discrim_net as discrim_net
from imitation.rewards.reward_net import BasicShapedRewardNet
import imitation.util as util
from imitation.util import buffer, logger, reward_wrapper, rollout


class AdversarialTrainer:
  """Trainer for GAIL and AIRL."""

  venv: VecEnv
  """The original vectorized environment."""

  venv_train: VecEnv
  """Like `self.venv`, but wrapped with train reward unless in debug mode.

  If `debug_use_ground_truth=True` was passed into the initializer then
  `self.venv_train` is the same as `self.venv`.
  """

  venv_test: VecEnv
  """Like `self.venv`, but wrapped with test reward unless in debug mode.

  If `debug_use_ground_truth=True` was passed into the initializer then
  `self.venv_test` is the same as `self.venv`.
  """

  def __init__(self,
               venv: VecEnv,
               gen_policy: BaseRLModel,
               discrim: discrim_net.DiscrimNet,
               expert_demos: rollout.Transitions,
               *,
               disc_batch_size: int = 2048,
               disc_minibatch_size: int = 256,
               disc_opt_cls: tf.train.Optimizer = tf.train.AdamOptimizer,
               disc_opt_kwargs: dict = {},
               gen_replay_buffer_capacity: Optional[int] = None,
               init_tensorboard: bool = False,
               init_tensorboard_graph: bool = False,
               debug_use_ground_truth: bool = False,
               ):
    """Builds Trainer.

    Args:
        venv: The vectorized environment to train in.
        gen_policy: The generator policy that is trained to maximize
          discriminator confusion. The generator batch size
          `self.gen_batch_size` is inferred from `gen_policy.n_batch`.
        discrim: The discriminator network.
          For GAIL, use a DiscrimNetGAIL. For AIRL, use a DiscrimNetAIRL.
        expert_demos: Transitions from an expert dataset.
        disc_batch_size: The default number of expert and generator transitions
          samples to feed to the discriminator in each call to
          `self.train_disc()`. (Half of the samples are expert and half of the
          samples are generator).
        disc_minibatch_size: The discriminator minibatch size. Each
          discriminator batch is split into minibatches and an Adam update is
          applied on the gradient resulting form each minibatch. Must evenly
          divide `disc_batch_size`. Must be an even number.
        disc_opt_cls: The optimizer for discriminator training.
        disc_opt_kwargs: Parameters for discriminator training.
        gen_replay_buffer_capacity: The capacity of the
          generator replay buffer (the number of obs-action-obs samples from
          the generator that can be stored).

          By default this is equal to `20 * self.gen_batch_size`.
        init_tensorboard: If True, makes various discriminator
          TensorBoard summaries.
        init_tensorboard_graph: If both this and `init_tensorboard` are True,
          then write a Tensorboard graph summary to disk.
        debug_use_ground_truth: If True, use the ground truth reward for
          `self.train_env`.
          This disables the reward wrapping that would normally replace
          the environment reward with the learned reward. This is useful for
          sanity checking that the policy training is functional.
    """
    assert util.logger.is_configured(), ("Requires call to "
                                         "imitation.util.logger.configure")
    self._sess = tf.get_default_session()
    self._global_step = tf.train.create_global_step()

    assert disc_batch_size % self.disc_minibatch_size == 0
    assert disc_minibatch_size % 2 == 0, \
      "discriminator minibatch size must be even " \
      "(equal split between generator and expert samples)"
    self.disc_batch_size = disc_batch_size
    self.disc_minibatch_size = disc_minibatch_size

    self.debug_use_ground_truth = debug_use_ground_truth

    self.venv = venv
    self._expert_demos = expert_demos
    self._gen_policy = gen_policy

    # Discriminator and reward output
    self._discrim = discrim
    self._disc_opt_cls = disc_opt_cls
    self._disc_opt_kwargs = disc_opt_kwargs
    with tf.variable_scope("trainer"):
      with tf.variable_scope("discriminator"):
        self._build_disc_train()
    self._init_tensorboard = init_tensorboard
    self._init_tensorboard_graph = init_tensorboard_graph
    if init_tensorboard:
      with tf.name_scope("summaries"):
        self._build_summarize()
    self._sess.run(tf.global_variables_initializer())

    if debug_use_ground_truth:
      # Would use an identity reward fn here, but RewardFns can't see rewards.
      self.reward_train = self.reward_test = None
      self.venv_train = self.venv_test = self.venv
    else:
      self.reward_train = partial(
          self.discrim.reward_train,
          gen_log_prob_fn=self._gen_policy.action_probability)
      self.reward_test = self.discrim.reward_test
      self.venv_train = reward_wrapper.RewardVecEnvWrapper(
          self.venv, self.reward_train)
      self.venv_test = reward_wrapper.RewardVecEnvWrapper(
          self.venv, self.reward_test)

    self.venv_train_norm = VecNormalize(self.venv_train)

    if gen_replay_buffer_capacity is None:
      gen_replay_buffer_capacity = 20 * self.gen_batch_size
    self._gen_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity,
                                                  self.venv)
    self._populate_gen_replay_buffer()
    self._exp_replay_buffer = buffer.ReplayBuffer.from_data(expert_demos)
    if self.disc_batch_size // 2 > len(self._exp_replay_buffer):
      warn("The discriminator batch size is more than twice the number of "
           "expert samples. This means that we will be reusing samples every "
           "discrim batch.")

  @property
  def gen_batch_size(self) -> int:
    return self.gen_policy.n_batch

  @property
  def discrim(self) -> discrim_net.DiscrimNet:
    """Discriminator being trained, used to compute reward for policy."""
    return self._discrim

  @property
  def expert_demos(self) -> util.rollout.Transitions:
    """The expert demonstrations that are being imitated."""
    return self._expert_demos

  @property
  def gen_policy(self) -> BaseRLModel:
    """Policy (i.e. the generator) being trained."""
    return self._gen_policy

  def train_disc(self, n_samples: Optional[int] = None) -> None:
    """Trains the discriminator to minimize classification cross-entropy.

    Args:
      n_samples: A number of transitions to sample from the generator
        replay buffer and the expert demonstration dataset. (Half of the
        samples are from each source). By default, `self.disc_batch_size`.
        `n_samples` must be a positive multiple of `self.disc_minibatch_size`.
    """
    if n_samples is None:
      n_samples = self.disc_batch_size
    n_updates = n_samples // self.disc_minibatch_size
    assert n_samples % self.disc_minibatch_size == 0
    assert n_updates >= 1
    for _ in range(n_updates):
      gen_samples = self._gen_replay_buffer.sample(self.disc_minibatch_size)
      self.train_disc_step(gen_samples=gen_samples)

  def train_disc_step(self, *,
                      gen_samples: Optional[rollout.Transitions] = None,
                      expert_samples: Optional[rollout.Transitions] = None,
                      ) -> None:
    """Perform a single discriminator update, optionally using provided samples.

    Args:
      gen_samples: Transition samples from the generator policy. If not
        provided, then take `self.disc_batch_size // 2` samples from the
        generator replay buffer.
      expert_samples: Transition samples from the expert. If not
        provided, then take `n_gen` expert samples from the expert
        dataset, where `n_gen` is the number of samples in `gen_samples`.
    """
    with logger.accumulate_means("disc"):
      fd = self._build_disc_feed_dict(gen_samples=gen_samples,
                                      expert_samples=expert_samples)
      step, _ = self._sess.run([self._global_step, self._disc_train_op],
                               feed_dict=fd)
      if self._init_tensorboard and step % 20 == 0:
        self._summarize(fd, step)

  def train_gen(self, total_timesteps: Optional[int] = None, callback=None):
    """Trains the generator (via PPO2) to maximize the discriminator loss.

    After the end of training populates the generator replay buffer (used in
    discriminator training) with `self.disc_batch_size` transitions.

    Args:
      total_timesteps: The number of transitions to sample from
        `self.venv_train_norm` during training. By default,
        `self.gen_batch_size`.
      callback: Callback argument to the Stable Baselines `RLModel.learn()`
        method.
    """
    if total_timesteps is None:
      total_timesteps = self.gen_batch_size

    with logger.accumulate_means("gen"):
      self.gen_policy.set_env(self.venv_train_norm)
      # TODO(adam): learn was not intended to be called for each training batch
      # It should work, but might incur unnecessary overhead: e.g. in PPO2
      # a new Runner instance is created each time. Also a hotspot for errors:
      # algorithms not tested for this use case, may reset state accidentally.
      self.gen_policy.learn(total_timesteps=total_timesteps,
                            reset_num_timesteps=False,
                            callback=callback)
      self._populate_gen_replay_buffer()

  def _populate_gen_replay_buffer(self) -> None:
    """Generate and store generator samples in the buffer.

    More specifically, rolls out generator-policy trajectories in the
    environment until `self.disc_batch_size` obs-act-obs samples are
    produced, and then stores these samples.
    """
    gen_samples = util.rollout.generate_transitions(
      self.gen_policy,
      self.venv_train_norm,
      n_timesteps=self.disc_batch_size // 2)
    self._gen_replay_buffer.store(gen_samples)

  def eval_disc_loss(self, **kwargs) -> float:
    """Evaluates the discriminator loss.

    The generator rollout parameters of the form "gen_*" are optional,
    but if one is given, then all such parameters must be filled (otherwise
    this method will error). If none of the generator rollout parameters are
    given, then a rollout with the same length as the expert rollout
    is generated on the fly.

    Args:
      gen_samples (Optional[rollout.Transitions]): Same as in `train_disc_step`.
      expert_samples (Optional[rollout.Transitions]): Same as in
        `train_disc_step`.

    Returns:
      The total cross-entropy error in the discriminator's classification.
    """
    fd = self._build_disc_feed_dict(**kwargs)
    return np.mean(self._sess.run(self.discrim.disc_loss, feed_dict=fd))

  def _build_summarize(self):
    graph = self._sess.graph if self._init_tensorboard_graph else None
    self._summary_writer = summaries.make_summary_writer(graph=graph)
    self.discrim.build_summaries()
    self._summary_op = tf.summary.merge_all()

  def _summarize(self, fd, step):
    events = self._sess.run(self._summary_op, feed_dict=fd)
    self._summary_writer.add_summary(events, step)

  def _build_disc_train(self):
    # Construct Train operation.
    disc_opt = self._disc_opt_cls(**self._disc_opt_kwargs)
    self._disc_train_op = disc_opt.minimize(
        tf.reduce_mean(self.discrim.disc_loss),
        global_step=self._global_step)

  def _build_disc_feed_dict(
      self, *,
      gen_samples: Optional[rollout.Transitions] = None,
      expert_samples: Optional[rollout.Transitions] = None,
  ) -> dict:
    """Build and return feed dict for the next discriminator training update.

    Args:
      gen_samples: Same as in `train_disc_step`.
      expert_samples: Same as in `train_disc_step`.
    """
    if gen_samples is None:
      gen_samples = self._gen_replay_buffer.sample(self.disc_batch_size // 2)
    n_gen = len(gen_samples.obs)

    if expert_samples is None:
      expert_samples = self._exp_replay_buffer.sample(n_gen)
    n_expert = len(expert_samples.obs)

    # Check dimensions.
    n_samples = n_expert + n_gen
    assert n_expert == len(expert_samples.acts)
    assert n_expert == len(expert_samples.next_obs)
    assert n_gen == len(gen_samples.acts)
    assert n_gen == len(gen_samples.next_obs)

    # Normalize expert observations to match generator observations.
    expert_obs_norm = self.venv_train_norm.normalize_obs(expert_samples.obs)

    # Concatenate rollouts, and label each row as expert or generator.
    obs = np.concatenate([expert_obs_norm, gen_samples.obs])
    acts = np.concatenate([expert_samples.acts, gen_samples.acts])
    next_obs = np.concatenate([expert_samples.next_obs, gen_samples.next_obs])
    labels = np.concatenate([np.zeros(n_expert, dtype=int),
                             np.ones(n_gen, dtype=int)])

    # Calculate generator-policy log probabilities.
    log_act_prob = self._gen_policy.action_probability(obs, actions=acts,
                                                       logp=True)
    assert len(log_act_prob) == n_samples
    log_act_prob = log_act_prob.reshape((n_samples,))

    fd = {
        self.discrim.obs_ph: obs,
        self.discrim.act_ph: acts,
        self.discrim.next_obs_ph: next_obs,
        self.discrim.labels_ph: labels,
        self.discrim.log_policy_act_prob_ph: log_act_prob,
    }
    return fd


def init_trainer(env_name: str,
                 expert_trajectories: Sequence[rollout.Trajectory],
                 *,
                 log_dir: str,
                 seed: int = 0,
                 use_gail: bool = False,
                 num_vec: int = 8,
                 parallel: bool = False,
                 max_episode_steps: Optional[int] = None,
                 scale: bool = True,
                 airl_entropy_weight: float = 1.0,
                 discrim_kwargs: dict = {},
                 reward_kwargs: dict = {},
                 trainer_kwargs: dict = {},
                 init_rl_kwargs: dict = {},
                 ):
  """Builds an AdversarialTrainer, ready to be trained on a vectorized
    environment and expert demonstrations.

  Args:
    env_name: The string id of a gym environment.
    expert_trajectories: Demonstrations from expert.
    seed: Random seed.
    log_dir: Directory for logging output.
    use_gail: If True, then train using GAIL. If False, then train
        using AIRL.
    num_vec: The number of vectorized environments.
    parallel: If True, then use SubprocVecEnv; otherwise, DummyVecEnv.
    max_episode_steps: If specified, wraps VecEnv in TimeLimit wrapper with
        this episode length before returning.
    policy_dir: The directory containing the pickled experts for
        generating rollouts.
    scale: If True, then scale input Tensors to the interval [0, 1].
    airl_entropy_weight: Only applicable for AIRL. The `entropy_weight`
        argument of `DiscrimNetAIRL.__init__`.
    trainer_kwargs: Arguments for the Trainer constructor.
    reward_kwargs: Arguments for the `*RewardNet` constructor.
    discrim_kwargs: Arguments for the `DiscrimNet*` constructor.
    init_rl_kwargs: Keyword arguments passed to `init_rl`,
        used to initialize the RL algorithm.
  """
  util.logger.configure(folder=osp.join(log_dir, 'generator'),
                        format_strs=['tensorboard', 'stdout'])
  env = util.make_vec_env(env_name, num_vec, seed=seed, parallel=parallel,
                          log_dir=log_dir, max_episode_steps=max_episode_steps)
  gen_policy = util.init_rl(env, verbose=1,
                            **init_rl_kwargs)

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

  expert_demos = util.rollout.flatten_trajectories(expert_trajectories)
  trainer = AdversarialTrainer(env, gen_policy, discrim, expert_demos,
                               **trainer_kwargs)
  return trainer
