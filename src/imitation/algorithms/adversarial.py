from functools import partial
from typing import Optional, Tuple, Union
from warnings import warn

import gym
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
import tensorflow as tf
from tqdm import tqdm

from imitation import summaries
import imitation.rewards.discrim_net as discrim_net
from imitation.rewards.reward_net import BasicShapedRewardNet
import imitation.util as util
from imitation.util import buffer, reward_wrapper


class AdversarialTrainer:
  """Trainer for GAIL and AIRL."""

  env: gym.Env
  """The original environment."""

  env_train: gym.Env
  """Like `self.env`, but wrapped with train reward unless in debug mode.

  If `debug_use_ground_truth=True` was passed into the initializer then
  `self.env_train` is the same as `self.env`.
  """

  env_test: gym.Env
  """Like `self.env`, but wrapped with test reward unless in debug mode.

  If `debug_use_ground_truth=True` was passed into the initializer then
  `self.env_test` is the same as `self.env`.
  """

  def __init__(self,
               env: Union[gym.Env, str],
               gen_policy: BaseRLModel,
               discrim: discrim_net.DiscrimNet,
               expert_rollouts: Tuple[np.ndarray, np.ndarray, np.ndarray],
               *,
               disc_opt_cls: tf.train.Optimizer = tf.train.AdamOptimizer,
               disc_opt_kwargs: dict = {},
               n_disc_samples_per_buffer: int = 200,
               gen_replay_buffer_capacity: Optional[int] = None,
               init_tensorboard: bool = False,
               debug_use_ground_truth: bool = False):
    """Builds Trainer.

    Args:
        env: A Gym environment or ID that the policy is trained on.
        gen_policy: The generator policy that is trained to maximize
                    discriminator confusion.
        discrim: The discriminator network.
            For GAIL, use a DiscrimNetGAIL. For AIRL, use a DiscrimNetAIRL.
        expert_rollouts: A tuple of three arrays from expert rollouts,
            `old_obs`, `act`, and `new_obs`.
        disc_opt_cls: The optimizer for discriminator training.
        disc_opt_kwargs: Parameters for discriminator training.
        n_disc_samples_per_buffer: The number of obs-act-obs triples
            sampled from each replay buffer (expert and generator) during each
            step of discriminator training. This is also the number of triples
            stored in the replay buffer after each epoch of generator training.
        gen_replay_buffer_capacity: The capacity of the
            generator replay buffer (the number of obs-action-obs samples from
            the generator that can be stored).

            By default this is equal to `20 * n_disc_samples_per_buffer`.
        init_tensorboard: If True, makes various discriminator
            TensorBoard summaries.
        debug_use_ground_truth: If True, use the ground truth reward for
            `self.train_env`.
            This disables the reward wrapping that would normally replace
            the environment reward with the learned reward. This is useful for
            sanity checking that the policy training is functional.
    """
    self._sess = tf.get_default_session()
    self._global_step = tf.train.create_global_step()

    self._n_disc_samples_per_buffer = n_disc_samples_per_buffer
    self.debug_use_ground_truth = debug_use_ground_truth

    self.env = util.maybe_load_env(env, vectorize=True)
    self._gen_policy = gen_policy

    # Discriminator and reward output
    self._discrim = discrim
    self._disc_opt_cls = disc_opt_cls
    self._disc_opt_kwargs = disc_opt_kwargs
    with tf.variable_scope("trainer"):
      with tf.variable_scope("discriminator"):
        self._build_disc_train()
    self._init_tensorboard = init_tensorboard
    if init_tensorboard:
      with tf.name_scope("summaries"):
        self._build_summarize()
    self._sess.run(tf.global_variables_initializer())

    if debug_use_ground_truth:
      self.env_train = self.env_test = self.env
    else:
      reward_train = partial(
          self.discrim.reward_train,
          gen_log_prob_fn=self._gen_policy.action_probability)
      self.env_train = reward_wrapper.RewardVecEnvWrapper(
          self.env, reward_train)
      self.env_test = reward_wrapper.RewardVecEnvWrapper(
          self.env, self.discrim.reward_test)

    if gen_replay_buffer_capacity is None:
      gen_replay_buffer_capacity = 20 * self._n_disc_samples_per_buffer
    self._gen_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity,
                                                  self.env)
    self._populate_gen_replay_buffer()
    self._exp_replay_buffer = buffer.ReplayBuffer.from_data(*expert_rollouts)
    if n_disc_samples_per_buffer > len(self._exp_replay_buffer):
      warn("The discriminator batch size is larger than the number of "
           "expert samples.")

  @property
  def discrim(self) -> discrim_net.DiscrimNet:
    """Discriminator being trained, used to compute reward for policy."""
    return self._discrim

  @property
  def gen_policy(self) -> BaseRLModel:
    """Policy (i.e. the generator) being trained."""
    return self._gen_policy

  def train_disc(self, n_steps=10, **kwargs):
    """Trains the discriminator to minimize classification cross-entropy.

    Args:
        n_steps (int): The number of training steps.
        gen_old_obs (np.ndarray): See `_build_disc_feed_dict`.
        gen_act (np.ndarray): See `_build_disc_feed_dict`.
        gen_new_obs (np.ndarray): See `_build_disc_feed_dict`.
    """
    for _ in range(n_steps):
      fd = self._build_disc_feed_dict(**kwargs)
      step, _ = self._sess.run([self._global_step, self._disc_train_op],
                               feed_dict=fd)
      if self._init_tensorboard and step % 20 == 0:
        self._summarize(fd, step)

  def train_gen(self, n_steps=10000):
    self._gen_policy.set_env(self.env_train)
    # TODO(adam): learn was not intended to be called for each training batch
    # It should work, but might incur unnecessary overhead: e.g. in PPO2
    # a new Runner instance is created each time. Also a hotspot for errors:
    # algorithms not tested for this use case, may reset state accidentally.
    self._gen_policy.learn(n_steps, reset_num_timesteps=False)
    self._populate_gen_replay_buffer()

  def _populate_gen_replay_buffer(self) -> None:
    """Generate and store generator samples in the buffer.

    More specifically, rolls out generator-policy trajectories in the
    environment until `self._n_disc_samples_per_buffer` obs-act-obs samples are
    produced, and then stores these samples.
    """
    gen_rollouts = util.rollout.generate_transitions(
        self._gen_policy, self.env_train,
        n_timesteps=self._n_disc_samples_per_buffer)[:3]
    self._gen_replay_buffer.store(*gen_rollouts)

  def train(self, n_epochs=100, *, n_gen_steps_per_epoch=None,
            n_disc_steps_per_epoch=None):
    """Trains the discriminator and generator against each other.

    Args:
        n_epochs (int): The number of epochs to train. Every epoch consists
            of training the discriminator and then training the generator.
        n_disc_steps_per_epoch (int): The number of steps to train the
            discriminator every epoch. More precisely, the number of full batch
            Adam optimizer steps to perform.
        n_gen_steps_per_epoch (int): The number of generator training steps
            during each epoch. (ie, the timesteps argument in in
            `policy.learn(timesteps)`).
    """
    for i in tqdm(range(n_epochs), desc="AIRL train"):
      self.train_disc(**_n_steps_if_not_none(n_disc_steps_per_epoch))
      self.train_gen(**_n_steps_if_not_none(n_gen_steps_per_epoch))

  def eval_disc_loss(self, **kwargs):
    """Evaluates the discriminator loss.

    The generator rollout parameters of the form "gen_*" are optional,
    but if one is given, then all such parameters must be filled (otherwise
    this method will error). If none of the generator rollout parameters are
    given, then a rollout with the same length as the expert rollout
    is generated on the fly.

    Args:
        gen_old_obs (np.ndarray): See `_build_disc_feed_dict`.
        gen_act (np.ndarray): See `_build_disc_feed_dict`.
        gen_new_obs (np.ndarray): See `_build_disc_feed_dict`.

    Returns:
        discriminator_loss (float): The total cross-entropy error in the
            discriminator's classification.
    """
    fd = self._build_disc_feed_dict(**kwargs)
    return np.mean(self._sess.run(self.discrim.disc_loss, feed_dict=fd))

  def _build_summarize(self):
    self._summary_writer = summaries.make_summary_writer(
        graph=self._sess.graph)
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

  def _build_disc_feed_dict(self, *,
                            gen_old_obs: Optional[np.ndarray] = None,
                            gen_act: Optional[np.ndarray] = None,
                            gen_new_obs: Optional[np.ndarray] = None,
                            ) -> dict:
    """Build a feed dict that holds the next training batch of generator
    and expert obs-act-obs triples.

    Args:
        gen_old_obs (np.ndarray): A numpy array with shape
            `[self.n_disc_samples_per_buffer_per_buffer] + env.observation_space.shape`.
            The ith observation in this array is the observation seen when the
            generator chooses action `gen_act[i]`.
        gen_act (np.ndarray): A numpy array with shape
            `[self.n_disc_samples_per_buffer_per_buffer] + env.action_space.shape`.
        gen_new_obs (np.ndarray): A numpy array with shape
            `[self.n_disc_samples_per_buffer_per_buffer] + env.observation_space.shape`.
            The ith observation in this array is from the transition state after
            the generator chooses action `gen_act[i]`.
    """  # noqa: E501

    # Sample generator training batch from replay buffers, unless provided
    # in argument.
    none_count = sum(int(x is None)
                     for x in (gen_old_obs, gen_act, gen_new_obs))
    if none_count == 3:
      tf.logging.debug("_build_disc_feed_dict: No generator rollout "
                       "parameters were "
                       "provided, so we are generating them now.")
      gen_old_obs, gen_act, gen_new_obs = self._gen_replay_buffer.sample(
          self._n_disc_samples_per_buffer)
    elif none_count != 0:
      raise ValueError("Gave some but not all of the generator params.")

    # Sample expert training batch from replay buffer.
    expert_old_obs, expert_act, expert_new_obs = self._exp_replay_buffer.sample(
        self._n_disc_samples_per_buffer)

    # Check dimensions.
    n_expert = len(expert_old_obs)
    n_gen = len(gen_old_obs)
    N = n_expert + n_gen
    assert n_expert == len(expert_act)
    assert n_expert == len(expert_new_obs)
    assert n_gen == len(gen_act)
    assert n_gen == len(gen_new_obs)

    # Concatenate rollouts, and label each row as expert or generator.
    old_obs = np.concatenate([expert_old_obs, gen_old_obs])
    act = np.concatenate([expert_act, gen_act])
    new_obs = np.concatenate([expert_new_obs, gen_new_obs])
    labels = np.concatenate([np.zeros(n_expert, dtype=int),
                             np.ones(n_gen, dtype=int)])

    # Calculate generator-policy log probabilities.
    log_act_prob = self._gen_policy.action_probability(old_obs, actions=act,
                                                       logp=True)
    assert len(log_act_prob) == N
    log_act_prob = log_act_prob.reshape((N,))

    fd = {
        self.discrim.old_obs_ph: old_obs,
        self.discrim.act_ph: act,
        self.discrim.new_obs_ph: new_obs,
        self.discrim.labels_ph: labels,
        self.discrim.log_policy_act_prob_ph: log_act_prob,
    }
    return fd


def _n_steps_if_not_none(n_steps):
  if n_steps is None:
    return {}
  else:
    return dict(n_steps=n_steps)


def init_trainer(env_id: str,
                 rollout_glob: str,
                 *,
                 n_expert_demos: Optional[int] = None,
                 seed: int = 0,
                 log_dir: str = None,
                 use_gail: bool = False,
                 num_vec: int = 8,
                 parallel: bool = False,
                 max_episode_steps: Optional[int] = None,
                 max_n_files: int = 1,
                 scale: bool = True,
                 airl_entropy_weight: float = 1.0,
                 discrim_kwargs: bool = {},
                 reward_kwargs: bool = {},
                 trainer_kwargs: bool = {},
                 make_blank_policy_kwargs: bool = {},
                 ):
  """Builds an AdversarialTrainer, ready to be trained on a vectorized
    environment and expert demonstrations.

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
    max_episode_steps: If specified, wraps VecEnv in TimeLimit wrapper with
        this episode length before returning.
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
                          log_dir=log_dir, max_episode_steps=max_episode_steps)
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
  trainer = AdversarialTrainer(env, gen_policy, discrim, expert_rollouts,
                               **trainer_kwargs)
  return trainer
