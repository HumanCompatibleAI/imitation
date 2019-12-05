from contextlib import contextmanager
from functools import partial
from queue import Queue
from threading import Thread
from typing import Optional, Sequence
from warnings import warn

import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv
import tensorflow as tf
from tqdm import tqdm

from imitation import summaries
import imitation.rewards.discrim_net as discrim_net
from imitation.rewards.reward_net import BasicShapedRewardNet
import imitation.util as util
from imitation.util import buffer, reward_wrapper, rollout


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
               disc_opt_cls: tf.train.Optimizer = tf.train.AdamOptimizer,
               disc_opt_kwargs: dict = {},
               gen_replay_buffer_capacity: Optional[int] = None,
               init_tensorboard: bool = False,
               init_tensorboard_graph: bool = False,
               debug_use_ground_truth: bool = False):
    """Builds Trainer.

    Args:
        venv: The vectorized environment to train in.
        gen_policy: The generator policy that is trained to maximize
                    discriminator confusion. The discriminator batch size is
                    inferred from `gen_policy.batch_size`.
        discrim: The discriminator network.
            For GAIL, use a DiscrimNetGAIL. For AIRL, use a DiscrimNetAIRL.
        expert_demos: Transitions from an expert dataset.
        disc_opt_cls: The optimizer for discriminator training.
        disc_opt_kwargs: Parameters for discriminator training.
        gen_replay_buffer_capacity: The capacity of the
            generator replay buffer (the number of obs-action-obs samples from
            the generator that can be stored).

            By default this is equal to `batch_size`.
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
    self._sess = tf.get_default_session()
    self._global_step = tf.train.create_global_step()

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
      self.venv_train = self.venv_test = self.venv
    else:
      reward_train = partial(
          self.discrim.reward_train,
          gen_log_prob_fn=self._gen_policy.action_probability)
      self.venv_train = reward_wrapper.RewardVecEnvWrapper(
          self.venv, reward_train)
      self.venv_test = reward_wrapper.RewardVecEnvWrapper(
          self.venv, self.discrim.reward_test)

    if gen_replay_buffer_capacity is None:
      gen_replay_buffer_capacity = self.batch_size
    self._gen_replay_buffer = buffer.ReplayBuffer(gen_replay_buffer_capacity,
                                                  self.venv)
    self._populate_gen_replay_buffer()
    self._exp_replay_buffer = buffer.ReplayBuffer.from_data(expert_demos)
    if self.batch_size > len(self._exp_replay_buffer):
      warn("The batch size is larger than the number of expert samples. "
           "This means that we will be reusing samples every discrim batch.")

  @property
  def batch_size(self) -> int:
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

  def train_disc(self, total_timesteps: int, **kwargs):
    """Trains the discriminator to minimize classification cross-entropy.

    Every discriminator update, uses `self.batch_size` transition samples from
    each of the expert and generator replay buffers.

    Args:
        total_timesteps: The number of training steps.
    """
    n_epochs = total_timesteps // self.batch_size
    assert n_epochs >= 1, "should have at least one update"
    for _ in tqdm(range(n_epochs),  desc="Adversarial train discriminator"):
      fd = self._build_disc_feed_dict()
      step, _ = self._sess.run([self._global_step, self._disc_train_op],
                               feed_dict=fd)
      if self._init_tensorboard and step % 20 == 0:
        self._summarize(fd, step)

  def train_disc_on_samples(self, **kwargs):
    """Trains the discriminator to minimize classification cross-entropy.

    Uses the provided samples to perform a single discriminator update.

    Args:
        gen_obs (np.ndarray): See `_build_disc_feed_dict`.
        gen_acts (np.ndarray): See `_build_disc_feed_dict`.
        gen_next_obs (np.ndarray): See `_build_disc_feed_dict`.
    """
    fd = self._build_disc_feed_dict(**kwargs)
    step, _ = self._sess.run([self._global_step, self._disc_train_op],
                             feed_dict=fd)
    if self._init_tensorboard and step % 20 == 0:
      self._summarize(fd, step)

  def train_gen(self, total_timesteps: int, callback=None):
    """Trains the generator (via PPO2) to maximize the discriminator loss.

    Applies `self._gen_policy.n_minibatch` updates to the generator policy,
    using a total of `self._gen_policy.n_batch` new updates.
    """
    self.gen_policy.set_env(self.venv_train)
    # TODO(adam): learn was not intended to be called for each training batch
    # It should work, but might incur unnecessary overhead: e.g. in PPO2
    # a new Runner instance is created each time. Also a hotspot for errors:
    # algorithms not tested for this use case, may reset state accidentally.
    self.gen_policy.learn(total_timesteps=self.batch_size,
                          reset_num_timesteps=False,
                          callback=callback)
    self._populate_gen_replay_buffer()

  @contextmanager
  def train_gen_by_batch(self,
                         total_timesteps: int,
                         populate_replay_buffer: bool = True,
                         ):
    """Train the generator batch-by-batch without exitting `gen_policy.learn()`.

    Useful for adding discriminator training and plotting code in between
    `gen_policy.learn()` updates without relying on a callback.

    Args:
      total_timesteps: Argument to `self.gen_policy.learn`. (An upper bound on
        the number of timesteps in `self.venv_train` to use during training.)
      populate_replay_buffer: If True, then automatically generator samples to
        the generator replay buffer after each policy update.

    Returns:
      A ContextManager for a function `train_gen_part` which performs a single
      PPO2 update and which returns the state `(_locals, _globals)` inside PPO2.
      It must be called exactly `total_timesteps // self.batch_size` times
      before the ContextManager exits (otherwise `self.gen_policy.learn()` was
      not run to completion, and an error is thrown).
    """

    # Details:
    #
    # `PPO2.learn(total_timesteps, callback)` calls callback after each update.
    # We use the callback and a Thread to tie `PPO2.learn` to a generator that
    # yields and puts `PPO2.learn` on hold after each update.
    #
    # Based on: https://stackoverflow.com/a/9968886/1091722
    #
    # TODO(shwang): Getting the details right turned out more complicated than
    # I thought and increased the complexity of this function. Should discuss
    # whether to replace with something simpler.
    queue = Queue(1)
    job_finished = object()

    def callback(_locals, _globals):
      if populate_replay_buffer:
        self._populate_gen_replay_buffer()
      queue.put((_locals, _globals))
      queue.join()

    def job():
      self._gen_policy.set_env(self.venv_train)
      self._gen_policy.learn(total_timesteps,
                             reset_num_timesteps=False,
                             callback=callback)
      queue.put(job_finished)

    def generator():
      t = Thread(target=job)
      t.start()

      while True:
        item = queue.get()
        if item is not job_finished:
          yield item
          queue.task_done()
        else:
          queue.task_done()
          break
      t.join(timeout=1)
      assert not t.is_alive()

    gen = generator()
    try:
      yield lambda: next(gen)
    finally:
      # To complete the call to `PPO2.learn` (and allow job() to exit),
      # we need to call `next(generator)` one final time after the final
      # `yield`, triggering a StopIteratorException. Otherwise Thread `t`
      # hangs forever in the background.
      default = object()
      assert next(gen, default) is default

  def train(self, total_timesteps: int) -> None:
    """Trains the discriminator and generator against each other.

    Roughly equivalent to calling `self.train_disc(self.batch_size)` and
    `self.train_gen(self.batch_size)` `total_timesteps // self.batch_size`
    times.

    Args:
        total_timesteps (int): The number of transitions to sample from
          `self.venv_train`. (More precisely, this is an upper bound. The
          actual number of transition samples used is
          `total_timesteps // self.n_batch * self.n_batch`).
    """
    n_epochs = total_timesteps // self.batch_size
    with self.train_gen_by_batch(total_timesteps) as train_gen:
      for i in tqdm(range(n_epochs), desc="Adversarial train"):
        self.train_disc(total_timesteps=self.batch_size)
        train_gen()

  def _populate_gen_replay_buffer(self) -> None:
    """Generate and store generator samples in the buffer.

    More specifically, rolls out generator-policy trajectories in the
    environment until `self._n_disc_samples_per_buffer` obs-act-obs samples are
    produced, and then stores these samples.
    """
    gen_rollouts = util.rollout.generate_transitions(
      self._gen_policy, self.venv_train,
      n_timesteps=self.batch_size)
    self._gen_replay_buffer.store(gen_rollouts)

  def eval_disc_loss(self, **kwargs):
    """Evaluates the discriminator loss.

    The generator rollout parameters of the form "gen_*" are optional,
    but if one is given, then all such parameters must be filled (otherwise
    this method will error). If none of the generator rollout parameters are
    given, then a rollout with the same length as the expert rollout
    is generated on the fly.

    Args:
        gen_obs (np.ndarray): See `_build_disc_feed_dict`.
        gen_acts (np.ndarray): See `_build_disc_feed_dict`.
        gen_next_obs (np.ndarray): See `_build_disc_feed_dict`.

    Returns:
        discriminator_loss (float): The total cross-entropy error in the
            discriminator's classification.
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

  def _build_disc_feed_dict(self, *,
                            gen_obs: Optional[np.ndarray] = None,
                            gen_acts: Optional[np.ndarray] = None,
                            gen_next_obs: Optional[np.ndarray] = None,
                            ) -> dict:
    """Build a feed dict that holds the next training batch of generator
    and expert obs-act-obs triples.

    Args:
        gen_obs (np.ndarray): A numpy array with shape
            `[self.batch_size] + env.observation_space.shape`.
            The ith observation in this array is the observation seen when the
            generator chooses action `gen_acts[i]`.
        gen_acts (np.ndarray): A numpy array with shape
            `[self.batch_size] + env.action_space.shape`.
        gen_next_obs (np.ndarray): A numpy array with shape
            `[self.batch_size] + env.observation_space.shape`.
            The ith observation in this array is from the transition state after
            the generator chooses action `gen_acts[i]`.
    """  # noqa: E501

    # Sample generator training batch from replay buffers, unless provided
    # in argument.
    none_count = sum(int(x is None)
                     for x in (gen_obs, gen_acts, gen_next_obs))
    if none_count == 3:
      tf.logging.debug("_build_disc_feed_dict: No generator rollout "
                       "parameters were "
                       "provided, so we are generating them now.")
      gen_sample = self._gen_replay_buffer.sample(self.batch_size)
      gen_obs = gen_sample.obs
      gen_acts = gen_sample.acts
      gen_next_obs = gen_sample.next_obs
    elif none_count != 0:
      raise ValueError("Gave some but not all of the generator params.")

    # Sample expert training batch from replay buffer.
    expert_sample = self._exp_replay_buffer.sample(self.batch_size)

    # Check dimensions.
    n_expert = len(expert_sample.obs)
    n_gen = len(gen_obs)
    N = n_expert + n_gen
    assert n_expert == len(expert_sample.acts)
    assert n_expert == len(expert_sample.next_obs)
    assert n_gen == len(gen_acts)
    assert n_gen == len(gen_next_obs)

    # Concatenate rollouts, and label each row as expert or generator.
    obs = np.concatenate([expert_sample.obs, gen_obs])
    acts = np.concatenate([expert_sample.acts, gen_acts])
    next_obs = np.concatenate([expert_sample.next_obs, gen_next_obs])
    labels = np.concatenate([np.zeros(n_expert, dtype=int),
                             np.ones(n_gen, dtype=int)])

    # Calculate generator-policy log probabilities.
    log_act_prob = self._gen_policy.action_probability(obs, actions=acts,
                                                       logp=True)
    assert len(log_act_prob) == N
    log_act_prob = log_act_prob.reshape((N,))

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
                 seed: int = 0,
                 log_dir: str = None,
                 use_gail: bool = False,
                 num_vec: int = 8,
                 parallel: bool = False,
                 max_episode_steps: Optional[int] = None,
                 scale: bool = True,
                 airl_entropy_weight: float = 1.0,
                 discrim_kwargs: bool = {},
                 reward_kwargs: bool = {},
                 trainer_kwargs: bool = {},
                 init_rl_kwargs: bool = {},
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
