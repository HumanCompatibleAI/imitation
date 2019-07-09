from typing import Optional, Sequence, Union
from warnings import warn

import gym
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnvWrapper
import tensorflow as tf
from tqdm import tqdm

from imitation import summaries
from imitation.discrim_net import DiscrimNet
from imitation.util import maybe_load_env, rollout
from imitation.util.buffer import ReplayBuffer


class Trainer:
  """Trainer for GAIL and AIRL."""

  def __init__(self,
               env: Union[gym.Env, str],
               gen_policy: BaseRLModel,
               discrim: DiscrimNet,
               expert_policies: Sequence[BaseRLModel], *,
               disc_opt_cls: tf.train.Optimizer = tf.train.AdamOptimizer,
               disc_opt_kwargs: dict = {},
               n_disc_samples_per_buffer: int = 200,
               n_expert_samples: int = 4000,
               gen_replay_buffer_capacity: Optional[int] = None,
               init_tensorboard: bool = False,
               debug_use_ground_truth: bool = False):
    """Builds Trainer.

    Args:
        env: A Gym environment or ID that the policy is trained on.
        gen_policy: The generator policy that trained to maximize discriminator
                    confusion.
        discrim: The discriminator network.
            For GAIL, use a DiscrimNetGAIL. For AIRL, use a DiscrimNetAIRL.
        expert_policies: An expert policy
            or a list of expert policies that are used to generate example
            obs-action-obs triples.

            WARNING:
            Due to the way VecEnvs handle episode completion states, the last
            obs-act-obs triple in every episode is omitted. (See issue #1.)
        disc_opt_cls: The optimizer for discriminator training.
        disc_opt_kwargs: Parameters for discriminator training.
        n_disc_samples_per_buffer: The number of obs-act-obs triples
            sampled from each replay buffer (expert and generator) during each
            step of discriminator training. This is also the number of triples
            stored in the replay buffer after each epoch of generator training.
        n_expert_samples: The number of expert obs-action-obs triples
            that are generated. If the number of expert policies given
            doesn't divide this number evenly, then the last expert policy
            generates more timesteps.
        gen_replay_buffer_capacity: The capacity of the
            generator replay buffer (the number of obs-action-obs samples from
            the generator that can be stored).

            By default this is equal to `20 * n_disc_samples_per_buffer`.
        init_tensorboard: If True, makes various discriminator
            TensorBoard summaries. (Generator summaries appear under a
            different runname than the discriminator summaries because they
            are configured by initializing the stable_baselines policy).
        debug_use_ground_truth: If True, use the ground truth reward.
            This disables the reward wrapping that would normally replace
            the environment reward with the learned reward. This is useful for
            sanity checking that the policy training is functional.
    """
    if n_disc_samples_per_buffer > n_expert_samples:
      warn("The discriminator batch size is larger than the number of "
           "expert samples.")

    # TODO(adam): we're not guaranteed to use this session, see issue #31
    self._sess = tf.Session()

    self.env = maybe_load_env(env, vectorize=True)
    self.gen_policy = gen_policy
    self.expert_policies = expert_policies
    self._n_disc_samples_per_buffer = n_disc_samples_per_buffer
    self.debug_use_ground_truth = debug_use_ground_truth

    self._global_step = tf.train.create_global_step()

    # Discriminator and reward output
    self._disc_opt_cls = disc_opt_cls
    self._disc_opt_kwargs = disc_opt_kwargs
    with tf.variable_scope("trainer"):
      with tf.variable_scope("discriminator"):
        self.discrim = discrim
        self._build_disc_train()
      self._build_policy_train_reward()
      self._build_test_reward()
    self._init_tensorboard = init_tensorboard
    if init_tensorboard:
      with tf.name_scope("summaries"):
        self._build_summarize()

    self._sess.run(tf.global_variables_initializer())

    # TODO(adam): make this wrapping configurable for debugging purposes
    self.env = self.wrap_env_train_reward(self.env)
    self.gen_policy.set_env(self.env)

    if gen_replay_buffer_capacity is None:
        gen_replay_buffer_capacity = 20 * self._n_disc_samples_per_buffer
    self._gen_replay_buffer = ReplayBuffer(gen_replay_buffer_capacity, self.env)
    self._populate_gen_replay_buffer()

    exp_rollouts = rollout.generate_multiple(
        self.expert_policies, self.env, n_expert_samples)[:3]
    self._exp_replay_buffer = ReplayBuffer.from_data(*exp_rollouts)

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
    self.gen_policy.set_env(self.env)
    # TODO(adam): learn was not intended to be called for each training batch
    # It should work, but might incur unnecessary overhead: e.g. in PPO2
    # a new Runner instance is created each time. Also a hotspot for errors:
    # algorithms not tested for this use case, may reset state accidentally.
    self.gen_policy.learn(n_steps, reset_num_timesteps=False)
    self._populate_gen_replay_buffer()

  def _populate_gen_replay_buffer(self) -> None:
    """Generate and store generator samples in the buffer.

    More specifically, rolls out generator-policy trajectories in the
    environment until `self._n_disc_samples_per_buffer` obs-act-obs samples are
    produced, and then stores these samples.
    """
    gen_rollouts = rollout.generate(
        self.gen_policy, self.env,
        n_timesteps=self._n_disc_samples_per_buffer)[:3]
    self._gen_replay_buffer.store(*gen_rollouts)

  def train(self, *, n_epochs=100, n_gen_steps_per_epoch=None,
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

  def wrap_env_train_reward(self, env):
    """Returns the given Env wrapped with a reward function that returns
    the AIRL training reward (discriminator confusion).

    The wrapped `Env`'s reward is directly evaluated from the reward network,
    and therefore changes whenever `self.train()` is called.

    Args:
        env (str, Env, or VecEnv): The Env that we want to wrap. If a
            string environment name is given or a Env is given, then we first
            convert to a VecEnv before continuing.
    wrapped_env (VecEnv): The wrapped environment with a new reward.
    """
    env = maybe_load_env(env, vectorize=True)
    if self.debug_use_ground_truth:
      return env
    else:
      return _RewardVecEnvWrapper(env, self._policy_train_reward_fn)

  def wrap_env_test_reward(self, env):
    """Returns the given Env wrapped with a reward function that returns
    the reward learned by this Trainer.

    The wrapped `Env`'s reward is directly evaluated from the reward network,
    and therefore changes whenever `self.train()` is called.

    Args:
        env (str, Env, or VecEnv): The Env that should be wrapped. If a
            string environment name is given or a Env is given, then we first
            make a VecEnv before continuing.

    Returns:
        wrapped_env (VecEnv): The wrapped environment with a new reward.
    """
    env = maybe_load_env(env, vectorize=True)
    if self.debug_use_ground_truth:
      return env
    else:
      return _RewardVecEnvWrapper(env, self._test_reward_fn)

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
                            gen_new_obs: Optional[np.ndarray] = None
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
    log_act_prob = self.gen_policy.action_probability(old_obs, actions=act,
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

  def _build_policy_train_reward(self):
    """Sets self._policy_train_reward_fn, the reward function to use when
    running a policy optimizer (e.g. PPO).
    """

    def R(old_obs, act, new_obs):
      """Vectorized reward function.

      Args:
          old_obs (array): The observation input. Its shape is
              `((None,) + self.env.observation_space.shape)`.
          act (array): The action input. Its shape is
              `((None,) + self.env.action_space.shape)`. The None dimension is
              expected to be the same as None dimension from `obs_input`.
          new_obs (array): The observation input. Its shape is
              `((None,) + self.env.observation_space.shape)`.
      """
      old_obs = np.atleast_1d(old_obs)
      act = np.atleast_1d(act)
      new_obs = np.atleast_1d(new_obs)

      n_gen = len(old_obs)
      assert len(act) == n_gen
      assert len(new_obs) == n_gen

      # Calculate generator-policy log probabilities.
      log_act_prob = self.gen_policy.action_probability(old_obs, actions=act,
                                                        logp=True)
      assert len(log_act_prob) == n_gen
      log_act_prob = log_act_prob.reshape((n_gen,))

      fd = {
          self.discrim.old_obs_ph: old_obs,
          self.discrim.act_ph: act,
          self.discrim.new_obs_ph: new_obs,
          self.discrim.labels_ph: np.ones(n_gen),
          self.discrim.log_policy_act_prob_ph: log_act_prob,
      }
      rew = self._sess.run(self.discrim.policy_train_reward, feed_dict=fd)
      return rew.flatten()

    self._policy_train_reward_fn = R

  def _build_test_reward(self):
    """Sets self._test_reward_fn, the transfer reward function"""
    def R(old_obs, act, new_obs):
      fd = {
          self.discrim.old_obs_ph: old_obs,
          self.discrim.act_ph: act,
          self.discrim.new_obs_ph: new_obs,
      }
      rew = self._sess.run(self.discrim._policy_test_reward,
                           feed_dict=fd)
      return rew.flatten()

    self._test_reward_fn = R


class _RewardVecEnvWrapper(VecEnvWrapper):

  def __init__(self, venv, reward_fn):
    """A RewardVecEnvWrapper uses a provided reward_fn to replace
    the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization.
    A tricky part about this class keeping track of the most recent
    observation from each environment.

    Args:
        venv (VecEnv): The VecEnv to wrap.
        reward_fn (Callable): A function that wraps takes in arguments
            (old_obs, act, new_obs) and returns a vector of rewards.
    """
    assert not isinstance(venv, _RewardVecEnvWrapper)
    super().__init__(venv)
    self.reward_fn = reward_fn
    self.reset()

  @property
  def envs(self):
    return self.venv.envs

  def reset(self):
    self._old_obs = self.venv.reset()
    return self._old_obs

  def step_async(self, actions):
    self._actions = actions
    return self.venv.step_async(actions)

  def step_wait(self):
    obs, rew, done, info = self.venv.step_wait()
    rew = self.reward_fn(self._old_obs, self._actions, obs)
    # XXX: We never get to see episode end. (See Issue #1).
    # Therefore, the final obs of every episode is incorrect.
    self._old_obs = obs
    return obs, rew, done, info


def _n_steps_if_not_none(n_steps):
  if n_steps is None:
    return {}
  else:
    return dict(n_steps=n_steps)
