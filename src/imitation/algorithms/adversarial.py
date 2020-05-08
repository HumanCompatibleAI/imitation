import os
from functools import partial
from typing import Callable, Optional, Sequence
from warnings import warn

import numpy as np
import tensorflow as tf
import tqdm
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecNormalize

from imitation.data import buffer, rollout, types, wrappers
from imitation.rewards import discrim_net
from imitation.rewards.reward_net import BasicShapedRewardNet
from imitation.util import logger, reward_wrapper, util


class AdversarialTrainer:
    """Trainer for GAIL and AIRL."""

    venv: VecEnv
    """The original vectorized environment."""

    venv_train: VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    venv_test: VecEnv
    """Like `self.venv`, but wrapped with test reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_test` is the same as `self.venv`."""

    def __init__(
        self,
        venv: VecEnv,
        gen_policy: BaseRLModel,
        discrim: discrim_net.DiscrimNet,
        expert_demos: types.Transitions,
        *,
        log_dir: str = "output/",
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
            log_dir: Directory to store TensorBoard logs, plots, etc. in.
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

              By default this is equal to `self.gen_batch_size`, meaning that we
              sample only from the most recent batch of generator samples.
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
        assert (
            logger.is_configured()
        ), "Requires call to imitation.util.logger.configure"
        self._sess = tf.get_default_session()
        self._global_step = tf.train.create_global_step()

        assert disc_batch_size % disc_minibatch_size == 0
        assert disc_minibatch_size % 2 == 0, (
            "discriminator minibatch size must be even "
            "(equal split between generator and expert samples)"
        )
        self.disc_batch_size = disc_batch_size
        self.disc_minibatch_size = disc_minibatch_size

        self.debug_use_ground_truth = debug_use_ground_truth

        self.venv = venv
        self._expert_demos = expert_demos
        self._gen_policy = gen_policy

        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self._discrim = discrim
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._build_graph()
        self._sess.run(tf.global_variables_initializer())

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.reward_train = self.reward_test = None
            self.venv_train = self.venv_test = self.venv
        else:
            self.reward_train = partial(
                self.discrim.reward_train,
                # The generator policy uses normalized observations
                # but the reward function (self.reward_train) and discriminator use
                # and receive unnormalized observations. Therefore to get the right
                # log action probs for AIRL's ent bonus, we need to normalize obs.
                gen_log_prob_fn=self._gen_log_action_prob_from_unnormalized,
            )
            self.reward_test = self.discrim.reward_test
            self.venv_train = reward_wrapper.RewardVecEnvWrapper(
                self.venv, self.reward_train
            )
            self.venv_test = reward_wrapper.RewardVecEnvWrapper(
                self.venv, self.reward_test
            )

        self.venv_train_buffering = wrappers.BufferingWrapper(self.venv_train)
        self.venv_train_norm = VecNormalize(self.venv_train_buffering)
        self.gen_policy.set_env(self.venv_train_norm)

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_batch_size
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity, self.venv
        )
        self._exp_replay_buffer = buffer.ReplayBuffer.from_data(expert_demos)
        if self.disc_batch_size // 2 > len(self._exp_replay_buffer):
            warn(
                "The discriminator batch size is more than twice the number of "
                "expert samples. This means that we will be reusing samples every "
                "discrim batch."
            )

    @property
    def gen_batch_size(self) -> int:
        return self.gen_policy.n_batch

    @property
    def discrim(self) -> discrim_net.DiscrimNet:
        """Discriminator being trained, used to compute reward for policy."""
        return self._discrim

    @property
    def expert_demos(self) -> types.Transitions:
        """The expert demonstrations that are being imitated."""
        return self._expert_demos

    @property
    def gen_policy(self) -> BaseRLModel:
        """Policy (i.e. the generator) being trained."""
        return self._gen_policy

    def _gen_log_action_prob_from_unnormalized(
        self, observation: np.ndarray, *, actions: np.ndarray, logp=True,
    ) -> np.ndarray:
        """Calculate generator log action probabilility.

        Params:
          observation: Unnormalized observation.
          actions: action.
        """
        obs = self.venv_train_norm.normalize_obs(observation)
        return self.gen_policy.action_probability(obs, actions=actions, logp=logp)

    def train_disc(self, n_samples: Optional[int] = None) -> None:
        """Trains the discriminator to minimize classification cross-entropy.

        Must call `train_gen` first (otherwise there will be no saved generator
        samples for training, and will error).

        Args:
          n_samples: A number of transitions to sample from the generator
            replay buffer and the expert demonstration dataset. (Half of the
            samples are from each source). By default, `self.disc_batch_size`.
            `n_samples` must be a positive multiple of `self.disc_minibatch_size`.
        """
        if len(self._gen_replay_buffer) == 0:
            raise RuntimeError(
                "No generator samples for training. " "Call `train_gen()` first."
            )

        if n_samples is None:
            n_samples = self.disc_batch_size
        n_updates = n_samples // self.disc_minibatch_size
        assert n_samples % self.disc_minibatch_size == 0
        assert n_updates >= 1
        for _ in range(n_updates):
            gen_samples = self._gen_replay_buffer.sample(self.disc_minibatch_size)
            self.train_disc_step(gen_samples=gen_samples)

    def train_disc_step(
        self,
        *,
        gen_samples: Optional[types.Transitions] = None,
        expert_samples: Optional[types.Transitions] = None,
    ) -> None:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
          gen_samples: Transition samples from the generator policy. If not
            provided, then take `self.disc_batch_size // 2` samples from the
            generator replay buffer. Observations should not be normalized.
          expert_samples: Transition samples from the expert. If not
            provided, then take `n_gen` expert samples from the expert
            dataset, where `n_gen` is the number of samples in `gen_samples`.
            Observations should not be normalized.
        """
        with logger.accumulate_means("disc"):
            fetches = {
                "train_op_out": self._disc_train_op,
                "train_stats": self._discrim.train_stats,
            }
            # optionally write TB summaries for collected ops
            step = self._sess.run(self._global_step)
            write_summaries = self._init_tensorboard and step % 20 == 0
            if write_summaries:
                fetches["events"] = self._summary_op

            # do actual update
            fd = self._build_disc_feed_dict(
                gen_samples=gen_samples, expert_samples=expert_samples
            )
            fetched = self._sess.run(fetches, feed_dict=fd)

            if write_summaries:
                self._summary_writer.add_summary(fetched["events"], fetched["step"])

            logger.logkv("step", step)
            for k, v in fetched["train_stats"].items():
                logger.logkv(k, v)
            logger.dumpkvs()

    def eval_disc_loss(self, **kwargs) -> float:
        """Evaluates the discriminator loss.

        Args:
          gen_samples (Optional[rollout.Transitions]): Same as in `train_disc_step`.
          expert_samples (Optional[rollout.Transitions]): Same as in
            `train_disc_step`.

        Returns:
          The total cross-entropy error in the discriminator's classification.
        """
        fd = self._build_disc_feed_dict(**kwargs)
        return np.mean(self._sess.run(self.discrim.disc_loss, feed_dict=fd))

    def train_gen(
        self, total_timesteps: Optional[int] = None, learn_kwargs: Optional[dict] = None
    ):
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
          total_timesteps: The number of transitions to sample from
            `self.venv_train_norm` during training. By default,
            `self.gen_batch_size`.
          learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
            method.
        """
        if total_timesteps is None:
            total_timesteps = self.gen_batch_size
        if learn_kwargs is None:
            learn_kwargs = {}

        with logger.accumulate_means("gen"):
            self.gen_policy.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                **learn_kwargs,
            )

        gen_samples = self.venv_train_norm.pop_transitions()
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self, total_timesteps: int, callback: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every epoch consists of a call to `train_gen(self.gen_batch_size)`,
        a call to `train_disc(self.disc_batch_size)`, and
        finally a call to `callback(epoch)`.

        Training ends once an additional epoch would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Params:
          total_timesteps: An upper bound on the number of transitions to sample
            from the environment during training.
          callback: A function called at the end of every epoch which takes in a
            single argument, the epoch number. Epoch numbers are in
            `range(total_timesteps // self.gen_batch_size)`.
        """
        n_epochs = total_timesteps // self.gen_batch_size
        assert n_epochs >= 1, (
            "No updates (need at least "
            f"{self.gen_batch_size} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for epoch in tqdm.tqdm(range(0, n_epochs), desc="epoch"):
            self.train_gen(self.gen_batch_size)
            self.train_disc(self.disc_batch_size)
            if callback:
                callback(epoch)
            logger.dumpkvs()

    def _build_graph(self):
        # Build necessary parts of the TF graph. Most of the real action happens in
        # constructors for self.discrim and self.gen_policy.
        with tf.variable_scope("trainer"):
            with tf.variable_scope("discriminator"):
                disc_opt = self._disc_opt_cls(**self._disc_opt_kwargs)
                self._disc_train_op = disc_opt.minimize(
                    tf.reduce_mean(self.discrim.disc_loss),
                    global_step=self._global_step,
                )

        if self._init_tensorboard:
            with tf.name_scope("summaries"):
                tf.logging.info("building summary directory at " + self._log_dir)
                graph = self._sess.graph if self._init_tensorboard_graph else None
                summary_dir = os.path.join(self._log_dir, "summary")
                os.makedirs(summary_dir, exist_ok=True)
                self._summary_writer = tf.summary.FileWriter(summary_dir, graph=graph)
                self._summary_op = tf.summary.merge_all()

    def _build_disc_feed_dict(
        self,
        *,
        gen_samples: Optional[types.Transitions] = None,
        expert_samples: Optional[types.Transitions] = None,
    ) -> dict:
        """Build and return feed dict for the next discriminator training update.

        Args:
          gen_samples: Same as in `train_disc_step`.
          expert_samples: Same as in `train_disc_step`.
        """
        if gen_samples is None:
            if len(self._gen_replay_buffer) == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first."
                )
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

        # Policy and reward network were trained on normalized observations.
        expert_obs_norm = self.venv_train_norm.normalize_obs(expert_samples.obs)
        gen_obs_norm = self.venv_train_norm.normalize_obs(gen_samples.obs)

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_obs_norm, gen_obs_norm])
        acts = np.concatenate([expert_samples.acts, gen_samples.acts])
        next_obs = np.concatenate([expert_samples.next_obs, gen_samples.next_obs])
        dones = np.concatenate([expert_samples.dones, gen_samples.dones])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )

        # Calculate generator-policy log probabilities.
        log_act_prob = self._gen_policy.action_probability(obs, actions=acts, logp=True)
        assert len(log_act_prob) == n_samples
        log_act_prob = log_act_prob.reshape((n_samples,))

        fd = {
            self.discrim.obs_ph: obs,
            self.discrim.act_ph: acts,
            self.discrim.next_obs_ph: next_obs,
            self.discrim.done_ph: dones,
            self.discrim.labels_gen_is_one_ph: labels_gen_is_one,
            self.discrim.log_policy_act_prob_ph: log_act_prob,
        }
        return fd


def init_trainer(
    env_name: str,
    expert_trajectories: Sequence[types.Trajectory],
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
    """Builds an AdversarialTrainer, ready to be trained on expert demonstrations.

    Args:
      env_name: The string id of a gym environment.
      expert_trajectories: Demonstrations from expert.
      seed: Random seed.
      log_dir: Directory for logging output. Will generate a unique sub-directory
          within this directory for all output.
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
    logger.configure(folder=log_dir, format_strs=["tensorboard", "stdout"])
    env = util.make_vec_env(
        env_name,
        num_vec,
        seed=seed,
        parallel=parallel,
        log_dir=log_dir,
        max_episode_steps=max_episode_steps,
    )
    gen_policy = util.init_rl(env, verbose=1, **init_rl_kwargs)

    if use_gail:
        discrim = discrim_net.DiscrimNetGAIL(
            env.observation_space, env.action_space, scale=scale, **discrim_kwargs
        )
    else:
        rn = BasicShapedRewardNet(
            env.observation_space, env.action_space, scale=scale, **reward_kwargs
        )
        discrim = discrim_net.DiscrimNetAIRL(
            rn, entropy_weight=airl_entropy_weight, **discrim_kwargs
        )

    expert_demos = rollout.flatten_trajectories(expert_trajectories)
    trainer = AdversarialTrainer(
        env, gen_policy, discrim, expert_demos, log_dir=log_dir, **trainer_kwargs
    )
    return trainer
