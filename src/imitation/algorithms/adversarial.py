import logging
import os
import warnings
from typing import Callable, Dict, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.tensorboard as thboard
import tqdm
from stable_baselines3.common import base_class, preprocessing, vec_env

from imitation.data import buffer, datasets, types, wrappers
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper


class AdversarialTrainer:
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    venv_test: vec_env.VecEnv
    """Like `self.venv`, but wrapped with test reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_test` is the same as `self.venv`."""

    def __init__(
        self,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        discrim: discrim_nets.DiscrimNet,
        expert_data: Union[datasets.Dataset[types.Transitions], types.Transitions],
        *,
        log_dir: str = "output/",
        disc_batch_size: int = 2048,
        disc_minibatch_size: int = 256,
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        device: Union[str, th.device] = "auto",
    ):
        """Builds AdversarialTrainer.

        Args:
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
              discriminator confusion. The generator batch size
              `self.gen_batch_size` is inferred from `gen_algo.n_steps`.
            discrim: The discriminator network. This will be moved to the same
              device as `gen_algo`.
            expert_data: Either a `Dataset` of expert `Transitions`, or an instance of
              `Transitions` to be automatically converted into a
              `Dataset[Transitions]`.
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
        self._global_step = 0
        self._disc_step = 0

        assert disc_batch_size % disc_minibatch_size == 0
        assert disc_minibatch_size % 2 == 0, (
            "discriminator minibatch size must be even "
            "(equal split between generator and expert samples)"
        )
        self.disc_batch_size = disc_batch_size
        self.disc_minibatch_size = disc_minibatch_size
        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self.discrim = discrim.to(self.gen_algo.device)
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self.discrim.parameters(), **self._disc_opt_kwargs
        )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_train = self.venv_test = self.venv
        else:
            self.venv_train = reward_wrapper.RewardVecEnvWrapper(
                self.venv, self.discrim.predict_reward_train
            )
            self.venv_test = reward_wrapper.RewardVecEnvWrapper(
                self.venv, self.discrim.predict_reward_test
            )

        self.venv_train_buffering = wrappers.BufferingWrapper(self.venv_train)
        self.venv_train_norm = vec_env.VecNormalize(self.venv_train_buffering)
        self.gen_algo.set_env(self.venv_train_norm)

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_batch_size
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity, self.venv
        )

        if isinstance(expert_data, types.Transitions):
            # Somehow, pytype doesn't recognize that `expert_data` is Transitions.
            expert_data = datasets.TransitionsDictDatasetAdaptor(
                expert_data,  # pytype: disable=wrong-arg-types
            )
        self._expert_dataset = expert_data

        expert_ds_size = self._expert_dataset.size()
        if expert_ds_size is not None and self.disc_batch_size // 2 > expert_ds_size:
            warnings.warn(
                "The discriminator batch size is more than twice the number of "
                "expert samples. This means that we will be reusing expert samples "
                "every discrim batch.",
                category=RuntimeWarning,
            )

    @property
    def gen_batch_size(self) -> int:
        return self.gen_algo.n_steps

    def _gen_log_action_prob_from_unnormalized(
        self, observation: np.ndarray, *, actions: np.ndarray, logp=True,
    ) -> np.ndarray:
        """Calculate generator log action probabilility.

        Params:
          observation: Unnormalized observation.
          actions: action.
        """
        obs = self.venv_train_norm.normalize_obs(observation)
        return self.gen_algo.action_probability(obs, actions=actions, logp=logp)

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
        if self._gen_replay_buffer.size() == 0:
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
    ) -> Dict[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
          gen_samples: Transition samples from the generator policy. If not
            provided, then take `self.disc_batch_size // 2` samples from the
            generator replay buffer. Observations should not be normalized.
          expert_samples: Transition samples from the expert. If not
            provided, then take `n_gen` expert samples from the expert
            dataset, where `n_gen` is the number of samples in `gen_samples`.
            Observations should not be normalized.

        Returns:
           dict: Statistics for discriminator (e.g. loss, accuracy).
        """
        with logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            batch = self._make_disc_train_batch(
                gen_samples=gen_samples, expert_samples=expert_samples
            )
            disc_logits = self.discrim.logits_gen_is_high(
                batch["state"],
                batch["action"],
                batch["next_state"],
                batch["done"],
                batch["log_policy_act_prob"],
            )
            loss = self.discrim.disc_loss(disc_logits, batch["labels_gen_is_one"])

            # do gradient step
            self._disc_opt.zero_grad()
            loss.backward()
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = rew_common.compute_train_stats(
                    disc_logits, batch["labels_gen_is_one"], loss
                )
            logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                logger.record(k, v)
            logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
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
            self.gen_algo.learn(
                total_timesteps=total_timesteps,
                reset_num_timesteps=False,
                **learn_kwargs,
            )
            self._global_step += 1

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
            logger.dump(self._global_step)

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor,
            space,
            # TODO(sam): can I remove "scale" kwarg in DiscrimNet etc.?
            normalize_images=self.discrim.scale,
        )
        return preprocessed

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[types.Transitions] = None,
        expert_samples: Optional[types.Transitions] = None,
    ) -> dict:
        """Build and return training batch for the next discriminator update.

        Args:
          gen_samples: Same as in `train_disc_step`.
          expert_samples: Same as in `train_disc_step`.
        """
        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. " "Call `train_gen()` first."
                )
            gen_samples = self._gen_replay_buffer.sample(self.disc_batch_size // 2)
        n_gen = len(gen_samples.obs)

        if expert_samples is None:
            expert_samples = self._expert_dataset.sample(n_gen)
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
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            acts_th = th.as_tensor(acts, device=self.gen_algo.device)
            _, log_act_prob_th, _ = self.gen_algo.policy.evaluate_actions(
                obs_th, acts_th
            )
            log_act_prob = log_act_prob_th.detach().cpu().numpy()
            del obs_th, acts_th, log_act_prob_th  # unneeded
        assert len(log_act_prob) == n_samples
        log_act_prob = log_act_prob.reshape((n_samples,))

        batch_dict = {
            "state": self._torchify_with_space(obs, self.discrim.observation_space),
            "action": self._torchify_with_space(acts, self.discrim.action_space),
            "next_state": self._torchify_with_space(
                next_obs, self.discrim.observation_space
            ),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
            "log_policy_act_prob": self._torchify_array(log_act_prob),
        }

        return batch_dict


class GAIL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[types.Transitions, datasets.Dataset[types.Transitions]],
        gen_algo: base_class.BaseAlgorithm,
        *,
        # FIXME(sam) pass in discrim net directly; don't ask for kwargs indirectly
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `GAIL` adds on top of its superclass initializer are
        as follows:

        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.

        """
        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetGAIL(
            venv.observation_space, venv.action_space, **discrim_kwargs
        )
        super().__init__(venv, gen_algo, discrim, expert_data, **kwargs)


class AIRL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[types.Transitions, datasets.Dataset[types.Transitions]],
        gen_algo: base_class.BaseAlgorithm,
        *,
        # FIXME(sam): pass in reward net directly, not via _cls and _kwargs
        reward_net_cls: Type[reward_nets.RewardNet] = reward_nets.BasicShapedRewardNet,
        reward_net_kwargs: Optional[Mapping] = None,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Adversarial Inverse Reinforcement Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `AIRL` adds on top of its superclass initializer are
        as follows:

        Args:
            reward_net_cls: Reward network constructor. The reward network is part of
                the AIRL discriminator.
            reward_net_kwargs: Optional keyword arguments to use while constructing
                the reward network.
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetAIRL.
        """
        # TODO(shwang): Maybe offer str=>RewardNet conversion like
        #  stable_baselines3 does with policy classes.
        reward_net_kwargs = reward_net_kwargs or {}
        reward_network = reward_net_cls(
            action_space=venv.action_space,
            observation_space=venv.observation_space,
            # pytype is afraid that we'll directly call RewardNet() which is an abstract
            # class, hence the disable.
            **reward_net_kwargs,  # pytype: disable=not-instantiable
        )

        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetAIRL(reward_network, **discrim_kwargs)
        super().__init__(venv, gen_algo, discrim, expert_data, **kwargs)
