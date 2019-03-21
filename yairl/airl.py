import logging

import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper
import tensorflow as tf
from tqdm import tqdm

import yairl.summaries as summaries
import yairl.util as util


class AIRLTrainer():

    def __init__(self, env, gen_policy, discrim, expert_policies, *,
                 n_expert_timesteps=4000, init_tensorboard=False):
        """
        Adversarial IRL. After training, the RewardNet will have recovered
        the reward.

        Params:
          env (gym.Env or str) -- A gym environment to train in. AIRL will
            modify env's step() function. Internally, we will wrap this
            in a DummyVecEnv.
          gen_policy (stable_baselines.BaseRLModel) --
            The generator policy that AIRL will train to maximize discriminator
            confusion.
          reward_net (RewardNet) -- The reward network to train. Used to
            discriminate generated trajectories from other trajectories, and
            also holds the inferred reward for transfer learning.
          expert_policies (BaseRLModel or [BaseRLModel]) -- An expert policy
            or a list of expert policies that will be used to generate example
            obs-action-obs triples.

            WARNING:
            Due to the way VecEnvs handle
            episode completion states, the last obs-state-obs triple in every
            episode is omitted. (See GitHub issue #1)
          n_expert_timesteps (int) -- The number of expert obs-action-obs
            triples to generate. If the number of expert policies given doesn't
            divide this number evenly, then the last expert policy will generate
            more timesteps.
          init_tensorboard (bool) -- Make various discriminator tensorboard
            summaries under the run name "AIRL_{date}_{runnumber}". (Generator
            summaries appear under a different runname because they are
            configured by initializing the stable_baselines policy).
        """
        self._sess = tf.Session()
        self.epochs_so_far = 0

        self.env = util.maybe_load_env(env, vectorize=True)
        self.gen_policy = gen_policy
        self.expert_old_obs, self.expert_act, self.expert_new_obs = \
            util.rollout.generate_multiple(
                expert_policies, self.env, n_expert_timesteps)
        self.init_tensorboard = init_tensorboard

        self._global_step = tf.train.create_global_step()

        with tf.variable_scope("AIRLTrainer"):
            with tf.variable_scope("discriminator"):
                self.discrim = discrim

                self._build_disc_train()
            self._build_policy_train_reward()
            self._build_test_reward()

        if self.init_tensorboard:
            with tf.name_scope("summaries"):
                self._build_summarize()

        self._sess.run(tf.global_variables_initializer())

        self.env = self.wrap_env_train_reward(self.env)

    def train_disc(self, *, n_steps=10, **kwargs):
        """
        Train the discriminator to minimize classification cross-entropy.

        The generator rollout parameters of the form "gen_*" are optional,
        but if one is given, then all such parameters must be filled (otherwise
        this method will error). If none of the generator rollout parameters are
        given, then a rollout with the same length as the expert rollout
        is generated on the fly.

        Params:
          n_steps (int) -- The number of training steps to take.
          gen_old_obs (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the generator chooses
            action `gen_act[i]`.
          gen_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          gen_new_obs (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the generator
            chooses action `gen_act[i]`.
        """
        fd = self._build_disc_feed_dict(**kwargs)
        for _ in range(n_steps):
            step, _ = self._sess.run([self._global_step, self._disc_train_op],
                                     feed_dict=fd)
            if self.init_tensorboard and step % 20 == 0:
                self._summarize(fd, step)

    def train_gen(self, n_steps=10000):
        # Can't guarantee that env is the same.
        self.gen_policy.set_env(self.env)
        self.gen_policy.learn(n_steps)

    def train(self, *, n_epochs=100, n_gen_steps_per_epoch=None,
              n_disc_steps_per_epoch=None):
        """
        Train the discriminator and generator against each other.

        Params:
          n_epochs (int) -- The number of epochs to train. Every epoch consists
            of training the discriminator and then training the generator.
          n_disc_steps_per_epoch (int) -- The number of steps to train the
            discriminator every epoch. More precisely, the number of full batch
            Adam optimizer steps to perform.
          n_gen_steps_per_epoch (int) -- The number of steps to train the
            generator every epoch. (ie, the number of timesteps to train in
            `policy.learn(timesteps)`).
        """
        for i in tqdm(range(n_epochs), desc="AIRL train"):
            self.train_disc(**_n_steps_if_not_none(n_disc_steps_per_epoch))
            self.train_gen(**_n_steps_if_not_none(n_gen_steps_per_epoch))
        self.epochs_so_far += n_epochs

    def eval_disc_loss(self, **kwargs):
        """
        Evaluate the discriminator loss.

        The generator rollout parameters of the form "gen_*" are optional,
        but if one is given, then all such parameters must be filled (otherwise
        this method will error). If none of the generator rollout parameters are
        given, then a rollout with the same length as the expert rollout
        is generated on the fly.

        Params:
          gen_old_obs (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the generator chooses
            action `gen_act[i]`.
          gen_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          gen_new_obs (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the generator
            chooses action `gen_act[i]`.

        Return:
          discriminator_loss (type?) -- The cross-entropy error in the
            discriminator's clasifications.
        """
        fd = self._build_disc_feed_dict(**kwargs)
        return np.sum(self._sess.run(self.discrim.disc_loss, feed_dict=fd))

    def wrap_env_train_reward(self, env):
        """
        Returns the given Env wrapped with a reward function that returns
        the AIRL training reward (discriminator confusion).

        The reward network referenced (not copied) into the Env
        wrapper, and therefore the rewards are changed by calls to
        AIRLTrainer.train().

        Params:
        env (str, Env, or VecEnv) -- The Env that we want to wrap. If a
          string environment name is given or a Env is given, then we first
          make a VecEnv before continuing.
        """
        env = util.maybe_load_env(env, vectorize=True)
        return _RewardVecEnvWrapper(env, self._policy_train_reward_fn)

    def wrap_env_test_reward(self, env):
        """
        Returns the given Env wrapped with a reward function that returns
        the reward learned by this AIRLTrainer.

        The reward network referenced (not copied) into the Env
        wrapper, and therefore the rewards are changed by calls to
        AIRLTrainer.train().

        Params:
        env (str, Env, or VecEnv) -- The Env that we want to wrap. If a
          string environment name is given or a Env is given, then we first
          make a VecEnv before continuing.

        Returns:
        env
        """
        env = util.maybe_load_env(env, vectorize=True)
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
        self._disc_opt = tf.train.AdamOptimizer()
        # XXX: I am passing a [None] Tensor as loss. Can this be problematic?
        self._disc_train_op = self._disc_opt.minimize(self.discrim.disc_loss,
                                                      global_step=self._global_step)

    def _build_disc_feed_dict(self, gen_old_obs=None, gen_act=None,
                              gen_new_obs=None):

        none_count = sum(int(x is None)
                         for x in (gen_old_obs, gen_act, gen_new_obs))
        if none_count == 3:
            logging.debug("_build_disc_feed_dict: No generator rollout "
                          "parameters were "
                          "provided, so we are generating them now.")
            n_timesteps = len(self.expert_old_obs)
            (gen_old_obs, gen_act, gen_new_obs, _) = util.rollout.generate(
                self.gen_policy, self.env, n_timesteps=n_timesteps)
        elif none_count != 0:
            raise ValueError("Gave some but not all of the generator params.")

        # Alias saved expert rollout.
        expert_old_obs = self.expert_old_obs
        expert_act = self.expert_act
        expert_new_obs = self.expert_new_obs

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
        log_act_prob = np.log(self.gen_policy.action_probability(
            old_obs, actions=act)).flatten()  # (N,)
        assert len(log_act_prob) == N

        fd = {
            self.discrim.old_obs_ph: old_obs,
            self.discrim.act_ph: act,
            self.discrim.new_obs_ph: new_obs,
            self.discrim.labels_ph: labels,
            self.discrim.log_policy_act_prob_ph: log_act_prob,
        }
        return fd

    def _build_policy_train_reward(self):
        """
        Sets self._policy_train_reward_fn, the reward function to use when
        running a policy optimizer (e.g. PPO).
        """

        def R(old_obs, act, new_obs):
            """
            Vectorized reward function.

            Params:
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
            log_act_prob = np.log(self.gen_policy.action_probability(
                old_obs, actions=act)).flatten()  # (N,)
            assert len(log_act_prob) == n_gen

            fd = {
                self.discrim.old_obs_ph: old_obs,
                self.discrim.act_ph: act,
                self.discrim.new_obs_ph: new_obs,
                self.discrim.labels_ph: np.ones(n_gen),
                self.discrim.log_policy_act_prob_ph: log_act_prob,
            }
            rew = self._sess.run(
                self.discrim.policy_train_reward, feed_dict=fd)
            return rew.flatten()

        self._policy_train_reward_fn = R

    def _build_test_reward(self):
        """
        Sets self._test_reward_fn, the reward function learned by AIRL.
        """
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
        """
        A RewardVecEnvWrapper uses a provided reward_fn to replace
        the reward function returned by step().

        Automatically resets the inner VecEnv upon initialization.

        A tricky part about this class keeping track of the most recent
        observation from each environment.

        Params:
          venv -- The VirtualEnv to wrap.
          reward_fn -- A function that wraps takes in arguments
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
