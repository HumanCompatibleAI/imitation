import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper
import tensorflow as tf
from tqdm import tqdm

import summaries
import util


class AIRLTrainer():

    def __init__(self, env, policy, reward_net,
            expert_obs_old, expert_act, expert_obs_new, init_tensorboard=False):
        """
        Adversarial IRL. After training, the RewardNet will have recovered
        the reward.

        Params:
          env (gym.Env or str) -- A gym environment to train in. AIRL will
            modify env's step() function. Internally, we will wrap this
            in a DummyVecEnv.
          policy (stable_baselines.BaseRLModel) --
            The policy acts as generator. We train this policy
            to maximize discriminator confusion.
          reward_net (RewardNet) -- The reward network to train. Used to
            discriminate generated trajectories from other trajectories.
          expert_obs_old (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the expert chooses action
            `expert_act[i]`.
          expert_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          expert_obs_new (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the expert chooses
            action `expert_act[i]`.
          init_tensorboard (bool) -- Make various tensorboard summaries under
            the run name "AIRL_{date}_{runnumber}".
        """
        self._sess = tf.Session()

        self.env = util.maybe_load_env(env, vectorize=True)
        self.policy = policy

        self.reward_net = reward_net
        self.expert_obs_old = expert_obs_old
        self.expert_act = expert_act
        self.expert_obs_new = expert_obs_new
        self.init_tensorboard = init_tensorboard

        self._global_step = tf.train.create_global_step()
        if self.init_tensorboard:
            with tf.name_scope("summaries"):
                self._build_summarize()

        with tf.variable_scope("AIRLTrainer"):
            with tf.variable_scope("discriminator"):
                self._build_disc_train()
            self._build_policy_train_reward()
            self._build_test_reward()

        self._sess.run(tf.global_variables_initializer())

        self.env = self.wrap_env_train_reward(self.env)

    def train_disc(self, *args, n_steps=100, **kwargs):
        """
        Train the discriminator to minimize classification cross-entropy.

        Params:
          expert_obs_old (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the expert chooses action
            `expert_act[i]`.
          expert_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          expert_obs_new (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the expert chooses
            action `expert_act[i]`.
          gen_obs_old (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the generator chooses
            action `gen_act[i]`.
          gen_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          gen_obs_new (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the generator
            chooses action `gen_act[i]`.
          n_steps (int) -- The number of training steps to take.
        """
        fd = self._build_disc_feed_dict(*args, **kwargs)
        for _ in range(n_steps):
            step, _ = self._sess.run([self._global_step, self._disc_train_op],
                    feed_dict=fd)
            if self.init_tensorboard and step % 20 == 0:
                self._summarize(fd, step)

    def train_gen(self, n_steps=100):
        # Adam: It's not necessary to train to convergence.
        # (Probably should take a look at Justin's code for intuit.)
        self.policy.set_env(self.env)  # Can't guarantee that env is the same.
        self.policy.learn(n_steps)

    def train(self, n_epochs=1000):
        for i in tqdm(range(n_epochs)):
            self._train_epoch()

    def eval_disc_loss(self, *args, **kwargs):
        """
        Evaluate the discriminator loss.

        Params:
          expert_obs_old (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the expert chooses action
            `expert_act[i]`.
          expert_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          expert_obs_new (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the expert chooses
            action `expert_act[i]`.
          gen_obs_old (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is the observation seen when the generator chooses
            action `gen_act[i]`.
          gen_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          gen_obs_new (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`. The ith observation
            in this array is from the transition state after the generator
            chooses action `gen_act[i]`.

        Return:
          discriminator_loss (type?) -- The cross-entropy error in the
            discriminator's clasifications.
        """
        fd = self._build_disc_feed_dict(*args, **kwargs)
        return np.sum(self._sess.run(self._disc_loss, feed_dict=fd))

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
        tf.summary.histogram("reward", self.reward_net.reward_output)
        tf.summary.histogram("shaping_old", self.reward_net.old_shaping_output)
        tf.summary.histogram("shaping_new", self.reward_net.new_shaping_output)
        tf.summary.histogram("shaped_reward",
                self.reward_net.shaped_reward_output)
        self._summary_op = tf.summary.merge_all()

    def _summarize(self, fd, step):
        events = self._sess.run(self._summary_op, feed_dict=fd)
        self._summary_writer.add_summary(events, step)

    def _build_disc_train(self):
        # Holds the generator-policy log action probabilities of every
        # state-action pair that the discriminator is being trained on.
        self._log_policy_act_prob_ph = tf.placeholder(shape=(None,),
                dtype=tf.float32, name="log_ro_act_prob_ph")

        # Holds the label of every state-action pair that the discriminator
        # is being trained on. Use 0.0 for expert policy. Use 1.0 for generated
        # policy.
        self._labels_ph = tf.placeholder(shape=(None,), dtype=tf.int32,
                name="log_ro_act_prob_ph")

        # Construct discriminator logits by stacking predicted rewards
        # and log action probabilities.
        self._presoftmax_disc_logits = tf.stack(
                [self.reward_net.shaped_reward_output,
                    self._log_policy_act_prob_ph],
                axis=1, name="presoftmax_discriminator_logits")  # (None, 2)

        # Construct discriminator loss.
        self._disc_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self._labels_ph,
                logits=self._presoftmax_disc_logits,
                name="discrim_loss"
            )

        # Construct Train operation.
        self._disc_opt = tf.train.AdamOptimizer()
        # XXX: I am passing a [None] Tensor as loss. Can this be problematic?
        self._disc_train_op = self._disc_opt.minimize(self._disc_loss,
                global_step=self._global_step)

    def _build_disc_feed_dict(self, expert_obs_old, expert_act, expert_obs_new,
            gen_obs_old, gen_act, gen_obs_new):

        n_expert = len(expert_obs_old)
        n_gen = len(gen_obs_old)
        N = n_expert + n_gen
        assert n_expert == len(expert_act)
        assert n_expert == len(expert_obs_new)
        assert n_gen == len(gen_act)
        assert n_gen == len(gen_obs_new)

        # Concatenate rollouts, and label each row as expert or generator.
        obs_old = np.concatenate([expert_obs_old, gen_obs_old])
        act = np.concatenate([expert_act, gen_act])
        obs_new = np.concatenate([expert_obs_new, gen_obs_new])
        labels = np.concatenate([np.zeros(n_expert, dtype=int),
            np.ones(n_gen, dtype=int)])

        # Calculate generator-policy log probabilities.
        log_act_prob = np.log(util.rollout_action_probability(
            self.policy, obs_old, act))  # (N,)

        fd = {
                self.reward_net.old_obs_ph: obs_old,
                self.reward_net.act_ph: act,
                self.reward_net.new_obs_ph: obs_new,
                self._labels_ph: labels,
                self._log_policy_act_prob_ph: log_act_prob,
            }
        return fd

    def _build_policy_train_reward(self):
        """
        Sets self._policy_train_reward_fn, the reward function to use when
        running a policy optimizer (e.g. PPO).
        """
        # Construct generator reward.
        self._log_softmax_logits = tf.nn.log_softmax(
                self._presoftmax_disc_logits)
        self._log_D, self._log_D_compl = tf.split(
                self._log_softmax_logits, [1, 1], axis=1)
        self._policy_train_reward = self._log_D - self._log_D_compl

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

            log_prob = np.log(
                    util.rollout_action_probability(self.policy, old_obs, act))

            fd = {
                    self.reward_net.old_obs_ph: old_obs,
                    self.reward_net.act_ph: act,
                    self.reward_net.new_obs_ph: new_obs,
                    self._labels_ph: np.ones(n_gen),
                    self._log_policy_act_prob_ph: log_prob,
                }
            return self._sess.run(self._policy_train_reward, feed_dict=fd)

        self._policy_train_reward_fn = R

    def _build_test_reward(self):
        """
        Sets self._test_reward_fn, the reward function learned by AIRL.
        """
        def R(old_obs, act, new_obs):
            fd = {
                self.reward_net.old_obs_ph: old_obs,
                self.reward_net.act_ph: act,
                self.reward_net.new_obs_ph: new_obs,
            }
            return self._sess.run(self.reward_net.reward_output, feed_dict=fd)

        self._test_reward_fn = R

    def _train_epoch(self):
        n_timesteps = len(self.expert_obs_old)
        (gen_obs_old, gen_act, gen_obs_new) = util.generate_rollouts(
                self.policy, self.env, n_timesteps)

        self.train_disc(self.expert_obs_old, self.expert_act,
                self.expert_obs_new, gen_obs_old, gen_act, gen_obs_new)
        self.train_gen()


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

    def reset(self):
        self._old_obs =  self.venv.reset()
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
