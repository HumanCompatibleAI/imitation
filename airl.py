import numpy as np
import tensorflow as tf

from tqdm import tqdm

import util


class AIRLTrainer():

    def __init__(self, env, policy, reward_net,
            expert_obs_old, expert_act, expert_obs_new):
        """
        Adversarial IRL. After training, the RewardNet will have recovered
        the reward.

        Params:
          env (gym.Env or str) -- A gym environment to train in. AIRL will
            modify env's step() function. Internally, we will wrap this
            in a DummyVecEnv.
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
          policy (stable_baselines.BaseRLModel) --
            The policy acts as generator. We train this policy
            to maximize discriminator confusion.
          reward_net (RewardNet) -- The reward network to train. Used to
            discriminate generated trajectories from other trajectories.
        """
        self._sess = tf.Session()

        self.env = util.maybe_load_env(env, vectorize=True)
        self.policy = policy
        policy.set_env(self.env)

        self.reward_net = reward_net
        self.expert_obs_old = expert_obs_old
        self.expert_act = expert_act
        self.expert_obs_new = expert_obs_new

        with tf.variable_scope("AIRLTrainer"):
            with tf.variable_scope("discriminator"):
                self._build_disc_train()
            self._build_policy_train()

        self._sess.run(tf.global_variables_initializer())

        # Even after wrapping the reward, we should still have a
        # VecEnv.
        assert util.is_vec_env(self.env)


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
        # TODO: Maybe add global step for Tensorboard stuff.
        self._disc_train_op = self._disc_opt.minimize(self._disc_loss)


    def _build_policy_train(self):
        """
        Sets self._policy_train_reward_fn, the reward function to use when
        running a policy optimizer (e.g. PPO). Then wraps the environment
        to return this new reward during step().
        """
        # Construct generator reward.
        self._log_softmax_logits = tf.nn.log_softmax(
                self._presoftmax_disc_logits)
        self._log_D, self._log_D_compl = tf.split(
                self._log_softmax_logits, [1, 1], axis=1)
        self._policy_train_reward = self._log_D - self._log_D_compl

        def R(self, old_obs, act, new_obs):
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
            fd = {
                    self.old_obs_ph: old_obs,
                    self.act_ph: act,
                    self.new_obs_ph: new_obs,
                }
            return self.sess.run(self._policy_train_reward, feed_dict=fd)

        self._policy_train_reward_fn = R
        util.reset_and_wrap_env_reward(self.env, self._policy_train_reward_fn)


    def train(self, n_epochs=1000):
        for i in tqdm(range(n_epochs)):
            self._train_epoch()


    def _train_epoch(self):
        n_timesteps = len(self.expert_obs_old)
        (gen_obs_old, gen_act, gen_obs_new) = util.generate_rollouts(
                self.policy, self.env, n_timesteps)

        self.train_disc(self.expert_obs_old, self.expert_act,
                self.expert_obs_new, gen_obs_old, gen_act, eng_obs_new)
        self.train_gen()


    def train_disc(self, expert_obs_old, expert_act, expert_obs_new,
            gen_obs_old, gen_act, gen_obs_new, n_steps=100):

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
        for _ in range(n_steps):
            self._sess.run(self._disc_train_op, feed_dict=fd)


    def train_gen(self, n_steps=100):
        # Adam: It's not necessary to train to convergence.
        # (Probably should take a look at Justin's code for intuit.)
        self.policy.learn(n_steps)
