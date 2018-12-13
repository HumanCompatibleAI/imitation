import tensorflow as tf
import tqdm

import util


class AIRLTrainer():

    def __init__(self, env, policy, reward_net,
            expert_rollout_obs, expert_rollout_act):
        """
        Adversarial IRL. After training, the RewardNet will have recovered
        the reward.

        Params:
          env (gym.Env) -- A gym environment to train in. AIRL will modify
            env's step() function.
          expert_rollout_obs (array) -- A numpy array with shape
            `[n_timesteps] + env.observation_space.shape`.
          expert_rollout_act (array) -- A numpy array with shape
            `[n_timesteps] + env.action_space.shape`.
          policy (stable_baselines.BaseRLModel) --
            The policy acts as generator. We train this policy
            to maximize discriminator confusion.
          reward_net (RewardNet) -- The reward network to train. Used to
            discriminate generated trajectories from other trajectories.
        """
        self._sess = tf.Session()

        self.env = util.maybe_load_env(env)
        self.policy = policy
        policy.set_env(self.env)

        self.reward_net = reward_net
        self.expert_rollout_obs = expert_rollout_obs
        self.expert_rollout_act = expert_rollout_act

        with tf.variable_scope("AIRLTrainer"):
            with tf.variable_scope("discriminator"):
                self._build_discriminator()
            self._build_policy_train_reward()


    def _build_discriminator(self):
        """
        Sets self.D, a scalar Tensor returns the probability that
        (s, a, s_prime) is from the expert.

        Also sets self.D_complement, self.log_D, and self.log_D_complement.
        D_complement is the complement of D.

        Every Tensor defined here should have shape (None,), corresponding
        to the number of rollout state-action pairs fed into the reward and
        policy networks.
        """
        # TODO: OK, we have an API problem, because policy.action_probability()
        # might only return probabilities for discrete input spaces. It doesn't
        # take in a (state-action) pair, instead it takes in a state only,
        # and returns the probability of each of several discrete actions, it
        # seems. I don't know how I can make this work for continuous action
        # spaces?

        self.policy_act_prob_ph = tf.placeholder(dtype=tf.float32,
                shape=(None,),
                name="policy_act_prob_ph")
        R = self.reward_net.shaped_reward_output

        # TODO: Double check me... it's easy to get these equations wrong.
        log_denom = tf.log(tf.exp(R) + self.policy_act_prob_ph)

        self.log_D = R - log_denom
        self.log_D_complement = tf.log(self.policy_act_prob_ph) - log_denom
        self.D = tf.exp(self.log_D)
        self.D_complement = tf.exp(self.log_D_complement)
        # TODO: assert that all the outputs above have shape (None,).


    def _build_policy_train_reward(self):
        """
        Sets self._policy_train_reward_fn, the reward function to use when
        running a policy optimizer (e.g. PPO).
        """
        self._policy_train_reward = self.log_D - self.log_D_complement
        def R(self, s, a, s_prime):
            fd = {
                    self.s_ph: s,
                    self.a_ph: a,
                    self.s_prime_ph: s_prime
                    }
            return self.sess.run(self._policy_train_reward, feed_dict=fd)
        self._policy_train_reward_fn = R


    def _wrap_env(self):
        """
        Wrap the environment with the reward net. (Modifies in-place)
        """
        util.reset_and_wrap_env_reward(self.env, self._policy_train_reward)


    def train(self, n_epochs=1000):
        for i in tqdm(range(n_epochs)):
            self._train_epoch()


    def _train_epoch(self):
        traj_list_gen = generate_traj_policy(self.env, self.policy)
        self._train_D_via_logistic_regress(traj_list_gen)

        self._train_disc(traj_list_gen)
        self._train_gen()


    def _train_disc(self, traj_list_gen, n_epochs=100):
        pass


    def _train_gen(self):
        # Adam: It's not necessary to train to convergence.
        # (Probably should take a look at Justin's code for intuit.)
        self.policy_model.learn(10)
