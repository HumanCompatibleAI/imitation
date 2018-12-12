import tensorflow as tf
import tqdm

import util


class AIRLTrainer():

    def __init__(self, env, traj_list_expert, policy_net, reward_net):
        """
        Adversarial IRL. After training, we recover reward as a
        function of state and action. (State-only rewards will ignore action).

        Params:
          env (gym.Env) -- A gym environment to train in. AIRL will modify
            env's step() function.

          traj_list_expert (list) -- A list of expert trajectories from `env`.
            We will infer reward from these trajectories.
            TODO: What is the format of trajectories?
            TODO: Might a better format be simply a list of state-action pairs?

          policy_net (type?) -- The policy network. Acts as generator.

          reward_net (type?) -- The reward network to train. Used to
            discriminate generated trajectories from other trajectories.
            Parameterized by both theta (reward state) and phi (reward shaping).
        """
        self._sess = tf.Session()

        self.env = env
        self.traj_list_expert = traj_list_expert
        self.policy_net = policy_net
        self.reward_net = reward_net

        # (We can also load policy from a str policy network name.)
        self.policy_model = PPO1(policy=policy_net, env=env, verbose=1)
        self._build_discriminator()
        self._build_policy_train_reward()


    def _build_discriminator(self):
        """
        Sets self.D, a scalar Tensor returns the probability that
        (s, a, s_prime) is from the expert.

        Also sets self.D_complement, self.log_D, and self.log_D_complement.
        D_complement is the complement of D.
        """
        log_denom = tf.log(tf.exp(self.reward_net.shaped_reward)
            + self.policy_net.action_prob)

        self.log_D = self.reward_net.shaped_reward - log_denom
        self.log_D_complement = tf.log(self.policy_net.action_prob) - log_denom
        self.D = tf.exp(self.log_D)
        self.D_complement = tf.exp(self.log_D_complement)


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
        traj_list_gen = generate_traj_policy(self.env, self.policy_net)
        self._train_D_via_logistic_regress(traj_list_gen)

        self._train_disc(traj_list_gen)
        self._train_gen()


    def _train_disc(self, traj_list_gen):
        # Binary logistic regression over (phi, theta)
        # I'm not sure how to do this yet, but maybe keras has an out of the
        # box solution!
        X_state = []  # List of states.
        X_action = []  # List of action.
        Y = []  # Whether X[i] comes from expert trajectory.
        TODO_bin_log_regress(self.sess, X, Y)


    def _train_gen(self):
        # Adam: It's not necessary to train to convergence.
        # (Probably should take a look at Justin's code for intuit.)
        self.policy_model.learn(10)
