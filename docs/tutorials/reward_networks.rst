.. _reward-net docs:

===============
Reward Networks
===============

The goal of both inverse reinforcement learning (IRL) algorithms (e.g. :ref:`AIRL <airl docs>`, :ref:`GAIL <gail docs>`) and :ref:`preference comparison <preference comparisons docs>` is to discover a reward function. In imitation these learned rewards are parameterized by reward networks.

Reward Network API
------------------

Reward networks need to support two separate but equally important modes of operation. First, these networks need to produce a reward that can be differentiated through and used for training the reward net. These rewards are provided by the :meth:`forward <imitation.rewards.reward_nets.RewardNet.forward>` method. Second, these networks need to produce a reward that can be used for training policies. These rewards are provided by the :meth:`predict_processed <imitation.rewards.reward_nets.RewardNet.predict_processed>` method which applies additional post processing that is unhelpful during reward net training.


Reward Network Architecture
---------------------------

In imitation reward networks are `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_. Out of the box imitation provides a few reward network architectures:
an multi-layer perceptron :class:`BasicRewardNet <imitation.rewards.reward_nets.BasicRewardNet>` and a convolutional neural net :class:`CNNRewardNet <imitation.rewards.reward_net.CNNRewardNet>`. To implement your own custom reward network you can subclass :class:`RewardNet <imitation.rewards.reward_nets.RewardNet>`.

.. testcode::
    :skipif: skip_doctests

    from imitation.rewards.reward_nets import RewardNet

    class MyRewardNet(RewardNet):
        def __init__(self, observation_space, action_space):
            super().__init__(observation_space, action_space)
            # initialize your custom reward network here

        def forward(self,
            state: th.Tensor, # (batch_size, *obs_shape)
            action: th.Tensor, # (batch_size, *action_shape)
            next_state: th.Tensor, # (batch_size, *obs_shape)
            done: th.Tensor, # (batch_size,)
        ) -> th.Tensor:
            # implement your custom reward network here
            return th.zeros_like(done) # (batch_size,)

Replace an Environments Reward with a Reward Network
----------------------------------------------------

In oder to use a reward network to train a policy, we need to integrate it into an environment. This is done by wrapping the environment in a :class:`RewardVecEnvWrapper <imitation.reward_wrapper.RewardVecEnvWrapper>`. This wrapper will replace the reward network to the environment's reward function.

.. testsetup::
    :skipif: skip_doctests

    import numpy as np
    from unittest.mock import MagicMock
    reward_net = MagicMock()
    reward_net.predict_processed.return_value = np.zeros(3)

.. testcode::
    :skipif: skip_doctests

    from imitation import util
    from imitation.reward_wrapper import RewardVecEnvWrapper

    venv = util.make_vec_env("Pendulum-v1", n_envs=3)
    venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)


Serializing and Deserializing Reward Networks
---------------------------------------------

Reward networks are serialized simply by calling `th.save(reward_net, path)`. When evaluating reward networks we may or may not want to include the wrappers it was trained with. To load the raw reward network we can simply call `th.load(path)`.

When using a learned reward network to train or evaluate a policy we can select whether or not to include the reward network wrappers. This is controlled by the `reward_type` parameter passed to :func:`load_reward <imitation.rewards.serialize>`. For example, we might want to remove the keep or remove the reward normalization fit during training in the evaluation phase.

.. testsetup::
    :skipif: skip_doctests

    from imitation import util
    from gym.spaces import Box
    from imitation.rewards.reward_nets import BasicRewardNet
    from tempfile import TemporaryDirectory

    tempdir = TemporaryDirectory()
    path = tempdir.name + "/reward_net.pt"

.. testcode::
    :skipif: skip_doctests

    import torch as th
    from imitation.rewards.reward_nets import NormalizedRewardNet
    from imitation.rewards.serialize import load_reward

    th.save(reward_net, path)
    reward_net = th.load(path)
    reward_net = NormalizedRewardNet(BasicRewardNet(Box(10), Box(2)))
    # We can also load the reward network as a reward function for use in evaluation
    rew_fn_normalized = load_reward(path, reward_type="RewardNet_normalized")
    rew_fn_unnormalization = load_reward(path, reward_type="RewardNet_unnormalized")


Reward Network Wrappers
-----------------------

Imitation learning algorithms should converge to a reward function that will theoretically induce the optimal/soft optimal policy. However, these reward functions may not be well suited for training RL agents or we may want to modify them to, say, encourage exploration.

Reward network wrappers are used to transform the reward function to be more suitable for training policies. Out of the box imitation provides a few reward network wrappers. The most commonly used is the :ref:`NormalizedRewardNet <imitating.rewards.reward_nets.NormalizedRewardNet>`. This class makes use of a normalization layer to standardize the *output* of the reward function using its running mean an variance. This is useful for stabilizing training and ensuring that the reward function is not drastically changed by the reward network. Note that when a reward network is saved its wrappers are saved along with it so that the normalization fit during reward learning will be used during policy learning or evaluation.

.. note::
    The reward normalization wrapper does _not_ function identically to stable baselines3's `VecNormalize <https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#stable_baselines3.common.vec_env.VecNormalize>`_ environment wrapper. First, it does not normalize the observations. Second it normalizes the reward based on the reward networks's mean and variance on and *not* a running estimate of the return.
