import re

import pytest
from stable_baselines3 import PPO

from imitation.policies import trainer


@pytest.fixture
def agent():
    return PPO("MlpPolicy", env="CartPole-v1")


def reward_fn(obs, acts, next_obs, dones):
    """Dummy RewardFn."""
    # This function is currently not actually called, so ignore it for
    # coverage. Still seems better to return something valid for future
    # tests.
    return dones.astype(float)  # pragma: no cover


@pytest.fixture
def agent_trainer(agent):
    return trainer.AgentTrainer(agent, reward_fn)


def test_missing_environment(agent):
    # Create an agent that doesn't have its environment set.
    # More realistically, this can happen when loading a stored agent.
    agent.env = None
    with pytest.raises(
        ValueError, match="The environment for the agent algorithm must be set."
    ):
        # not a valid RewardFn but that doesn't matter for this test
        trainer.AgentTrainer(agent, reward_fn)


def test_transitions_left_in_buffer(agent_trainer):
    # Faster to just set the counter than to actually fill the buffer
    # with transitions.
    agent_trainer.buffering_wrapper.n_transitions = 2
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "There are 2 transitions left in the buffer. "
            "Call AgentTrainer.sample() first to clear them."
        ),
    ):
        agent_trainer.train(steps=1)
