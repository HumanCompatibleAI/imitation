# import gym  # noqa: F401
import pytest
from stable_baselines3 import PPO

from imitation.policies import trainer
from imitation.rewards import common as rewards_common


@pytest.fixture
def agent():
    return PPO("MlpPolicy", env="CartPole-v1")


@pytest.fixture
def reward_fn() -> rewards_common.RewardFn:
    # dummy RewardFn
    return lambda obs, acts, next_obs, dones: dones.astype(float)


@pytest.fixture
def agent_trainer(agent, reward_fn):
    return trainer.AgentTrainer(agent, reward_fn)


def test_missing_environment(agent, reward_fn):
    # Create an agent that doesn't have its environment set.
    # More realistically, this can happen when loading a stored agent.
    agent.env = None
    with pytest.raises(ValueError) as err:
        # not a valid RewardFn but that doesn't matter for this test
        trainer.AgentTrainer(agent, reward_fn)

    assert str(err.value) == "The environment for the agent algorithm must be set."


def test_transitions_left_in_buffer(agent_trainer):
    # Faster to just set the counter than to actually fill the buffer
    # with transitions.
    agent_trainer.buffering_wrapper.n_transitions = 2
    with pytest.raises(RuntimeError) as err:
        agent_trainer.train(steps=1)
    assert str(err.value) == (
        "There are 2 transitions left in the buffer. "
        "Call AgentTrainer.sample() first to clear them."
    )
