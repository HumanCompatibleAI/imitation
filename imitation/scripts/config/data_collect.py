import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import FeedForward64Policy

data_collect_ex = sacred.Experiment("data_collect")


@data_collection_exp.config
def data_collect_defaults():
    data_collect_total_timesteps = int(1e6)
    make_blank_policy_kwargs = DEFAULT_BLANK_POLICY_KWARGS


@data_collect_ex.named_config
def ant(init_trainer_kwargs):
    env_name = "Ant-v2"
    n_epochs = 2000


@data_collect_ex.named_config
def cartpole(init_trainer_kwargs):
    env_name = "CartPole-v1"
    data_collect_total_timesteps = int(4e5)


@data_collect_ex.named_config
def halfcheetah(init_trainer_kwargs):
    env_name = "HalfCheetah-v2"
    n_epochs = 1000

    init_trainer_kwargs.update(dict(
        use_gail=True,
        discrim_kwargs=dict(entropy_weight=0.1),
    ))


@data_collect_ex.named_config
def pendulum(init_trainer_kwargs):
    env_name = "Pendulum-v0"


@data_collect_ex.named_config
def swimmer(init_trainer_kwargs, make_blank_policy_kwargs):
    env_name = "Swimmer-v2"
    n_epochs = 1000
    make_blank_policy_kwargs.update(dict(
        policy_network_class=FeedForward64Policy
    ))
