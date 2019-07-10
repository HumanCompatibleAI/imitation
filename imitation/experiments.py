import sacred

# NOTE: To call `train_exp.run()`, must import `imitation.scripts.train`.
# A quick way to do this is to `from imitation.scripts.train import train_exp`.
train_exp = sacred.Experiment("train", interactive=True)

# NOTE: To call `data_collection.run()`, must import
# `imitation.scripts.data_collection`.
data_collection_exp = sacred.Experiment("data_collection")


DEFAULT_BLANK_POLICY_KWARGS = dict(
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    # WARNING: this is actually 8*256=2048 due to 8 vector environments
    n_steps=256,
)


@train_exp.config
def train_defaults():
    n_epochs = 50
    n_disc_steps_per_epoch = 50
    n_gen_steps_per_epoch = 2048
    policy_dir = "expert_models"

    load_policy_kwargs = dict(n_experts=1)

    make_blank_policy_kwargs = DEFAULT_BLANK_POLICY_KWARGS

    init_trainer_kwargs = dict(
        use_random_expert=False,
        num_vec=8,  # NOTE: changing this also changes the effective nsteps!
        reward_kwargs=dict(
            theta_units=[32, 32],
            phi_units=[32, 32],
        ),

        # Some environments (e.g. CartPole) have float max as limits, which
        # breaks the scaling.
        trainer_kwargs=dict(
            n_disc_samples_per_buffer=1000,
            # Setting buffer capacity and disc samples to 1000 effectively
            # disables the replay buffer. This seems to improve convergence
            # speed, but may come at a cost of stability.
            gen_replay_buffer_capacity=1000,
            n_expert_samples=1000,
        ),

        discrim_scale=False,
    )


@train_exp.named_config
def ant_airl(init_trainer_kwargs):
    env = "Ant-v2"
    n_epochs = 2000

    init_trainer_kwargs.update(dict(
        use_gail=False,
    ))


@train_exp.named_config
def ant_gail(init_trainer_kwargs):
    env = "Ant-v2"
    n_epochs = 2000
    init_trainer_kwargs.update(dict(
        use_gail=True,
    ))


@train_exp.named_config
def cartpole_airl(init_trainer_kwargs):
    env = "CartPole-v1"
    init_trainer_kwargs.update(dict(
        use_gail=False,
    ))


@train_exp.named_config
def cartpole_gail(init_trainer_kwargs):
    env = "CartPole-v1"
    init_trainer_kwargs.update(dict(
        use_gail=True,
    ))


@train_exp.named_config
def cartpole_orig_airl_repro(make_blank_policy_kwargs,
                             init_trainer_kwargs,
                             load_policy_kwargs):
    env = "CartPole-v1"
    policy_dir = "data"
    n_epochs = 100
    n_disc_steps_per_epoch = 10
    n_gen_steps_per_epoch = 10000

    make_blank_policy_kwargs.update(dict(
        learning_rate=3e-3,
        nminibatches=32,
        noptepochs=10,
        n_steps=2048,
    ))

    init_trainer_kwargs.update(dict(
        use_gail=False,
        use_random_expert=False,
    ))

    load_policy_kwargs.update(n_experts=5)


@train_exp.named_config
def halfcheetah_airl(init_trainer_kwargs):
    env = "HalfCheetah-v2"
    n_epochs = 250

    init_trainer_kwargs.update(dict(
        use_gail=False,
        discrim_kwargs=dict(entropy_weight=0.1),
    ))


@train_exp.named_config
def halfcheetah_gail(init_trainer_kwargs):
    env = "HalfCheetah-v2"
    n_epochs = 1000

    init_trainer_kwargs.update(dict(
        use_gail=True,
        discrim_kwargs=dict(entropy_weight=0.1),
    ))


@train_exp.named_config
def pendulum_airl(init_trainer_kwargs):
    env = "Pendulum-v0"
    init_trainer_kwargs.update(dict(
        use_gail=False,
    ))


@train_exp.named_config
def pendulum_gail(init_trainer_kwargs):
    env = "Pendulum-v0"
    init_trainer_kwargs.update(dict(
        use_gail=True,
    ))


@train_exp.named_config
def swimmer_airl(init_trainer_kwargs):
    env = "Swimmer-v2"
    n_epochs = 1000
    init_trainer_kwargs(dict(
        use_gail=False
    ))


@train_exp.named_config
def swimmer_gail(init_trainer_kwargs):
    env = "Swimmer-v2"
    n_epochs = 1000
    init_trainer_kwargs(dict(
        use_gail=True,
    ))


@train_exp.named_config
def debug_train():
    """Config for tests."""


@data_collection_exp.config
def data_collect_defaults():
    make_blank_policy_kwargs = DEFAULT_BLANK_POLICY_KWARGS


@data_collection_exp.named_config
def ant_data_collect():
    env_name = "Pendulum"
    total_timesteps = int(1e6)


@data_collection_exp.named_config
def cartpole_data_collect():
    env_name = "CartPole-v2"
    total_timesteps = int(4e5)


@data_collection_exp.named_config
def halfcheetah_data_collect():
    env_name = "HalfCheetah-v2",
    total_timesteps = int(1e6),


@data_collection_exp.named_config
def pendulum_data_collect():
    env_name = "Pendulum-v0",
    total_timesteps = int(1e6),


@data_collection_exp.named_config
def swimmer_data_collect(make_blank_policy_kwargs):
    from imitation.util import FeedForward64Policy
    env_name = "Swimmer-v2",
    total_timesteps = int(1e6),
    make_blank_policy_kwargs.update(dict(
        policy_network_class=FeedForward64Policy
    ))


@data_collection_exp.named_config
def debug_data_collection():
    """Config for tests."""
