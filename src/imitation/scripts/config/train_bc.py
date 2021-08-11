import pathlib

import sacred

from imitation.util import util

train_bc_ex = sacred.Experiment("train_bc")


@train_bc_ex.config
def config():
    expert_data_src = None
    expert_data_src_format = None  # Either "trajectory" or "path"
    observation_space = None
    action_space = None
    batch_size = 32
    n_epochs = None  # Number of training epochs (mutually exclusive with n_batches)
    n_batches = None  # Number of training batches (mutually exclusive with n_epochs)
    n_episodes_eval = 50  # Number of rollout episodes in final evaluation.
    # Number of trajectories to use during training, or None to use all.
    n_expert_demos = None
    l2_weight = 3e-5  # L2 regularization weight
    optimizer_kwargs = dict(
        lr=4e-4,
    )
    log_dir = None  # Log directory
    log_interval = 100  # Number of batches in between each training log.
    log_rollouts_n_episodes = 5  # Number of rollout episodes per training log.

    # Parent directory for automatic log_dir
    log_root = pathlib.Path("output", "train_bc")

    env_name = "CartPole-v1"  # Gym environment name used to automatically generate venv
    venv = None
    # Either Sequence[Trajectory] or path to Sequence[Trajectory]
    rollout_hint = None  # Used to generate default `expert_data_src`.


@train_bc_ex.config
def defaults(
    expert_data_src,
    expert_data_src_format,
    env_name,
    venv,
    observation_space,
    action_space,
    rollout_hint,
):
    if expert_data_src is None and expert_data_src_format is None:
        expert_data_src = (
            "data/expert_models/" f"{rollout_hint or 'cartpole'}_0/rollouts/final.pkl"
        )
        expert_data_src_format = "path"
    elif expert_data_src_format is None:
        expert_data_src_format = "path"

    if env_name is not None:
        # Automatically generated from `env_name`, or set to None for no evaluation.
        venv = util.make_vec_env(env_name)

    if venv is not None:
        if observation_space is None:
            # Automatically generated from venv
            observation_space = venv.observation_space

        if action_space is None:
            # Automatically generated from venv
            action_space = venv.action_space


@train_bc_ex.config
def default_train_duration(n_epochs, n_batches):
    if n_epochs is None and n_batches is None:
        n_epochs = 400


@train_bc_ex.config
def paths(log_root, env_name):
    if env_name is None:
        _env_name_part = "unknown_env_name"
    else:
        _env_name_part = env_name.replace("/", "_")

    log_dir = pathlib.Path(log_root) / _env_name_part / util.make_unique_timestamp()
    del _env_name_part


@train_bc_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    rollout_hint = "mountain_car"


@train_bc_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"
    rollout_hint = "seals_mountain_car"


@train_bc_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"


@train_bc_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"
    rollout_hint = "cartpole"


@train_bc_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


@train_bc_ex.named_config
def ant():
    env_name = "Ant-v2"
    rollout_hint = "ant"


@train_bc_ex.named_config
def half_cheetah():
    env_name = "HalfCheetah-v2"
    rollout_hint = "half_cheetah"


@train_bc_ex.named_config
def humanoid():
    env_name = "Humanoid-v2"
    rollout_hint = "humanoid"


@train_bc_ex.named_config
def fast():
    n_batches = 50
    n_episodes_eval = 1
    n_expert_demos = 1
