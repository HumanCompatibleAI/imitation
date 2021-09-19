"""Configuration for imitation.scripts.train_adversarial."""

import os

import sacred

from imitation.scripts.common import reward, rl, train
from imitation.util import util

train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    interactive=True,
    ingredients=[train.train_ingredient, rl.rl_ingredient, reward.reward_ingredient],
)


@train_adversarial_ex.config
def train_defaults():
    env_name = "seals/CartPole-v0"  # environment to train on
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    total_timesteps = 1e6  # Num of environment transitions to sample
    algorithm = "gail"  # Either "airl" or "gail"

    # Number of environments in VecEnv, must evenly divide gen_batch_size
    num_vec = 8

    # Use SubprocVecEnv rather than DummyVecEnv (generally faster if num_vec>1)
    parallel = True
    max_episode_steps = None  # Set to positive int to limit episode horizons

    # Kwargs for initializing GAIL and AIRL
    algorithm_kwargs = dict(
        shared=dict(
            demo_batch_size=1024,  # Number of expert samples per discriminator update
            # Number of discriminator updates after each round of generator updates
            n_disc_updates_per_round=4,
        ),
        airl={},
        gail={},
    )

    # Custom reward network
    reward_net_cls = None
    reward_net_kwargs = None

    log_root = os.path.join("output", "train_adversarial")  # output directory
    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)
    rollout_hint = None  # Used to generate default rollout_path
    data_dir = "data/"  # Default data directory


@train_adversarial_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, rl):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.

    algorithm_kwargs["adversarial"]["gen_replay_buffer_capacity"] = rl["batch_size"]


@train_adversarial_ex.config
def paths(env_name, log_root, rollout_hint, data_dir, train):
    log_dir = os.path.join(
        log_root,
        env_name.replace("/", "_"),
        util.make_unique_timestamp(),
    )


# Training algorithm named configs


@train_adversarial_ex.named_config
def gail():
    """Quick alias for algorithm=gail."""
    algorithm = "gail"


@train_adversarial_ex.named_config
def airl():
    """Quick alias for algorithm=airl."""
    algorithm = "airl"


# Shared settings

MUJOCO_SHARED_LOCALS = dict(rl_kwargs=dict(ent_coef=0.1))

ANT_SHARED_LOCALS = dict(
    total_timesteps=3e7,
    max_episode_steps=500,  # To match `inverse_rl` settings.
    algorithm_kwargs=dict(shared=dict(demo_batch_size=8192)),
    gen_batch_size=16384,
)


# Classic RL Gym environment named configs


@train_adversarial_ex.named_config
def acrobot():
    env_name = "Acrobot-v1"
    algorithm_kwargs = {"shared": {"allow_variable_horizon": True}}
    rollout_hint = "acrobot"


@train_adversarial_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    rollout_hint = "cartpole"
    algorithm_kwargs = {"shared": {"allow_variable_horizon": True}}


@train_adversarial_ex.named_config
def seals_cartpole():
    total_timesteps = 1.4e6
    env_name = "seals/CartPole-v0"
    # seals and vanilla CartPole have the same expert trajectories.
    rollout_hint = "cartpole"


@train_adversarial_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    algorithm_kwargs = {"shared": {"allow_variable_horizon": True}}
    rollout_hint = "mountain_car"


@train_adversarial_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"
    rollout_hint = "mountain_car"  # TODO(shwang): Use seals/MountainCar-v0 rollouts.


@train_adversarial_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"
    rollout_hint = "pendulum"


# Standard MuJoCo Gym environment named configs


@train_adversarial_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "seals/Ant-v0"
    rollout_hint = "ant"


HALF_CHEETAH_SHARED_LOCALS = dict(
    env_name="HalfCheetah-v2",
    rollout_hint="half_cheetah",
    gen_batch_size=16384,
    rl_kwargs=dict(
        batch_size=1024,
    ),
    algorithm_kwargs=dict(
        shared=dict(
            # Number of discriminator updates after each round of generator updates
            n_disc_updates_per_round=16,
            # Equivalent to no replay buffer if batch size is the same
            gen_replay_buffer_capacity=16384,
            demo_batch_size=8192,
        ),
        airl=dict(
            reward_net_kwargs=dict(
                reward_hid_sizes=(32,),
                potential_hid_sizes=(32,),
            ),
        ),
    ),
)


@train_adversarial_ex.named_config
def half_cheetah_gail():
    # TODO(shwang): Update experiment scripts to use different total_timesteps
    # for GAIL and AIRL
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**HALF_CHEETAH_SHARED_LOCALS)
    algorithm = "gail"
    total_timesteps = 8e6


@train_adversarial_ex.named_config
def half_cheetah_airl():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**HALF_CHEETAH_SHARED_LOCALS)
    algorithm = "airl"
    total_timesteps = 5e6


@train_adversarial_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "seals/Hopper-v0"
    rollout_hint = "hopper"


@train_adversarial_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "seals/Humanoid-v0"
    rollout_hint = "humanoid"
    total_timesteps = 4e6


@train_adversarial_ex.named_config
def reacher():
    env_name = "Reacher-v2"
    # TODO(adam): bc doesn't need allow_variable_horizon. Could add it but as a no-op
    # for consistency (could check demos, maybe?), or move this out of shared (but it
    # might be used places that aren't just adversarial... and it is something that
    # appears in base imitation algorithm.)
    algorithm_kwargs = {"shared": {"allow_variable_horizon": True}}
    rollout_hint = "reacher"


@train_adversarial_ex.named_config
def seals_swimmer():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "seals/Swimmer-v0"
    rollout_hint = "swimmer"
    total_timesteps = 2e6


@train_adversarial_ex.named_config
def seals_walker():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "seals/Walker2d-v0"
    rollout_hint = "walker"


# Custom Gym environment named configs


@train_adversarial_ex.named_config
def two_d_maze():
    locals().update(**MUJOCO_SHARED_LOCALS)
    env_name = "imitation/TwoDMaze-v0"
    rollout_hint = "two_d_maze"


@train_adversarial_ex.named_config
def custom_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    # Watch out -- ANT_SHARED_LOCALS could erroneously erase nested dict keys from
    # MUJOCO_SHARED_LOCALS because `locals().update()` doesn't merge dicts
    # "Sacred-style".
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "imitation/CustomAnt-v0"
    rollout_hint = "custom_ant"


@train_adversarial_ex.named_config
def disabled_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    env_name = "imitation/DisabledAnt-v0"
    rollout_hint = "disabled_ant"


# Debug configs


@train_adversarial_ex.named_config
def fast():
    """Minimize the amount of computation.

    Useful for test cases.
    """
    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    algorithm_kwargs = dict(
        shared=dict(
            demo_batch_size=1,
            n_disc_updates_per_round=4,
        ),
    )
    parallel = False  # easier to debug with everything in one process
    max_episode_steps = 5
    num_vec = 2
