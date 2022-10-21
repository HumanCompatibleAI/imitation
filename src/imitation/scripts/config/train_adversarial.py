"""Configuration for imitation.scripts.train_adversarial."""

import sacred
from torch import nn

from imitation.rewards import reward_nets
from imitation.scripts.common import common, demonstrations, expert, reward, rl, train

train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    ingredients=[
        common.common_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        train.train_ingredient,
        expert.expert_ingredient,
    ],
)


@train_adversarial_ex.config
def defaults():
    show_config = False

    total_timesteps = int(1e6)  # Num of environment transitions to sample
    algorithm_kwargs = dict(
        demo_batch_size=1024,  # Number of expert samples per discriminator update
        n_disc_updates_per_round=4,  # Num discriminator updates per generator round
    )
    algorithm_specific = {}  # algorithm_specific[algorithm] is merged with config

    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)
    agent_path = None  # Path to load agent from, optional.


@train_adversarial_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, rl):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.
    algorithm_kwargs["gen_replay_buffer_capacity"] = rl["batch_size"]


# Shared settings

MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))

ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    algorithm_kwargs=dict(shared=dict(demo_batch_size=8192)),
    rl=dict(batch_size=16384),
)


# Classic RL Gym environment named configs


@train_adversarial_ex.named_config
def acrobot():
    env_name = "Acrobot-v1"
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")
    total_timesteps = int(1.4e6)


@train_adversarial_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_adversarial_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")


# Standard MuJoCo Gym environment named configs


@train_adversarial_ex.named_config
def seals_ant():
    # locals().update(**MUJOCO_SHARED_LOCALS)
    # locals().update(**ANT_SHARED_LOCALS)
    common = dict(env_name="seals/Ant-v0")
    demonstrations = dict(
        rollout_path="/home/taufeeque/imitation/output/train_experts/"
        "2022-09-05T18:27:27-07:00/seals_ant_1/rollouts/final.pkl",
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=16,
            clip_range=0.3,
            ent_coef=3.1441389214159857e-06,
            gae_lambda=0.8,
            gamma=0.995,
            learning_rate=0.00017959211641976886,
            max_grad_norm=0.9,
            n_epochs=10,
            # policy_kwargs are same as the defaults
            vf_coef=0.4351450387648799,
        ),
    )


CHEETAH_SHARED_LOCALS = dict(
    MUJOCO_SHARED_LOCALS,
    rl=dict(batch_size=16384, rl_kwargs=dict(batch_size=1024)),
    algorithm_specific=dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    ),
    reward=dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    ),
    algorithm_kwargs=dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    ),
)


@train_adversarial_ex.named_config
def half_cheetah():
    locals().update(**CHEETAH_SHARED_LOCALS)
    common = dict(env_name="HalfCheetah-v2")


@train_adversarial_ex.named_config
def seals_half_cheetah():
    # locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/HalfCheetah-v0")
    demonstrations = dict(
        rollout_path="/home/taufeeque/imitation/output/train_experts/"
        "2022-09-05T18:27:27-07:00/seals_half_cheetah_1/rollouts/final.pkl",
    )
    rl = dict(
        batch_size=512,
        rl_kwargs=dict(
            batch_size=64,
            clip_range=0.1,
            ent_coef=3.794797423594763e-06,
            gae_lambda=0.95,
            gamma=0.95,
            learning_rate=0.0003286871805949382,
            max_grad_norm=0.8,
            n_epochs=5,
            vf_coef=0.11483689492120866,
        ),
    )
    # algorithm_specific = dict(
    #     airl=dict(total_timesteps=int(5e6)),
    #     gail=dict(total_timesteps=int(8e6)),
    # )
    # reward = dict(
    #     algorithm_specific=dict(
    #         airl=dict(
    #             net_cls=reward_nets.BasicShapedRewardNet,
    #             net_kwargs=dict(
    #                 reward_hid_sizes=(32,),
    #                 potential_hid_sizes=(32,),
    #             ),
    #         ),
    #     ),
    # )
    algorithm_kwargs = dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=512,
        demo_batch_size=8192,
    )


@train_adversarial_ex.named_config
def seals_hopper():
    # locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Hopper-v0")
    demonstrations = dict(
        rollout_path="/home/taufeeque/imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_hopper_2/rollouts/final.pkl",
    )
    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=512,
            clip_range=0.1,
            ent_coef=0.0010159833764878474,
            gae_lambda=0.98,
            gamma=0.995,
            learning_rate=0.0003904770450788824,
            max_grad_norm=0.9,
            n_epochs=20,
            vf_coef=0.20315938606555833,
        ),
    )


@train_adversarial_ex.named_config
def seals_swimmer():
    # locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Swimmer-v0")
    total_timesteps = int(2e6)
    demonstrations = dict(
        rollout_path="/home/taufeeque/imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_swimmer_4/rollouts/final.pkl",
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=8,
            clip_range=0.1,
            ent_coef=5.167107294612664e-08,
            gae_lambda=0.95,
            gamma=0.999,
            learning_rate=0.0001214437022727675,
            max_grad_norm=2,
            n_epochs=20,
            # policy_kwargs are same as the defaults
            vf_coef=0.6162112311062333,
        ),
    )


@train_adversarial_ex.named_config
def seals_walker():
    # locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Walker2d-v0")
    demonstrations = dict(
        rollout_path="/home/taufeeque/imitation/output/train_experts/"
        "2022-10-11T06:27:42-07:00/seals_walker_8/rollouts/final.pkl",
    )
    train = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        ),
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=8,
            clip_range=0.4,
            ent_coef=0.00013057334805552262,
            gae_lambda=0.92,
            gamma=0.98,
            learning_rate=3.791707778339674e-05,
            max_grad_norm=0.6,
            n_epochs=5,
            vf_coef=0.6167177795726859,
        ),
    )


@train_adversarial_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_adversarial_ex.named_config
def reacher():
    common = dict(env_name="Reacher-v2")
    algorithm_kwargs = {"allow_variable_horizon": True}


# Debug configs


@train_adversarial_ex.named_config
def fast():
    # Minimize the amount of computation. Useful for test cases.

    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    algorithm_kwargs = dict(
        demo_batch_size=1,
        n_disc_updates_per_round=4,
    )


@train_adversarial_ex.named_config
def debug_nans():
    common = {"wandb": {"wandb_kwargs": {"project": "algorithm-benchmark"}}}
    total_timesteps = 1e7
    algorithm_kwargs = dict(
        demo_batch_size=128,
        n_disc_updates_per_round=8,
        # both are same as rl.batch_size
        # gen_replay_buffer_capacity=tune.choice([512, 1024]),
        # gen_train_timesteps=0,
    )
    rl = {
        "batch_size": 4096,
        "rl_kwargs": {"ent_coef": 0.1, "learning_rate": 7.316377404994506e-05},
    }
    seed = 0
    checkpoint_interval = 1
