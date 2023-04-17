import dataclasses
import logging
from typing import Optional

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from imitation.algorithms.adversarial.airl import AIRL
from imitation.data import rollout
from imitation_cli.utils import environment as gym_env
from imitation_cli.utils import (
    optimizer,
    policy,
    reward_network,
    rl_algorithm,
    trajectories,
)


@dataclasses.dataclass
class AIRLConfig:
    defaults: list = dataclasses.field(
        default_factory=lambda: [
            {"environment": "gym_env"},
            {"reward_net": "shaped"},
            # {"gen_algo": "ppo"},
            "_self_",
        ]
    )
    environment: gym_env.Config = MISSING
    expert_trajs: trajectories.Config = MISSING
    total_timesteps: int = int(1e6)
    checkpoint_interval: int = 0
    gen_algo: rl_algorithm.Config = rl_algorithm.PPO()
    reward_net: reward_network.Config = MISSING
    seed: int = 0
    demo_batch_size: int = 64
    n_disc_updates_per_round: int = 2
    disc_opt_cls: optimizer.Config = optimizer.Adam
    gen_train_timesteps: Optional[int] = None
    gen_replay_buffer_capacity: Optional[int] = None
    init_tensorboard: bool = False
    init_tensorboard_graph: bool = False
    debug_use_ground_truth: bool = False
    allow_variable_horizon: bool = True  # TODO: true just for debugging


cs = ConfigStore.instance()
cs.store(name="airl", node=AIRLConfig)
policy.register_configs("expert_trajs/expert_policy")
rl_algorithm.register_configs("gen_algo")
trajectories.register_configs("expert_trajs")
gym_env.register_configs("environment")
reward_network.register_configs("reward_net")


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="airl",
)
def run_airl(cfg: AIRLConfig) -> None:

    rng = np.random.default_rng(cfg.seed)
    expert_trajs = trajectories.get_trajectories(cfg.expert_trajs, rng)
    print(len(expert_trajs))

    venv = gym_env.make_venv(cfg.environment, rng)

    reward_net = reward_network.make_reward_net(cfg.reward_net)

    gen_algo = rl_algorithm.make_rl_algo(cfg.gen_algo, rng)

    trainer = AIRL(
        venv=venv,
        demonstrations=expert_trajs,
        gen_algo=gen_algo,
        reward_net=reward_net,
        demo_batch_size=cfg.demo_batch_size,
        n_disc_updates_per_round=cfg.n_disc_updates_per_round,
        disc_opt_cls=optimizer.make_optimizer(cfg.disc_opt_cls),
        gen_train_timesteps=cfg.gen_train_timesteps,
        gen_replay_buffer_capacity=cfg.gen_replay_buffer_capacity,
        init_tensorboard=cfg.init_tensorboard,
        init_tensorboard_graph=cfg.init_tensorboard_graph,
        debug_use_ground_truth=cfg.debug_use_ground_truth,
        allow_variable_horizon=cfg.allow_variable_horizon,
    )

    def callback(round_num: int, /) -> None:
        if cfg.checkpoint_interval > 0 and round_num % cfg.checkpoint_interval == 0:
            logging.log(
                logging.INFO,
                f"Saving checkpoint at round {round_num}. TODO implement this",
            )

    trainer.train(cfg.total_timesteps, callback)
    # TODO: implement evaluation
    # imit_stats = policy_evaluation.eval_policy(trainer.policy, trainer.venv_train)

    # Save final artifacts.
    if cfg.checkpoint_interval >= 0:
        logging.log(logging.INFO, f"Saving final checkpoint. TODO implement this")

    return {
        # "imit_stats": imit_stats,
        "expert_stats": rollout.rollout_stats(expert_trajs),
    }


if __name__ == "__main__":
    run_airl()
