import dataclasses

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclasses.dataclass
class EnvironmentConfig:
    gym_id: str = MISSING  # The environment to train on
    n_envs: int = 8  # number of environments in VecEnv
    parallel: bool = True  # Use SubprocVecEnv rather than DummyVecEnv
    max_episode_steps: int = MISSING  # Set to positive int to limit episode horizons
    env_make_kwargs: dict = dataclasses.field(
        default_factory=dict
    )  # The kwargs passed to `spec.make`.


@dataclasses.dataclass
class RetroEnvironmentConfig(EnvironmentConfig):
    gym_id: str = "Retro-v0"
    max_episode_steps: int = 4500


@dataclasses.dataclass
class SealEnvironmentConfig(EnvironmentConfig):
    gym_id: str = "Seal-v0"
    max_episode_steps: int = 1000
    aaa: int = 555


@dataclasses.dataclass
class PolicyConfig:
    env: EnvironmentConfig
    type: str = MISSING


@dataclasses.dataclass
class PPOPolicyConfig(PolicyConfig):
    type: str = "ppo"


@dataclasses.dataclass
class RandomPolicyConfig(PolicyConfig):
    type: str = "random"


@dataclasses.dataclass
class Config:
    env: EnvironmentConfig
    policy: PolicyConfig


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

cs.store(group="env", name="retro", node=RetroEnvironmentConfig)
cs.store(group="env", name="seal", node=SealEnvironmentConfig)

cs.store(group="policy", name="ppo", node=PPOPolicyConfig(env="${env}"))
cs.store(group="policy", name="random", node=RandomPolicyConfig(env="${env}"))

# cs.store(group="policy/env", name="retro", node=RetroEnvironmentConfig)
# cs.store(group="policy/env", name="seal", node=SealEnvironmentConfig)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config):
    print(cfg)


if __name__ == "__main__":
    main()
