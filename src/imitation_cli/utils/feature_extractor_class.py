"""Register Hydra configs for stable_baselines3 feature extractors."""
import dataclasses
from enum import Enum

import stable_baselines3.common.torch_layers as torch_layers
from hydra.core.config_store import ConfigStore


class FeatureExtractorClass(Enum):
    """Enum of feature extractor classes."""

    FlattenExtractor = torch_layers.FlattenExtractor
    NatureCNN = torch_layers.NatureCNN


@dataclasses.dataclass
class Config:
    """Base config for stable_baselines3 feature extractors."""

    feature_extractor_class: FeatureExtractorClass
    _target_: str = "imitation_cli.utils.feature_extractor_class.Config.make"

    @staticmethod
    def make(feature_extractor_class: FeatureExtractorClass) -> type:
        return feature_extractor_class.value


FlattenExtractor = Config(FeatureExtractorClass.FlattenExtractor)
NatureCNN = Config(FeatureExtractorClass.NatureCNN)


def register_configs(group: str):
    cs = ConfigStore.instance()
    for cls in FeatureExtractorClass:
        cs.store(group=group, name=cls.name.lower(), node=Config(cls))
