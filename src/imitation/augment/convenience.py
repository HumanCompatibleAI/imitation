"""Convenience utilities for working with augmentations."""

import inspect
import math
from typing import Optional, Sequence, Set

import kornia.augmentation as aug
import torch as th
from torch import nn

from imitation.augment.augmentations import (
    CIELabJitter,
    GaussianNoise,
    Grayscale,
    Rot90,
)
from imitation.augment.color import ColorSpace


class KorniaAugmentations(nn.Module):
    """Container that applies a series of Kornia augmentations, one after the other.

    Underneath, it's essentially just an application of `nn.Sequential` to the
    given ops. It does a few extra interesting things, though:

    1. It does shape and type sanity checks.
    2. It supports stacked frames. For example, if your environment produces
       stacked RGB tensors of shape `[N,(3*F),H,W]`, then it can break them
       into `F` separate frames before passing them to the Kornia
       augmentations, then recombine them at the end.
    """

    def __init__(
        self,
        kornia_ops: Sequence[nn.Module],
        stack_color_space: Optional[ColorSpace] = None,
    ) -> None:
        super().__init__()
        self.stack_n_channels = None
        if stack_color_space == ColorSpace.RGB:
            self.stack_n_channels = 3
        elif stack_color_space == ColorSpace.GRAY:
            self.stack_n_channels = 1
        else:
            raise ValueError(f"Unrecognised color space '{stack_color_space}'")
        self.kornia_ops = nn.Sequential(*kornia_ops)

    def forward(self, images: th.Tensor) -> th.Tensor:
        """Apply augmentations to the given image batch."""
        # no_grad() ensures that we don't unnecessarily build up a backward graph
        with th.no_grad():
            # check type & number of dims
            assert th.is_floating_point(images)
            assert images.dim() == 4

            # push frames into the batch axis, if necessary
            orig_shape = images.shape
            if self.stack_n_channels is not None:
                # make sure this is a channels-last stack of images
                stacked_frames = images.size(1) // self.stack_n_channels
                if stacked_frames < 1:
                    raise ValueError(
                        f"No input frames in tensor of shape '{images.shape}' "
                        f"(channels per frame: {self.stack_n_channels})"
                    )
                if stacked_frames * self.stack_n_channels != images.size(1):
                    raise ValueError(
                        f"Image tensor shape '{images.shape}' not divisible by "
                        f"channels per frame ({self.stack_n_channels})"
                    )
                images = images.view(
                    (images.size(0) * stacked_frames, self.stack_n_channels)
                    + orig_shape[2:]
                )

            # apply augmentations
            images = self.kornia_ops(images)

            # restore shape
            images = images.reshape(orig_shape)

        return images


class StandardAugmentations(KorniaAugmentations):
    """Convenience class for data augmentation. Has a standard set of possible
    augmentations with sensible pre-set values."""

    def __init__(
        self,
        translate: bool = False,
        rotate: bool = False,
        noise: bool = False,
        flip_ud: bool = False,
        flip_lr: bool = False,
        color_jitter: bool = False,
        color_jitter_mid: bool = False,
        color_jitter_ex: bool = False,
        translate_ex: bool = False,
        rotate_mid: bool = False,
        rotate_ex: bool = False,
        rot90: bool = False,
        erase: bool = False,
        gray: bool = False,
        stack_color_space: Optional[ColorSpace] = None,
    ) -> None:
        transforms = []

        # color jitter
        assert sum([color_jitter, color_jitter_mid, color_jitter_ex]) <= 1
        if color_jitter_ex:
            transforms.append(CIELabJitter(max_lum_scale=1.05, max_uv_rads=math.pi))
        elif color_jitter_mid:
            transforms.append(CIELabJitter(max_lum_scale=1.01, max_uv_rads=0.6))
        elif color_jitter:
            transforms.append(CIELabJitter(max_lum_scale=1.01, max_uv_rads=0.15))

        # translation and rotation get combined into a single RandomAffine transform
        assert sum([rotate, rotate_ex, rotate_mid]) <= 1
        if rotate:
            rot_bounds = (-5, 5)
        elif rotate_mid:
            rot_bounds = (-20, 20)
        elif rotate_ex:
            rot_bounds = (-35, 35)
        else:
            rot_bounds = (0, 0)

        sum([translate, translate_ex]) <= 1
        if translate:
            trans_bounds = (0.05, 0.05)
        elif translate_ex:
            trans_bounds = (0.3, 0.3)
        else:
            trans_bounds = None

        if any(rot_bounds) or trans_bounds:
            transforms.append(
                aug.RandomAffine(
                    degrees=rot_bounds, translate=trans_bounds, padding_mode="border"
                )
            )

        if flip_lr:
            transforms.append(aug.RandomHorizontalFlip())

        if flip_ud:
            transforms.append(aug.RandomVerticalFlip())

        if rot90:
            transforms.append(Rot90())

        if erase:
            transforms.append(aug.RandomErasing(value=0.5))

        if noise:
            # Remember that values lie in [0,1], so std=0.01 (for example)
            # means there's a >99% chance that any given noise value will lie
            # in [-0.03,0.03]. I think any value <=0.03 will probably be
            # reasonable.
            noise_mod = GaussianNoise(std=0.01)
            # JIT doesn't make it any faster (unsurprisingly)
            # noise_mod = th.jit.script(noise_mod)
            transforms.append(noise_mod)

        if gray:
            transforms.append(Grayscale(p=0.5))

        super().__init__(transforms, stack_color_space=stack_color_space)

    @classmethod
    def known_options(cls) -> Set[str]:
        """Collect all Boolean options of this class (i.e. augmentations)."""
        sig = inspect.signature(cls)
        known_options = set(
            key for key, param in sig.parameters.items() if param.annotation is bool
        )
        return known_options

    @classmethod
    def from_string_spec(
        cls, spec: str, stack_color_space: Optional[ColorSpace] = None
    ) -> "StandardAugmentations":
        known_options = cls.known_options()
        kwargs = {}
        for item in spec.split(","):
            item = item.strip()
            if item not in known_options:
                raise ValueError(f"Unknown augmentation option '{item}'")
            kwargs[item] = True
        return StandardAugmentations(**kwargs, stack_color_space=stack_color_space)
