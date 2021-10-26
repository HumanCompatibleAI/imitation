"""Convenience utilities for working with augmentations."""

import inspect
import math
from typing import Sequence, Set

import kornia.augmentation as kornia_aug
import torch as th
from torch import nn

import imitation.augment.augmentations as im_aug
from imitation.augment.color import ColorSpace, split_unsplit_channel_stack


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
        color_space: ColorSpace,
        # If temporally_consistent=True, then we apply the same sampled
        # augmentation to each frame in a frame stack, but different
        # augmentations to each frame in a batch. Otherwise, very frame gets a
        # different augmentation.
        temporally_consistent: bool = False,
    ) -> None:
        super().__init__()
        self.kornia_ops = nn.Sequential(*kornia_ops)
        self.temporally_consistent = temporally_consistent
        self.color_space = color_space

    def forward(self, images: th.Tensor) -> th.Tensor:
        """Apply augmentations to the given image batch."""
        # no_grad() ensures that we don't unnecessarily build up a backward graph
        with th.no_grad():
            # check type & number of dims
            if images.dim() != 4 or not th.is_floating_point(images):
                raise ValueError(
                    f"Images of shape '{images.shape}' and type '{images.dtype}' do "
                    "not have rank 4 and floating point type"
                )

            if not self.temporally_consistent:
                # split [B,…,S*C,H,W] tensor to [B,…,S,C,H,W]
                images, unsplit = split_unsplit_channel_stack(images, self.color_space)
                old_shape = images.shape
                # Now reshape to [B*…*S,C,H,W] so that all intervening axis are
                # absorbed into the batch axis. Each element in the batch will
                # be augmented separately, which breaks temporal consistency
                # within a frame stack.
                images = images.reshape((-1, ) + images.shape[-3:])

            # apply augmentations
            images = self.kornia_ops(images)

            if not self.temporally_consistent:
                # undo reshaping of everything along the batch axis
                images = images.reshape(old_shape)
                # undo separation of channels
                images = unsplit(images)

        return images


class StandardAugmentations(KorniaAugmentations):
    """Convenience class for constructing data augmenters.

    Contains many common augmentations configured to several preset magnitudes.
    For instance, `color_jitter` enables slight colour jitter;
    `color_jitter_mid` enables more intense (mid-level) color jitter, and
    `color_jitter_ex` enables extreme color jitter (in this case, completely
    randomising hue, and slightly randomising luminance). The
    `.from_string_spec()` method makes it easy to construct image augmenters
    from readable specifications, like `"translate,rotate_ex,gray"`. This is
    most useful when you want to quickly try many different kinds of
    augmentations, but don't want to bother figuring out reasonable parameter
    settings for them.
    """

    def __init__(
        self,
        *,
        translate: bool = False,
        translate_ex: bool = False,
        rotate: bool = False,
        rotate_mid: bool = False,
        rotate_ex: bool = False,
        color_jitter: bool = False,
        color_jitter_mid: bool = False,
        color_jitter_ex: bool = False,
        flip_ud: bool = False,
        flip_lr: bool = False,
        noise: bool = False,
        rot90: bool = False,
        erase: bool = False,
        gray: bool = False,
        gaussian_blur: bool = False,
        stack_color_space: ColorSpace,
        **kwargs,
    ) -> None:
        """Construct an augmenter that sequentially applies the given augmentations.

        Args:
            translate: random translation by up to 5% of image dimensions.
            translate_ex: translation by up to 30% of image dimensions.
            rotate: randomise image rotation by up to 5 degrees.
            rotate_mid: randomise image rotation by up to 20 degrees.
            rotate_ex: randomise image rotation by up to 35 degrees.
            color_jitter: convert to Lab and randomise the (a,b) channel
                direction (~hue) by up to 0.15 radians, and rescale luminance
                by up to 1%.
            color_jitter_mid: color jitter by up to 0.6 rad on (a,b) and up to
                1% on luminance.
            color_jitter_ex: color jitter that chooses random orientation for
                (a,b) and scales luminance up/down by up to 5%.
            flip_ud: flip up-down with 50% probability.
            flip_lr: flip left-right with 50% probability.
            noise: add iid, zero-mean Gaussian noise with stddev 0.01.o
            rot90: randomly rotate by 0, 90, 180, or 270 degrees (faster than
                unconstrained rotation).
            erase: with 50% probability, randomly erase a rectangle from the
                image and replace it with value 0.5 (see `kornia`'s
                `RandomErasing`).
            gray: with 50% probability, convert the image to grayscale.
            gaussian_blur: always apply Gaussian blur with sigma=1.0.
            stack_color_space: color space for images that will be augmented.
            kwargs: additional keyword arguments to be passed to
                `KorniaAugmentations`.
        """
        transforms = []

        # color jitter
        assert sum([color_jitter, color_jitter_mid, color_jitter_ex]) <= 1
        if color_jitter_ex:
            transforms.append(
                im_aug.CIELabJitter(
                    max_lum_scale=1.05,
                    max_uv_rads=math.pi,
                    color_space=stack_color_space,
                )
            )
        elif color_jitter_mid:
            transforms.append(
                im_aug.CIELabJitter(
                    max_lum_scale=1.01,
                    max_uv_rads=0.6,
                    color_space=stack_color_space,
                )
            )
        elif color_jitter:
            transforms.append(
                im_aug.CIELabJitter(
                    max_lum_scale=1.01,
                    max_uv_rads=0.15,
                    color_space=stack_color_space,
                )
            )

        # translation and rotation get combined into a single RandomAffine transform
        assert sum([rotate, rotate_ex, rotate_mid]) <= 1
        if rotate:
            rot_bounds = (-5, 5)
        elif rotate_mid:
            rot_bounds = (-20, 20)
        elif rotate_ex:
            rot_bounds = (-35, 35)
        else:
            # note that (0,0) doesn't work (it raises an error in Torch when
            # Torch tries to sample a uniform random from the nonexistent
            # interval [0,0).)
            rot_bounds = 0

        sum([translate, translate_ex]) <= 1
        if translate:
            trans_bounds = (0.05, 0.05)
        elif translate_ex:
            trans_bounds = (0.3, 0.3)
        else:
            trans_bounds = None

        if rot_bounds != 0 or trans_bounds:
            transforms.append(
                kornia_aug.RandomAffine(
                    degrees=rot_bounds, translate=trans_bounds, padding_mode="border"
                )
            )

        if flip_lr:
            transforms.append(kornia_aug.RandomHorizontalFlip())

        if flip_ud:
            transforms.append(kornia_aug.RandomVerticalFlip())

        if rot90:
            transforms.append(im_aug.Rot90())

        if erase:
            transforms.append(kornia_aug.RandomErasing(value=0.5))

        if gaussian_blur:
            transforms.append(im_aug.GaussianBlur(kernel_hw=5, sigma=1))

        if noise:
            # Remember that values lie in [0,1], so std=0.01 (for example)
            # means there's a >99% chance that any given noise value will lie
            # in [-0.03,0.03]. I think any value <=0.03 will probably be
            # reasonable.
            noise_mod = im_aug.GaussianNoise(std=0.01)
            # JIT doesn't make it any faster (unsurprisingly)
            # noise_mod = th.jit.script(noise_mod)
            transforms.append(noise_mod)

        if gray:
            transforms.append(im_aug.Grayscale(p=0.5, color_space=stack_color_space))

        super().__init__(transforms, color_space=stack_color_space, **kwargs)

    @classmethod
    def known_options(cls) -> Set[str]:
        """Collect all Boolean options of this class (i.e. augmentations).

        Returns: a set of augmentation names that the constructor takes as
            kwargs.
        """
        sig = inspect.signature(cls)
        known_options = set(
            key for key, param in sig.parameters.items() if param.annotation is bool
        )
        return known_options

    @classmethod
    def from_string_spec(
        cls,
        spec: str,
        stack_color_space: ColorSpace,
        temporally_consistent: bool = False,
    ) -> "StandardAugmentations":
        """Construct an augmenter from a string specification.

        Args:
            spec: a string with comma-separated names of keyword arguments that
                should be set to True. For example, "translate,rotate,gray"
                would turn on random translation, random rotation, and random
                desaturation, but no other augmentations. See
                `StandardAugmentations.known_options()` for all options.
            stack_color_space: color space for augmented images.
            temporally_consistent: should we augment the frames in each frame
                stack in the same way?

        Returns: the constructed `StandardAugmentations` object.
        """
        known_options = cls.known_options()
        kwargs = {}
        for item in spec.split(","):
            item = item.strip()
            if item not in known_options:
                raise ValueError(f"Unknown augmentation option '{item}'")
            kwargs[item] = True
        return StandardAugmentations(**kwargs, stack_color_space=stack_color_space,
                                     temporally_consistent=temporally_consistent)
