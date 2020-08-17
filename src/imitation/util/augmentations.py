"""GPU-accelerated image augmentations. When working with image-based
environments, these can be helpful for BC, and essential for the GAIL
discriminator."""

import collections
import enum
import inspect
import math
from typing import Optional, Sequence, Tuple

import kornia.augmentation as aug
import torch as th
from kornia.color.gray import rgb_to_grayscale
from torch import nn


class ColorSpace(str, enum.Enum):
    """Color space specification for use with image augmentations."""

    RGB = "RGB"
    GRAY = "GRAY"


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
        elif self.stack_color_space == ColorSpace.GRAY:
            self.stack_n_channels = 1
        else:
            raise ValueError(f"Unrecognised colour space '{stack_color_space}'")
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
                stacked_frames = images.size(0) // self.stack_n_channels
                assert (
                    stacked_frames * self.stack_n_channels == images.size(0)
                    and stacked_frames >= 1
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


class GaussianNoise(nn.Module):
    """Apply zero-mean Gaussian noise with a given standard deviation to input
    tensor."""

    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = x + self.std * th.randn_like(x)
        out.clamp_(0.0, 1.0)
        return out


class CIELabJitter(nn.Module):
    """Apply 'jitter' in CIELab colour space."""

    def __init__(self, max_lum_scale, max_uv_rads):
        super().__init__()
        self.max_lum_scale = max_lum_scale
        self.max_uv_rads = max_uv_rads

    def forward(self, x):
        # we take in stacked [N,C,H,W] images, where C=3*T. We then reshape
        # into [N,T,C,H,W] like apply_lab_jitter expects.
        stack_depth = x.size(1) // 3
        assert x.size(1) == 3 * stack_depth, x.shape
        x_reshape = x.view(x.shape[:1] + (stack_depth, 3) + x.shape[2:])
        jittered_reshape = apply_lab_jitter(
            x_reshape, self.max_lum_scale, self.max_uv_rads
        )
        jittered = jittered_reshape.view(x.shape)
        return jittered


class Rot90(nn.Module):
    """Apply a 0/90/180/270 degree rotation to the image."""

    def forward(self, images):
        batch_size = images.size(0)
        generated_rots = th.randint(low=0, high=4, size=(batch_size,))
        rot_elems = []
        for batch_idx in range(batch_size):
            rot_opt = generated_rots[batch_idx]
            rotated_image = th.rot90(images[batch_idx], k=rot_opt, dims=(1, 2))
            rot_elems.append(rotated_image)
        rot_images = th.stack(rot_elems)
        return rot_images


class Greyscale(nn.Module):
    __constants__ = ["p"]

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x):
        # separate out channels, just like in CIELabJitter
        batch_size = x.size(0)
        stack_depth = x.size(1) // 3
        assert x.size(1) == 3 * stack_depth, x.shape
        x_reshape = x.view(x.shape[:1] + (stack_depth, 3) + x.shape[2:])

        recol_elems = []
        rand_nums = th.rand(batch_size)
        for batch_idx in range(batch_size):
            if rand_nums[batch_idx] < self.p:
                recol_reduced = rgb_to_grayscale(x_reshape[batch_idx])
                rep_spec = (1,) * (recol_reduced.ndim - 3) + (3, 1, 1)
                recol = recol_reduced.repeat(rep_spec)
                recol_elems.append(recol)
            else:
                recol_elems.append(x_reshape[batch_idx])

        unshaped_result = th.stack(recol_elems, dim=0)
        result = unshaped_result.view(x.shape)

        return result


class MILBenchAugmentations(KorniaAugmentations):
    """Convenience class for data augmentation. Has a standard set of possible
    augmentations with sensible pre-set values."""

    def __init__(
        self,
        translate: bool = False,
        rotate: bool = False,
        noise: bool = False,
        flip_ud: bool = False,
        flip_lr: bool = False,
        rand_grey: bool = False,
        colour_jitter: bool = False,
        colour_jitter_mid: bool = False,
        colour_jitter_ex: bool = False,
        translate_ex: bool = False,
        rotate_mid: bool = False,
        rotate_ex: bool = False,
        rot90: bool = False,
        crop: bool = False,
        crop_ex: bool = False,
        erase: bool = False,
        grey: bool = False,
    ) -> None:
        transforms = []
        if colour_jitter or colour_jitter_ex:
            assert sum([colour_jitter, colour_jitter_mid, colour_jitter_ex]) <= 1
            if colour_jitter_ex:
                transforms.append(CIELabJitter(max_lum_scale=1.05, max_uv_rads=math.pi))
            elif colour_jitter_mid:
                transforms.append(CIELabJitter(max_lum_scale=1.01, max_uv_rads=0.6))
            else:
                transforms.append(CIELabJitter(max_lum_scale=1.01, max_uv_rads=0.15))

        if translate or rotate or translate_ex or rotate_ex:
            assert sum([rotate, rotate_ex, rotate_mid]) <= 1
            if rotate:
                rot_bounds = (-5, 5)
            elif rotate_mid:
                rot_bounds = (-20, 20)
            elif rotate_ex:
                rot_bounds = (-35, 35)
            else:
                rot_bounds = (0, 0)

            assert not (translate and translate_ex)
            if translate:
                trans_bounds = (0.05, 0.05)
            elif translate_ex:
                trans_bounds = (0.3, 0.3)
            else:
                trans_bounds = None

            transforms.append(
                aug.RandomAffine(
                    degrees=rot_bounds, translate=trans_bounds, padding_mode="border"
                )
            )

        if flip_lr:
            transforms.append(aug.RandomHorizontalFlip())

        if flip_ud:
            transforms.append(aug.RandomVerticalFlip())

        if crop or crop_ex:
            assert sum([crop, crop_ex]) <= 1
            if crop_ex:
                transforms.append(
                    aug.RandomResizedCrop(
                        size=(96, 96), scale=(0.5, 1.0), ratio=(0.75, 1.333)
                    )
                )
            else:
                transforms.append(
                    aug.RandomResizedCrop(
                        size=(96, 96), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                    )
                )

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

        if grey:
            transforms.append(Greyscale(p=0.5))

        super().__init__(*transforms)

    @classmethod
    def known_options(cls) -> collections.abc.Set:
        sig = inspect.signature(cls)
        return sig.parameters().keys()

    @classmethod
    def from_string_spec(cls, spec: str) -> "MILBenchAugmentations":
        known_options = cls.known_options()
        kwargs = {}
        for item in spec.split(","):
            item = item.strip()
            if item not in known_options:
                raise ValueError(f"Unknown augmentation option '{item}'")
            kwargs[item] = True
        return MILBenchAugmentations(**kwargs)


@th.jit.script
def _lab_f(t: th.Tensor) -> th.Tensor:
    """Intermediate function used to convert from RGB to Lab."""
    delta = 6 / 29
    return th.where(t > delta ** 3, t ** (1 / 3), t / (3 * delta ** 2) + 4 / 29)


@th.jit.script
def _lab_f_inv(t: th.Tensor) -> th.Tensor:
    """Inverse of _lab_f."""
    delta = 6 / 29
    return th.where(t > delta, t ** 3, 3 * delta ** 2 * (t - 4 / 29))


@th.jit.script
def rgb_to_lab(
    r: th.Tensor, g: th.Tensor, b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    """JITted conversion from RGB to Lab.

    Conversion formulas have been adapted from kornia.color.luv.rgb_to_luv so that:

    1. The function can be easily JITted.
    2. It converts to Lab instead of LUV."""

    # # Convert from Linear RGB to sRGB
    rs = th.where(r > 0.04045, th.pow(((r + 0.055) / 1.055), 2.4), r / 12.92)
    gs = th.where(g > 0.04045, th.pow(((g + 0.055) / 1.055), 2.4), g / 12.92)
    bs = th.where(b > 0.04045, th.pow(((b + 0.055) / 1.055), 2.4), b / 12.92)

    # sRGB to XYZ
    x = 0.412453 * rs + 0.357580 * gs + 0.180423 * bs
    y = 0.212671 * rs + 0.715160 * gs + 0.072169 * bs
    z = 0.019334 * rs + 0.119193 * gs + 0.950227 * bs

    # XYZ to Lab
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    x_frac = _lab_f(x / X_n)
    y_frac = _lab_f(y / Y_n)
    z_frac = _lab_f(z / Z_n)
    L = 116 * y_frac - 16
    a = 500 * (x_frac - y_frac)
    b = 200 * (y_frac - z_frac)

    return L, a, b


@th.jit.script
def lab_to_rgb(
    L: th.Tensor, a: th.Tensor, b: th.Tensor,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
    # convert from Lab to XYZ
    X_n = 0.950489
    Y_n = 1.000
    Z_n = 1.088840
    L_16_frac = (L + 16) / 116.0
    x = X_n * _lab_f_inv(L_16_frac + a / 500.0)
    y = Y_n * _lab_f_inv(L_16_frac)
    z = Z_n * _lab_f_inv(L_16_frac - b / 200)

    # inlined call to rgb_to_xyz(th.stack((x, y, z)))
    # (note that these coefficients are from the matrix inverse of the
    # coefficients in rgb_to_lab; that is why they appear to have higher
    # precision)
    rs = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    gs = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    bs = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z

    # Convert from sRGB to RGB Linear
    r = th.where(rs > 0.0031308, 1.055 * th.pow(rs, 1 / 2.4) - 0.055, 12.92 * rs)
    g = th.where(gs > 0.0031308, 1.055 * th.pow(gs, 1 / 2.4) - 0.055, 12.92 * gs)
    b = th.where(bs > 0.0031308, 1.055 * th.pow(bs, 1 / 2.4) - 0.055, 12.92 * bs)

    # return th.stack((r, g, b), dim=-3)

    return r, g, b


@th.jit.script
def _unif_rand_range(lo: float, hi: float, size: int, device: th.device) -> th.Tensor:
    return (hi - lo) * th.rand((size,), device=device) + lo


@th.jit.script
def apply_lab_jitter(
    images: th.Tensor, max_lum_scale: float, max_uv_rads: float
) -> th.Tensor:
    """Apply random L*a*b* jitter to each element of a batch of images. The
    `images` tensor should be of shape `[B,...,C,H,W]`, where the ellipsis
    denotes extraneous dimensions which will not be transformed separately
    (e.g. there might be a time axis present after the batch axis)."""

    assert len(images.shape) >= 4 and images.shape[-3] == 3
    assert 2.0 >= max_lum_scale >= 1.0
    assert max_uv_rads >= 0.0

    L, a, b = rgb_to_lab(
        images[..., 0, :, :], images[..., 1, :, :], images[..., 2, :, :]
    )

    # random transforms
    batch_size = images.size(0)
    ab_angles = _unif_rand_range(-max_uv_rads, max_uv_rads, batch_size, images.device)
    lum_scale_factors = _unif_rand_range(
        1.0 / max_lum_scale, max_lum_scale, batch_size, images.device
    )
    sines = th.sin(ab_angles)
    cosines = th.cos(ab_angles)

    # resize transformations to take advantage of broadcasting
    new_shape = (batch_size,) + (1,) * (a.ndim - 1)
    sines = sines.view(new_shape)
    cosines = cosines.view(new_shape)
    lum_scale_factors = lum_scale_factors.view(new_shape)

    # now apply the transformations
    # (this is way faster than stacking and using th.matmul())
    trans_L = th.clamp(L * lum_scale_factors, 0.0, 100.0)
    trans_a = cosines * a - sines * b
    trans_b = sines * a + cosines * b

    trans_r, trans_g, trans_b = lab_to_rgb(trans_L, trans_a, trans_b)

    rgb_trans = th.stack((trans_r, trans_g, trans_b), dim=-3)

    # throw out colours that can't be expressed as RGB (this is probably not a
    # good way of doing it, but whatever)
    rgb_trans.clamp_(0.0, 1.0)

    return rgb_trans
