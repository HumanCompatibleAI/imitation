"""Additional GPU-accelerated image augmentations. When working with
image-based environments, these can be helpful for BC, and essential for the
GAIL discriminator."""

import torch as th
from kornia.color.gray import rgb_to_grayscale
from kornia.filters.filter import filter2d_separable
from kornia.filters.kernels import get_gaussian_kernel1d
from torch import nn

from imitation.augment.color import apply_lab_jitter, split_unsplit_channel_stack


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


@th.jit.script
def _blur(images: th.Tensor, kernel: th.Tensor, p: float):
    batch_size = images.size(0)
    blur_mask = th.rand(batch_size) < p
    out_images = images.clone()
    if blur_mask.sum() <= 0:
        # this function merely copies the input tensor if no images are chosen
        # for blurring
        return out_images
    extracted = images[blur_mask]
    blurred = filter2d_separable(extracted, kernel, kernel,
                                 "reflect")
    out_images[blur_mask] = blurred
    return out_images


class GaussianBlur(nn.Module):
    kernel: th.Tensor

    def __init__(self, kernel_hw: int = 5, sigma: float = 1.0, p: float = 0.5):
        super().__init__()
        assert isinstance(kernel_hw, int) and kernel_hw > 0, kernel_hw
        assert sigma > 0, sigma
        assert 0 <= p <= 1, p
        # kernel must be of size [1, kernel_hw] for filter2d_separable
        kernel: th.Tensor = get_gaussian_kernel1d(kernel_hw, sigma)[None]
        self.register_buffer('kernel', kernel)
        assert self.kernel.shape == (1, kernel_hw), self.kernel.shape
        self.p = p

    def forward(self, images):
        # Note that Kornia 0.6 has a built-in GaussianBlur2d class, but it's
        # slow because it recomputes the kernel on each forward pass.
        return _blur(images, self.kernel, self.p)


class CIELabJitter(nn.Module):
    """Apply 'jitter' in CIELab color space."""

    def __init__(self, max_lum_scale, max_uv_rads, color_space):
        super().__init__()
        self.max_lum_scale = max_lum_scale
        self.max_uv_rads = max_uv_rads
        self.color_space = color_space

    def forward(self, x):
        x_reshape, unsplit = split_unsplit_channel_stack(x, self.color_space)
        jittered_reshape = apply_lab_jitter(
            x_reshape, self.max_lum_scale, self.max_uv_rads
        )
        jittered = unsplit(jittered_reshape)
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


class Grayscale(nn.Module):
    __constants__ = ["p", "color_space"]

    def __init__(self, p: float = 0.5, *, color_space) -> None:
        super().__init__()
        self.p = p
        self.color_space = color_space

    def forward(self, x):
        # separate out channels, just like in CIELabJitter
        x_reshape, unsplit = split_unsplit_channel_stack(x, self.color_space)
        batch_size = x.size(0)

        recol_elems = []
        rand_nums = th.rand(batch_size)
        for batch_idx, sample in enumerate(rand_nums):
            if sample < self.p:
                recol_reduced = rgb_to_grayscale(x_reshape[batch_idx])
                rep_spec = (1,) * (recol_reduced.ndim - 3) + (3, 1, 1)
                recol = recol_reduced.repeat(rep_spec)
                recol_elems.append(recol)
            else:
                recol_elems.append(x_reshape[batch_idx])

        unshaped_result = th.stack(recol_elems, dim=0)
        result = unsplit(unshaped_result)

        return result
