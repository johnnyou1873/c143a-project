import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise


class MeanDriftNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        _, C = x.shape
        noise = torch.randn(1, C) * self.std
        return x + noise


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


class TimeMasking(nn.Module):
    """
    Randomly mask contiguous time steps (SpecAugment-style).

    Supports:
        - (T, C) tensors: time x features
        - (B, T, C) tensors: batch x time x features

    Args:
        max_mask_frac: Max fraction of time steps to mask in one mask (0–1).
        num_masks: Number of independent time masks to apply.
        replace_with_zero: If True, masked frames are set to 0,
                           else they are set to the mean over time.
    """
    def __init__(self, max_mask_frac=0.1, num_masks=1, replace_with_zero=True):
        super().__init__()
        self.max_mask_frac = max_mask_frac
        self.num_masks = num_masks
        self.replace_with_zero = replace_with_zero

    def forward(self, x):
        orig_dim = x.dim()

        # Normalize shape to (B, T, C)
        if orig_dim == 2:  # (T, C)
            x = x.unsqueeze(0)
        elif orig_dim == 3:  # (B, T, C)
            pass
        else:
            raise ValueError(f"TimeMasking expects 2D or 3D tensor, got shape {x.shape}")

        B, T, C = x.shape
        x = x.clone()  # avoid in-place on input

        if self.max_mask_frac <= 0 or self.num_masks <= 0 or T == 0:
            return x if orig_dim == 3 else x.squeeze(0)

        max_mask_len = max(1, int(self.max_mask_frac * T))
        max_mask_len = min(max_mask_len, T)

        for b in range(B):
            for _ in range(self.num_masks):
                mask_len = torch.randint(1, max_mask_len + 1, (1,), device=x.device).item()
                if mask_len >= T:
                    start = 0
                else:
                    start = torch.randint(0, T - mask_len + 1, (1,), device=x.device).item()

                if self.replace_with_zero:
                    x[b, start:start + mask_len, :] = 0.0
                else:
                    mean_vec = x[b].mean(dim=0, keepdim=True)  # (1, C)
                    x[b, start:start + mask_len, :] = mean_vec

        # Restore original shape
        return x if orig_dim == 3 else x.squeeze(0)
