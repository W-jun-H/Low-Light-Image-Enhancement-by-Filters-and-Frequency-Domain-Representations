"""Differentiable image and frequency-domain filters used by LLIE_F2.

All modules accept a four-dimensional ``NCHW`` tensor and preserve its shape.
The frequency-domain modules construct their masks from the current input
size, so they can be used with both cropped training images and full-resolution
test images.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_image_tensor(x):
    if x.ndim != 4:
        raise ValueError("Filter inputs must have shape [batch, channels, height, width].")


def _frequency_radius(x):
    """Return a normalized radius grid for the one-sided rFFT of ``x``."""
    height, width = x.shape[-2:]
    fy = torch.fft.fftfreq(height, device=x.device, dtype=x.dtype)
    fx = torch.fft.rfftfreq(width, device=x.device, dtype=x.dtype)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    return radius / math.sqrt(0.5)


def _apply_frequency_mask(x, mask):
    _check_image_tensor(x)
    spectrum = torch.fft.rfft2(x, norm="ortho")
    filtered = spectrum * mask.to(dtype=spectrum.real.dtype)
    return torch.fft.irfft2(filtered, s=x.shape[-2:], norm="ortho")


def _butterworth_lowpass(radius, cutoff, order):
    cutoff = max(float(cutoff), 1e-6)
    order = max(float(order), 1e-6)
    return 1.0 / (1.0 + (radius / cutoff).pow(2.0 * order))


def _depthwise_kernel_conv(x, kernel):
    channels = x.shape[1]
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    kernel = kernel.expand(channels, 1, *kernel.shape[-2:])
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=channels)


class ButterworthLowPassFilter(nn.Module):
    """Fixed Butterworth low-pass filter applied in the Fourier domain."""

    def __init__(self, cutoff=0.25, order=2, strength=1.0):
        super().__init__()
        self.cutoff = float(cutoff)
        self.order = float(order)
        self.strength = float(strength)

    def forward(self, x):
        radius = _frequency_radius(x)
        lowpass = _butterworth_lowpass(radius, self.cutoff, self.order)
        mask = 1.0 + self.strength * (lowpass - 1.0)
        return _apply_frequency_mask(x, mask[None, None])


class TBSIF(nn.Module):
    """Two-band spatial illumination filter with low- and high-pass branches.

    The low-pass branch carries illumination structure and the high-pass branch
    carries detail.  The two branches are combined into a tensor with the same
    shape as the input for direct use in the reconstruction network.
    """

    def __init__(
        self,
        low_cutoff=0.20,
        high_cutoff=0.20,
        order=2,
        lowpass_weight=1.0,
        highpass_weight=1.0,
    ):
        super().__init__()
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.order = float(order)
        self.lowpass_weight = float(lowpass_weight)
        self.highpass_weight = float(highpass_weight)

    def forward(self, x):
        radius = _frequency_radius(x)
        lowpass = _butterworth_lowpass(radius, self.low_cutoff, self.order)
        highpass = 1.0 - _butterworth_lowpass(radius, self.high_cutoff, self.order)
        low_features = _apply_frequency_mask(x, lowpass[None, None])
        high_features = _apply_frequency_mask(x, highpass[None, None])
        return self.lowpass_weight * low_features + self.highpass_weight * high_features


class FirstOrderIIRFilter(nn.Module):
    """Differentiable first-order recursive low-pass filter."""

    def __init__(self, alpha=0.85, learnable=False):
        super().__init__()
        alpha = min(max(float(alpha), 1e-4), 1.0 - 1e-4)
        value = math.log(alpha / (1.0 - alpha))
        parameter = nn.Parameter(torch.tensor(value), requires_grad=learnable)
        self.alpha_logit = parameter

    def _scan(self, x, dim):
        alpha = torch.sigmoid(self.alpha_logit)
        x = x.transpose(dim, -1)
        output = [x[..., 0]]
        for index in range(1, x.shape[-1]):
            output.append((1.0 - alpha) * x[..., index] + alpha * output[-1])
        return torch.stack(output, dim=-1).transpose(dim, -1)

    def forward(self, x):
        _check_image_tensor(x)
        horizontal = self._scan(x, -1)
        vertical = self._scan(x, -2)
        return 0.5 * (horizontal + vertical)


class OptimumFIRFilter(nn.Module):
    """Trainable depthwise FIR filter initialized as an identity filter."""

    def __init__(self, kernel_size=3, residual=False):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.kernel = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        center = kernel_size // 2
        with torch.no_grad():
            self.kernel[0, 0, center, center] = 1.0
        self.residual = bool(residual)

    def forward(self, x):
        filtered = _depthwise_kernel_conv(x, self.kernel)
        return x + (filtered - x) if self.residual else filtered


class FixedHighOrderIIRFilter(nn.Module):
    """Fixed high-order IIR low-pass filter formed by cascaded first-order stages."""

    def __init__(self, order=4, alpha=0.85):
        super().__init__()
        if order < 1:
            raise ValueError("order must be at least 1.")
        self.stages = nn.ModuleList(
            [FirstOrderIIRFilter(alpha=alpha, learnable=False) for _ in range(order)]
        )

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class LMSFIRFilter(nn.Module):
    """Trainable FIR filter representing the LMS adaptive-filter stage."""

    def __init__(self, kernel_size=3, step_size=0.01):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        self.step_size = float(step_size)
        self.kernel = nn.Parameter(torch.zeros(1, 1, kernel_size, kernel_size))
        with torch.no_grad():
            self.kernel[0, 0, kernel_size // 2, kernel_size // 2] = 1.0

    def forward(self, x):
        filtered = _depthwise_kernel_conv(x, self.kernel)
        return x + self.step_size * (filtered - x)


class BandPassFilter(nn.Module):
    """Fixed Butterworth band-pass filter applied in the Fourier domain."""

    def __init__(self, low_cutoff=0.10, high_cutoff=0.50, order=2, strength=1.0):
        super().__init__()
        if low_cutoff < 0 or high_cutoff <= low_cutoff:
            raise ValueError("Require 0 <= low_cutoff < high_cutoff.")
        self.low_cutoff = float(low_cutoff)
        self.high_cutoff = float(high_cutoff)
        self.order = float(order)
        self.strength = float(strength)

    def forward(self, x):
        radius = _frequency_radius(x)
        highpass = 1.0 - _butterworth_lowpass(radius, self.low_cutoff, self.order)
        lowpass = _butterworth_lowpass(radius, self.high_cutoff, self.order)
        bandpass = highpass * lowpass
        mask = 1.0 + self.strength * (bandpass - 1.0)
        return _apply_frequency_mask(x, mask[None, None])


class LearnableAdaptiveFilter(nn.Module):
    """Learn a smooth, resolution-independent radial frequency response."""

    def __init__(self, hidden_channels=16, strength=1.0, residual=True):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive.")
        self.strength = float(strength)
        self.residual = bool(residual)
        self.response = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.response[2].weight)
        nn.init.zeros_(self.response[2].bias)

    def forward(self, x):
        _check_image_tensor(x)
        radius = _frequency_radius(x)
        response = self.response(radius.reshape(-1, 1)).reshape_as(radius)
        mask = 1.0 + self.strength * response
        filtered = _apply_frequency_mask(x, mask[None, None])
        return x + (filtered - x) if self.residual else filtered


__all__ = [
    "ButterworthLowPassFilter",
    "TBSIF",
    "FirstOrderIIRFilter",
    "OptimumFIRFilter",
    "FixedHighOrderIIRFilter",
    "LMSFIRFilter",
    "BandPassFilter",
    "LearnableAdaptiveFilter",
]
