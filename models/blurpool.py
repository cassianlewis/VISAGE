from typing import Union, Type

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReflectionPad2d, ReplicationPad2d, ZeroPad2d


def get_pad_layer(pad_type: str) -> Type[Union[ReflectionPad2d, ReplicationPad2d, ZeroPad2d]]:
    """
    Returns the padding layer class given a padding type.

    Parameters:
    - pad_type (str): The type of padding (e.g., 'reflect', 'replicate', 'zero').

    Returns:
    - Type[Union[ReflectionPad2d, ReplicationPad2d, ZeroPad2d]]: The padding layer class.
    """
    if pad_type in ['refl', 'reflect']:
        return ReflectionPad2d
    elif pad_type in ['repl', 'replicate']:
        return ReplicationPad2d
    elif pad_type == 'zero':
        return ZeroPad2d
    else:
        raise ValueError(f'Pad type [{pad_type}] not recognized')


def create_binomial_filter(filt_size: int) -> np.ndarray:
    """
    Creates a binomial filter for the given filter size (ie the nth row of Pascals triangle).

    Parameters:
    - filt_size (int): Size of the filter.

    Returns:
    - np.ndarray: A 1D numpy array representing the binomial filter.
    """
    if filt_size == 1:
        return np.array([1.])
    a = np.array([1.])
    for _ in range(1, filt_size):
        a = np.convolve(a, np.array([1, 1]))
    return a / a.sum()



class BlurPool(nn.Module):
    """
    A BlurPool layer module for down-sampling with antialiasing.

    This module applies a predefined blurring filter before down-sampling the input,
    reducing aliasing effects and preserving feature quality.

    Custom implementation from 'Making Convolutional Networks Shift-Invariant Again' -
    https://arxiv.org/abs/1904.11486 -

    Parameters:
    - channels (int): Number of input channels.
    - pad_type (str): Type of padding ('reflect', 'replicate', 'zero'). Default: 'reflect'.
    - filt_size (int): Size of the blur filter. Default: 4.
    - stride (int): Stride of the convolution. Default: 2.
    - pad_off (int): Additional padding offset. Default: 0.

    Attributes:
    - filt_size (int): Size of the blur filter.
    - pad_off (int): Additional padding offset.
    - pad_sizes (list): Calculated padding sizes.
    - stride (int): Stride of the convolution.
    - off (int): Offset for stride adjustment.
    - channels (int): Number of input channels.
    - filt (torch.Tensor): The blur filter.
    - pad (Callable): Padding layer.
    """

    def __init__(self, channels: int, pad_type: str = 'reflect', filt_size: int = 4, stride: int = 2, pad_off: int = 0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int((filt_size - 1) / 2), int(np.ceil((filt_size - 1) / 2))] * 2
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((stride - 1) / 2)
        self.channels = channels

        # Creating the binomial filter
        a = create_binomial_filter(filt_size)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))

        # Initializing the padding layer
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


# We also need one for Upsampling
class BlurTranspose(nn.Module):
    """
    A BlurTranspose layer module for up-sampling with antialiasing by applying a binomial filter.

    This module aims to reduce aliasing effects when up-sampling, by applying a predefined blurring
    filter prior to up-sampling the input, thereby preserving feature quality.

    Parameters:
    - channels (int): Number of input channels.
    - pad_type (str): Type of padding ('reflect', 'replicate', 'zero'). Default: 'reflect'.
    - filt_size (int): Size of the blur filter. Default: 4.
    - stride (int): Stride of the convolution. Default: 2.
    - pad_off (int): Additional padding offset. Default: 0.

    Attributes:
    - filt_size (int): Size of the blur filter.
    - pad_off (int): Additional padding offset.
    - pad_sizes (list): Calculated padding sizes.
    - stride (int): Stride of the convolution.
    - off (int): Offset for stride adjustment.
    - channels (int): Number of input channels.
    - filt (torch.Tensor): The blur filter.
    - pad (Callable): Padding layer.
    """

    def __init__(self, channels: int, pad_type: str = 'reflect', filt_size: int = 4, stride: int = 2, pad_off: int = 0):
        super(BlurTranspose, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int((filt_size - 1) / 2), int(np.ceil((filt_size - 1) / 2))] * 2
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((stride - 1) / 2.)
        self.channels = channels

        # Creating the binomial filter
        a = create_binomial_filter(filt_size)
        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((channels, 1, 1, 1)))

        # Initializing the padding layer
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)


    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # padded_input = inp if self.filt_size == 1 and self.pad_off == 0 else self.pad(inp)
        return F.conv_transpose2d(inp, self.filt, stride=self.stride, groups=inp.shape[1], padding=1)
