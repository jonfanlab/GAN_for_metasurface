import torch
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import scipy.stats as st
from typing import Tuple
import math
import logging

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    

def index_along(tensor, key, axis):
    indexer = [slice(None)] * len(tensor.shape)
    indexer[axis] = key
    return tensor[tuple(indexer)]


def pad_reflect(inputs, padding: int, axis: int):
    if padding % 2 != 0:
        raise ValueError('cannot do centered padding if padding is not even')
    ndim = len(inputs.shape)
    
    if axis < 0:
        axis += ndim
    axis = ndim - axis - 1
    
    if axis == 0:
        paddings = (padding // 2, padding // 2, 0, 0)
    else:
        paddings = ((0, 0) * axis +
                  (padding // 2, padding // 2))

    return F.pad(inputs, paddings, mode='reflect')



def pad_periodic(inputs, padding: int, axis: int, center: bool = True):
    if center:
        if padding % 2 != 0:
            raise ValueError('cannot do centered padding if padding is not even')
        inputs_list = [index_along(inputs, slice(-padding//2, None), axis),
                       inputs,
                       index_along(inputs, slice(None, padding//2), axis)]
    else:
        inputs_list = [inputs, index_along(inputs, slice(None, padding), axis)]
    return torch.cat(inputs_list, dim=axis)


def pad2d_meta(inputs, padding: Tuple[int, int]):
    padding_y, padding_x = padding
    return pad_periodic(pad_reflect(inputs, padding_y, axis=-2),
                                            padding_x, axis=-1, center=True)



def gkern(kernlen=7, nsig=4):
    """Returns a 2D Gaussian kernel array."""

    x_cord = torch.arange(kernlen)
    x_grid = x_cord.repeat(kernlen).view(kernlen, kernlen)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).type(Tensor)

    mean = (kernlen - 1)/2.
    variance = nsig**2.

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                  torch.exp(
                      -torch.sum((xy_grid - mean)**2., dim=-1) /\
                      (2*variance)
                  )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    return gaussian_kernel.type(Tensor).requires_grad_(False)

def conv2d(inputs, kernel, padding='same'):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 2d kernel
    """
    B, C, _, _ = inputs.size()
    kH, kW = kernel.size()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, C, 1, 1)

    if padding == 'valid':
        return F.conv2d(inputs, kernel)
    elif padding == 'same':
        pad = ((kH-1)//2, (kW-1)//2)
        return F.conv2d(inputs, kernel, padding = pad)


def conv2d_meta(inputs, kernel):
    """
    Args:
        inputs: B x C x H x W
        gkernel: 2d kernel
    """
    kH, kW = kernel.size()
    padded_inputs = pad2d_meta(inputs,(kH-1, kW-1))
    
    return conv2d(padded_inputs, kernel, padding='valid')



class ConvTranspose2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 groups=1, bias=True, dilation=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.trim = self.padding * stride // 2
        pad = (kernel_size - stride) // 2 
        self.output_padding = (kernel_size - stride) % 2 
        self.conv2d_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=pad,
                                          output_padding=0, groups=groups, bias=bias, dilation=dilation)
    
    def forward(self, inputs):
        padded_inputs = pad2d_meta(inputs, (self.padding, self.padding))
        padded_outputs = self.conv2d_transpose(padded_inputs)
        
        if self.output_padding:
            padded_outputs = padded_outputs[:, :, 1:, 1:]
            
        if self.trim:
            return padded_outputs[:, :, self.trim:-self.trim, self.trim:-self.trim]
        else:
            return padded_outputs
 
    
class Conv2d_meta(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.padding = (kernel_size - 1)*dilation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, 
                                dilation = dilation, groups = groups, bias = bias)
    
    def forward(self, inputs):
        padded_inputs = pad2d_meta(inputs, (self.padding, self.padding))
        return self.conv2d(padded_inputs) 





        