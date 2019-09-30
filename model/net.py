import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims
        self.label_dim = params.label_dims

        self.gkernel = gkern(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            nn.Linear((self.noise_dim + self.label_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 4*16*64, bias=False),
            nn.BatchNorm1d(4*16*64),
            nn.ReLU()
        )

        self.CONV = nn.Sequential(
            ConvTranspose2d_meta(64, 64, 5, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvTranspose2d_meta(64, 32, 5, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ConvTranspose2d_meta(32, 16, 5, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ConvTranspose2d_meta(16, 1, 5),
            )


    def forward(self, labels, noise):
        net = torch.cat([labels, noise], -1)
        net = self.FC(net)
        net = net.view(-1, 64, 4, 16)
        net = self.CONV(net)
        net = conv2d_meta(net, self.gkernel)
        return torch.tanh(net)




class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, params):
        super().__init__()

        self.CONV = nn.Sequential(
            Conv2d_meta(1, 64, 5, stride=2),
            nn.LeakyReLU(0.2)
            )

        self.FC = nn.Sequential(
            nn.Linear(64*16*64 + params.label_dims, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
            )


    def forward(self, img, labels):
        net = img + torch.randn(img.size()).type(Tensor)*0.1
        net = self.CONV(net)
        net = net.view(net.size(0), -1)
        net = torch.cat([net, labels], -1)
        net = self.FC(net)
        return net



