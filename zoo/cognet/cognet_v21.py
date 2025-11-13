import math
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import torch
import torchvision.transforms as transforms
from itertools import product as itp

"""
notes over previous version:
Layer cardinality now increases across layers, following RegNet paper.
Features increase by 2.5 with each layer, following RegNet paper.
GroupNorm 32 to either 16 or 25 for compatibility
Squeeze and excitation dynamics added to recurrent blocks.
"""

class Identity(nn.Module):
    def forward(self, x):
        return x


class LGN(nn.Module):

    def __init__(self, channels, kernel_size=11, stride=2):
        super().__init__()

        # parameters
        self.in_channels = ic = channels[0]
        self.out_channels = oc = channels[1]
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 2

        # layers
        self.conv = nn.Conv2d(ic, oc, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(oc)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.f = Identity()


    def forward(self, inputs):

        f = self.conv(inputs)
        f = self.norm(f)
        f = self.nonlin(f)
        f = self.pool(f)
        f = self.f(f)

        return f


class FLaBBlock(nn.Module):

    """
    Generic model block for FLaB
    :param channels: list of input channels for previous, current and next block
    :param kernel_size: int, size of kernel in middle bottleneck Conv2d
    :param stride: int, stride of convolutional in middle bottleneck Conv2d
    :param cardinality: int, number of groups in middle bottleneck Conv2d
    :param lat_pad: int, output_padding for lateral ConvTranspose2d
    :param back_pad: int, output_padding for backward ConvTranspose2d
    :param back: bool, whether to include backward connection
    """

    def __init__(self, channels, norm_groups, kernel_size=3, stride=2,
                 cardinality=32, lat_pad=1, back_pad=3, back=True):
        super().__init__()

        self.prev_channels = pc = channels[0]
        self.in_channels = ic = channels[1]
        self.out_channels = oc = channels[2]
        self.norm_groups = norm_groups
        self.stride = stride
        self.cardinality = cardinality
        self.back = back

        # skip connection
        self.conv_skip = nn.Conv2d(ic, oc, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.GroupNorm(norm_groups[2], oc)

        # resnext-style bottleneck
        self.conv1 = nn.Conv2d(ic, oc, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(norm_groups[2], oc)
        self.nonlin1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(oc, oc, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2,
                               groups=cardinality, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups[2], oc)
        self.nonlin2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(oc, oc, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(norm_groups[2], oc)
        self.nonlin3 = nn.LeakyReLU()

        # squeeze and excitation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_se1 = nn.Conv2d(oc, oc // 4, kernel_size=1, bias=False)
        self.nonlin_se = nn.LeakyReLU()
        self.conv_se2 = nn.Conv2d(oc // 4, oc, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

        # lateral connection
        self.conv_lat = nn.ConvTranspose2d(oc, ic, kernel_size=kernel_size,
            stride=stride, padding=1, output_padding=lat_pad, bias=False)
        self.norm_lat = nn.GroupNorm(norm_groups[1], ic)
        self.nonlin_lat = nn.LeakyReLU()

        # backward connection
        if back:
            self.conv_back = nn.ConvTranspose2d(oc, pc, kernel_size=5, stride=4,
                padding=2, output_padding=back_pad, bias=False)
            self.norm_back = nn.GroupNorm(norm_groups[0], pc)
            self.nonlin_back = nn.LeakyReLU()
        
        # allow for retrieval with forward hook
        self.f = Identity()
        self.l = Identity()
        self.b = Identity()


    def forward(self, inputs):

        # skip connection
        skip = self.conv_skip(inputs)
        skip = self.norm_skip(skip)

        # expansion convolution
        f = self.conv1(inputs)
        f = self.norm1(f)
        f = self.nonlin1(f)

        # middle convolution
        f = self.conv2(f)
        f = self.norm2(f)
        f = self.nonlin2(f)

        # contraction convolution
        f = self.conv3(f)
        f = self.norm3(f)

        # outgoing lateral connection
        l = self.conv_lat(f)
        l = self.norm_lat(l)
        l = self.nonlin_lat(l)
        
        # outgoing backward connection
        if self.back:
            b = self.conv_back(f)
            b = self.norm_back(b)
            b = self.nonlin_back(b)
        else:
            b = 0

        # squeeze and excitation
        se = self.global_pool(f)
        se = self.conv_se1(se)
        se = self.nonlin_se(se)
        se = self.conv_se2(se)
        se = self.sig(se)

        # combine with residual, final nonlin
        f = f * se + skip
        f = self.nonlin3(f)
        
        # allow for retrieval with forward hook
        f = self.f(f)
        l = self.l(l)
        b = self.b(b)

        return f, l, b


class FLaBDecoder(nn.Module):

    """
    Decoder for FLaB, containing adapative avgpool followed by a series of
    fully-connected layers interspersed with ReLUs. The number of units in each
    fc layer can be specified or defaults to a linspace between the number of
    input and output units.
    :param in_channels: int, number of feature maps in input
    :param out_features: int, number of output features
    :param head_depth: int, number of linear layers
    :param head_features: list, number of features in each linear layer
    :param head_width: int, height/width of feature maps after avgpool
    """

    def __init__(self, in_channels, out_features=1000, head_depth=1,
                 head_features=None, head_width=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(head_width)
        self.f = Identity()
        if head_features is None:
            self.head_features = torch.linspace(
                start=int(in_channels * (head_width ** 2)),
                end=out_features,
                steps=head_depth + 1,
                dtype=int)
        else:
            self.head_features = head_features

        # flexibly generate decoder based on head_depth
        for layer in range(head_depth):
            setattr(self, f'linear_{layer + 1}',
                    nn.Linear(self.head_features[layer],
                              self.head_features[layer + 1]))
            if layer < head_depth - 1:
                setattr(self, f'nonlin_{layer + 1}', nn.ReLU())

    def forward(self, inputs):

        f = self.avgpool(inputs)
        f = f.view(f.size(0), -1)
        for layer in range(self.head_depth):
            f = getattr(self, f'linear_{layer + 1}')(f)
            if layer < self.head_depth - 1:
                f = getattr(self, f'nonlin_{layer + 1}')(f)
        f = self.f(f)

        return f


class FLaB(nn.Module):

    def __init__(self, cycles=8):
        super().__init__()

        self.cycles = cycles
        chn = [3, 64, 80, 200, 500, 1250]
        crd = [4, 8, 20, 50]
        gnm = [1, 32, 40, 40, 50, 50]

        self.LGN = LGN(chn[0:2])
        self.V1 = FLaBBlock(channels=chn[0:3], cardinality=crd[0],
                            norm_groups=gnm[0:3], back=False)
        self.V2 = FLaBBlock(channels=chn[1:4], cardinality=crd[1],
                            norm_groups=gnm[1:4])
        self.V4 = FLaBBlock(channels=chn[2:5], cardinality=crd[2],
                            norm_groups=gnm[2:5])
        self.IT = FLaBBlock(channels=chn[3:6], cardinality=crd[3],
                            norm_groups=gnm[3:6], lat_pad=0, back_pad=1)
        self.decoder = FLaBDecoder(in_channels=chn[-1], out_features=1000, 
                                  head_depth=1, head_width=1)
        self.blocks = ['LGN', 'V1', 'V2', 'V4', 'IT', 'decoder']
        self.recurrent_blocks = self.blocks[1:-1]

    def forward(self, inputs):

        # initialize block states
        states = {block: {'f': 0, 'l': 0, 'b': 0} for block in self.blocks}

        # adjust cycles if video input is submitted
        video = len(inputs.shape) == 5
        if video:
            print(f'Video input detected, changing num cycles to num frames')
            cycles = inputs.shape[0]
        else:
            cycles = self.cycles
            states['LGN']['f'] = self.LGN(inputs)  # pass image once only

        # start processing
        for c in range(cycles):

            # if video, input next frame to LGN with each cycle
            if video:
                states['LGN']['f'] = self.LGN(inputs[c])

            for block in self.recurrent_blocks:

                blk = self.blocks.index(block)  # index of current block
                f = states[self.blocks[blk - 1]]['f']  # feedforward
                l = states[block]['l']   # lateral
                b = states[self.blocks[blk + 1]]['b']  # feedback
                inputs = f + l + b  # sum inputs
                outputs = getattr(self, block)(inputs)  # forward pass
                states[block] = {k: v for k, v in zip('flb', outputs)}  # store

            states['decoder']['f'] = self.decoder(states['IT']['f'])

        return states['decoder']['f']

