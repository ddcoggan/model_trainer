import math
from collections import OrderedDict
from torch import nn, stack
from argparse import Namespace
import torch
import numpy as np


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)



class CORblock_FLaB(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, out_shape=None, V1=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.V1 = V1

        if V1:
            # special processing for V1
            self.conv_input = nn.Conv2d(3, out_channels, kernel_size=7, stride=2, padding=3, bias=False) # hard code 3 input channels
            self.norm_input = nn.BatchNorm2d(out_channels)
            self.nonlin_input = nn.ReLU(inplace=True)
            self.pool_input = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # standard processing
            self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            self.norm_input = nn.Identity()
            self.nonlin_input = nn.Identity()
            self.pool_input = nn.Identity()

        # forward
        self.conv_skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels * self.scale)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels * self.scale)
        self.nonlin2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.nonlin3 = nn.ReLU()

        # lateral
        self.conv_lateral = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm_lateral = nn.BatchNorm2d(out_channels)
        self.nonlin_lateral = nn.ReLU()

        # backward
        if V1:
            # no feedback from V1
            self.conv_back = nn.Identity()
            self.norm_back = nn.Identity()
            self.nonlin_back = nn.Identity()
        else:
            # standard processing
            self.conv_back = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=3, padding=1, bias=False)
            self.norm_back = nn.BatchNorm2d(in_channels)
            self.nonlin_back = nn.ReLU()


    def forward(self, f, la, b):

        f = self.conv_input(f)
        f = self.norm_input(f)
        f = self.nonlin_input(f)
        f = self.pool_input(f)

        # combine with lateral and feed back signals
        f = f + la + b

        # skip connection
        skip = self.conv_skip(f)
        skip = self.norm_skip(skip)

        # expansion convolution
        f = self.conv1(f)
        f = self.norm1(f)
        f = self.nonlin1(f)

        # middle convolution
        f = self.conv2(f)
        f = self.norm2(f)
        f = self.nonlin2(f)

        # contraction convolution
        f = self.conv3(f)
        f = self.norm3(f)

        # combine with skip
        f += skip

        # lateral connection
        la = self.conv_lateral(f)
        la = self.norm_lateral(la)
        la = self.nonlin_lateral(la)

        # backward connection
        b = self.conv_back(f)
        b = self.norm_back(b)
        b = self.nonlin_back(b)

        # final nonlin
        f = self.nonlin3(f)

        return f, la, b


class CORnet_decoder(nn.Module):

    def __init__(self, num_features, out_channels, head_depth):
        super().__init__()

        self.out_channels = out_channels
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        head_sizes = [int(x) for x in (np.linspace(num_features, 1000, head_depth + 1))]

        for out_channel in range(out_channels):
            for layer in range(head_depth):
                setattr(self, f'linear_{out_channel + 1}_{layer + 1}',
                        nn.Linear(head_sizes[layer], head_sizes[layer + 1]))
                if layer < head_depth:
                    setattr(self, f'nonlin_{out_channel + 1}_{layer + 1}', nn.ReLU())



    def forward(self, inp):

        x = self.avgpool(inp)
        x = self.flatten(x)

        out = []
        for out_channel in range(self.out_channels):

            x_out = x.clone()

            for layer in range(self.head_depth):

                x_out = getattr(self, f'linear_{out_channel+1}_{layer+1}')(x_out)

                if layer < self.head_depth:
                    x_out = getattr(self, f'nonlin_{out_channel+1}_{layer+1}')(x_out)

            out.append(x_out)

        # do not stack up on axis 0 as this is interferes with aggregation of responses across devices
        if self.out_channels > 1:
            return stack(out, axis=2)
        else:
            return out[0]


class CORnet_FLaB(nn.Module):

    def __init__(self, M):
        super().__init__()

        # fill in any missing parameters with defaults
        class defaults:
            kernel_size = (3,3,3,3)
            num_features = (64,128,256,512)
            out_channels = 1
            times = 4
            head_depth = 1
            batch_size = 32
        [setattr(M,param,value) for param, value in defaults.__dict__.items() if not param.startswith('_') and not hasattr(M, param)]

        self.times = M.times

        self.V1 = CORblock_FLaB(M.num_features[0], M.num_features[0], kernel_size=M.kernel_size[0], out_shape=56, V1=True,)
        self.V2 = CORblock_FLaB(M.num_features[0], M.num_features[1], kernel_size=M.kernel_size[1], out_shape=28)
        self.V4 = CORblock_FLaB(M.num_features[1], M.num_features[2], kernel_size=M.kernel_size[2], out_shape=14)
        self.IT = CORblock_FLaB(M.num_features[2], M.num_features[3], kernel_size=M.kernel_size[3], out_shape=7)
        self.decoder = CORnet_decoder(M.num_features[3], M.out_channels, M.head_depth)


    def forward(self, inp):

        # initialize block states
        blocks = ['V1', 'V2', 'V4', 'IT', 'decoder']
        states = {}
        for block in blocks[:-1]:
            states[block] = {'f': 0, 'la': 0, 'b': 0}
        states['decoder'] = {'f': 0}

        states['V1']['f'] = inp  # static image input
        for t in range(self.times):

            #states['V1']['f'] = inp[t]  # movie input

            for block in blocks[:-1]:

                # get inputs
                f, la, b = [states[block][key] for key in ['f','la','b']]

                # forward pass
                f, la, b = getattr(self, block)(f, la, b)

                # forward projection
                next_block = blocks[blocks.index(block) + 1]
                states[next_block]['f'] = f

                # lateral projection
                states[block]['la'] = la

                # backward projection
                if block != 'V1':
                    prev_block = blocks[blocks.index(block) - 1]
                    states[prev_block]['b'] = b

        out = self.decoder(states['decoder']['f'])
        return out
