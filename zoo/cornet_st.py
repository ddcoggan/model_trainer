import math
from collections import OrderedDict
from torch import nn, stack
from argparse import Namespace
import torch
import numpy as np
HASH = '1d3f7974'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_ST(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, out_shape=None, V1=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        # preprocessing for V1 only
        self.V1 = V1
        if V1:
            self.conv0 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2, padding=3, bias=False)
            self.norm0 = nn.BatchNorm2d(in_channels)
            self.nonlin0 = nn.ReLU(inplace=True)
            #self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # processing for all layers
        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.skip = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels * self.scale)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale, kernel_size=kernel_size, stride=2, padding=int((kernel_size-1)/2), bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels * self.scale)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.nonlin3 = nn.ReLU(inplace=True)
        self.output = Identity()  # for easy access to this block's output

        self.conv_lateral = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm_lateral = nn.GroupNorm(32, out_channels)
        self.nonlin_lateral = nn.ReLU(inplace=True)


    def forward(self, inp=None, state=None, batch_size=None):

        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:

            if self.V1:
                inp = self.conv0(inp)
                inp = self.norm0(inp)
                inp = self.nonlin0(inp)
                #inp = self.pool0(inp) # removing this pool allows rest of V1 block to be a cornet_st block

            inp = self.conv_input(inp)
            skip = self.norm_skip(self.skip(inp))
            inp = self.conv1(inp)
            inp = self.norm1(inp)
            inp = self.nonlin1(inp)

            inp = self.conv2(inp)
            inp = self.norm2(inp)
            inp = self.nonlin2(inp)

            inp = self.conv3(inp)
            inp = self.norm3(inp)
            inp += skip
            inp = self.nonlin3(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0

        lateral = inp + state

        x = self.conv_lateral(lateral)
        x = self.norm_lateral(x)
        x = self.nonlin_lateral(x)

        state = self.output(x)
        output = state
        return output, state


class CORnetCustomHead(nn.Module):

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
                    setattr(self, f'nonlin_{out_channel + 1}_{layer + 1}', nn.ReLU(inplace=True))

            self.output = Identity()


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


class CORnet_ST(nn.Module):

    def __init__(self, M):
        super().__init__()

        # fill in any missing parameters with defaults
        defaults = Namespace(kernel_size=(3,3,3,3), num_features=(64,128,256,512), out_channels=1, times=5, head_depth=1)
        [setattr(M,param,value) for param, value in defaults._get_kwargs() if not hasattr(M, param)]

        self.times = M.times

        self.V1 = CORblock_ST(3, M.num_features[0], kernel_size=M.kernel_size[0], out_shape=56, V1=True)
        self.V2 = CORblock_ST(M.num_features[0], M.num_features[1], kernel_size=M.kernel_size[1], out_shape=28)
        self.V4 = CORblock_ST(M.num_features[1], M.num_features[2], kernel_size=M.kernel_size[2], out_shape=14)
        self.IT = CORblock_ST(M.num_features[2], M.num_features[3], kernel_size=M.kernel_size[3], out_shape=7)
        self.decoder = CORnetCustomHead(M.num_features[3], M.out_channels, M.head_depth)


    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']

        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = outputs['inp']
            else:  # at t=0 there is no input yet to V2 and up
                this_inp = None
            new_output, new_state = getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            new_outputs = {'inp': inp}
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                new_outputs[block] = new_output
                states[block] = new_state
            outputs = new_outputs

        out = self.decoder(outputs['IT'])
        return out
