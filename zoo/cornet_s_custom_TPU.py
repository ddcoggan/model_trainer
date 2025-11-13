import math
from collections import OrderedDict
from torch import nn, stack
from argparse import Namespace
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


class CORblock_S_custom(nn.Module):

    def __init__(self, in_channels, out_channels, R, K, scale):
        super().__init__()

        self.times = R

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * scale, out_channels * scale,
                               kernel_size=K, stride=2, padding=int((K-1)/2), bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * scale))
            setattr(self, f'norm3_{t}', nn.BatchNorm2d(out_channels))

    def forward(self, inp):
        x = self.conv_input(inp)

        for t in range(self.times):
            if t == 0:
                skip = self.norm_skip(self.skip(x))
                self.conv2.stride = (2, 2)
            else:
                skip = x
                self.conv2.stride = (1, 1)

            x = self.conv1(x)
            x = getattr(self, f'norm1_{t}')(x)
            x = self.nonlin1(x)

            x = self.conv2(x)
            x = getattr(self, f'norm2_{t}')(x)
            x = self.nonlin2(x)

            x = self.conv3(x)
            x = getattr(self, f'norm3_{t}')(x)

            x += skip
            x = self.nonlin3(x)
            output = self.output(x)

        return output


class CORnetCustomHead(nn.Module):

    def __init__(self, F, out_channels, head_depth):
        super().__init__()
        
        self.F = F
        self.out_channels = out_channels
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        head_sizes = [int(x) for x in (np.linspace(F, 1000, head_depth+1))]

        for out_channel in range(out_channels):
            for layer in range(head_depth):
                setattr(self, f'linear_{out_channel + 1}_{layer+1}', nn.Linear(head_sizes[layer], head_sizes[layer+1]))
                if layer < head_depth:
                    setattr(self, f'nonlin_{out_channel + 1}_{layer+1}', nn.ReLU(inplace=True))

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
        

def CORnet_S_custom(m):
    
    # fill in any missing parameters with defaults
    defaults = {'R': (1,2,4,2), 'K': (3,3,3,3), 'F': (64,128,256,512), 'S': 4, 'out_channels': 1, 'head_depth': 1}
    for param, value in defaults.items():
        if not hasattr(m, param):
            setattr(m, param, value)

    model = nn.Sequential(OrderedDict([('V1', nn.Sequential(OrderedDict([('cycle0', nn.Sequential(OrderedDict([
                                                                                ('conv1', nn.Conv2d(3, m.F[0], kernel_size=7, stride=2, padding=3, bias=False)),
                                                                                ('norm1', nn.BatchNorm2d(m.F[0])),
                                                                                ('nonlin1', nn.ReLU(inplace=True)),
                                                                                ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))),
                                                                          ('CORblock', CORblock_S_custom(m.F[0], m.F[0], R=m.R[0], K=m.K[0], scale=m.S))]))),
                                        ('V2', CORblock_S_custom(m.F[0], m.F[1], R=m.R[1], K=m.K[1], scale=m.S)),
                                        ('V4', CORblock_S_custom(m.F[1], m.F[2], R=m.R[2], K=m.K[2], scale=m.S)),
                                        ('IT', CORblock_S_custom(m.F[2], m.F[3], R=m.R[3], K=m.K[3], scale=m.S)),
                                        ('decoder', CORnetCustomHead(m.F[3], m.out_channels, m.head_depth))]))

    # weight initialization
    for mod in model.modules():
        if isinstance(m, nn.Conv2d):
            n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
            mod.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Linear):
            n = mod.in_features * mod.out_features
            mod.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            mod.weight.data.fill_(1)
            mod.bias.data.zero_()

    return model
