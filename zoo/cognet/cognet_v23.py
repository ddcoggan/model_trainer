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
Retinotopic LGN layer added with increasing RF size and stride at wider eccentricities
"""

class Identity(nn.Module):
    def forward(self, x):
        return x


class LGN(nn.Module):
    """
    FLabNet LGN block, with eccentricity-dependent receptive field density,
    both in terms of stride and dilation.
    """

    def __init__(self, in_channels=3, out_channels=64, kernel_size=7,
                 image_size=224, num_centers=112):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.num_centers = num_centers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=False)
        center = image_size // 2

        # kernel center locations
        ls = torch.linspace(-1, 1, num_centers)
        out_grid = torch.stack(torch.meshgrid(ls, ls), dim=0)
        angs = torch.atan2(out_grid[0], out_grid[1])
        eccs = out_grid.float().norm(dim=0)
        log_eccs = torch.exp(eccs) - 1
        in_grid = torch.stack([log_eccs * torch.cos(angs),
                               log_eccs * torch.sin(angs)], dim=0)
        self.centers_i, self.centers_j = (
                in_grid / in_grid.max() * (center - .5) + center).int()

        # dilation factors
        self.dils = log_eccs.round().int() + 1
        self.padding = kernel_size // 2 * self.dils.max()

        # kernel unit locations relative to center for each dilation factor
        self.base_locs = {
            dil: torch.arange(0, kernel_size * dil, dil) for dil in
            torch.arange(self.dils.min(), self.dils.max() + 1)}
        self.base_locs = {k.item(): v - v[len(v) // 2] for k,
        v in self.base_locs.items()}

        # precalculate locations for unfolding
        self.locs_i = torch.empty(self.num_centers, self.num_centers,
                                  self.kernel_size, self.kernel_size,
                                  dtype=torch.int)
        self.locs_j = torch.empty(self.num_centers, self.num_centers,
                                  self.kernel_size, self.kernel_size,
                                  dtype=torch.int)
        for i, j in itp(range(self.num_centers), range(self.num_centers)):
            center_i = self.centers_i[i, j]
            center_j = self.centers_j[i, j]
            dil = self.dils[i, j]
            base_locs = self.base_locs[dil.item()]
            self.locs_i[i, j] = torch.stack([center_i + base_locs] *
                                            kernel_size)
            self.locs_j[i, j] = torch.stack([center_j + base_locs] *
                                            kernel_size).T

    def forward(self, inputs):
        f = self.unfold(inputs)
        f = self.conv(f)
        f = self.fold(f)
        return f

    def unfold(self, inputs):
        inputs_padded = F.pad(inputs, (self.padding,) * 4, value=.5)
        inputs_unfolded = inputs_padded[..., self.locs_i, self.locs_j]
        inputs_stacked = torch.movedim(inputs_unfolded, 1, 3).flatten(0, 2)
        return inputs_stacked

    def fold(self, inputs):
        f = inputs.squeeze(-2, -1)
        f = f.unflatten(0, (-1, self.num_centers, self.num_centers))
        f = torch.movedim(f, 3, 1)
        return f


class FLaBBlock(nn.Module):

    """
    Generic model block for FLaB, with forward, lateral, and an optional backward connection
    :param channels: list of input channels for previous, current and next block
    :param norm_groups: list of groupnorm groups for previous, current and next block
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
        #self.global_pool = nn.AdaptiveAvgPool2d(1)
        #self.conv_se1 = nn.Conv2d(oc, oc // 4, kernel_size=1, bias=False)
        #self.nonlin_se = nn.LeakyReLU()
        #self.conv_se2 = nn.Conv2d(oc // 4, oc, kernel_size=1, bias=False)
        #self.sig = nn.Sigmoid()

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
        #se = self.global_pool(f)
        #se = self.conv_se1(se)
        #se = self.nonlin_se(se)
        #se = self.conv_se2(se)
        #se = self.sig(se)
        se = 1
        
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

        self.LGN = LGN(in_channels=chn[0], out_channels=chn[1])
        self.V1 = FLaBBlock(channels=chn[0:3], cardinality=crd[0],
                            norm_groups=gnm[0:3], back=False)
        self.V2 = FLaBBlock(channels=chn[1:4], cardinality=crd[1],
                            norm_groups=gnm[1:4])
        self.V4 = FLaBBlock(channels=chn[2:5], cardinality=crd[2],
                            norm_groups=gnm[2:5])
        self.IT = FLaBBlock(channels=chn[3:6], cardinality=crd[3],
                            norm_groups=gnm[3:6], lat_pad=1, back_pad=3)
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
                try:
                    inputs = f + l + b  # sum inputs
                except:
                    print('sdf')
                outputs = getattr(self, block)(inputs)  # forward pass
                states[block] = {k: v for k, v in zip('flb', outputs)}  # store

            states['decoder']['f'] = self.decoder(states['IT']['f'])

        return states['decoder']['f']

