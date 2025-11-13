import math
from collections import OrderedDict
import itertools
import torch
from torch import nn, linspace
import torchvision.transforms as transforms
interp = nn.functional.interpolate

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

class COGV1(nn.Module):

    def __init__(self, out_channels=128, kernel_size=7, stride=2, image_size=224, num_scales=7,
                 max_scale=2, min_scale=1):
        super().__init__()

        # parameters for eccentricity-dependent convolution
        self.num_scales = num_scales
        self.max_scale = max_scale
        self.scales = linspace(max_scale, min_scale, num_scales)  # start at max
        self.borders = linspace(0, image_size // 2, num_scales + 1, dtype=int)
        self.output_size = image_size // stride
        self.padding = kernel_size // 2
        self.oc = out_channels

        """ 
        Define concentric squares of pixels relevant for each scale. 
            - each hollow square is defined by 4 windows, arranged in a 
              rotationally symmetric manner.
            - each image window is then padded to account for the kernel size
            - a separate set of windows is defined for the featuremap space, 
              in case this is different from the image space
        """

        # define image windows for kernel centers (unpadded)
        self.input_windows = torch.empty((num_scales, 4, 4), dtype=int)
        for s, scale in enumerate(self.scales):
            a = self.borders[s]
            b = self.borders[s + 1]
            c = image_size - b
            d = image_size - a
            top = torch.tensor([a, a, b, c])
            lft = torch.tensor([b, a, d, b])
            bot = torch.tensor([c, b, d, d])
            rgt = torch.tensor([a, c, c, d])
            self.input_windows[s] = torch.stack([top, lft, bot, rgt])

        # translate windows into the output space
        # this currently assumes that output dims = input dims / stride
        self.output_windows = self.input_windows // stride
        self.output_window_sizes = (
                self.output_windows[:, :, 2:] - self.output_windows[:, :, :2])

        # actual image window needs to account for initial image padding
        # then downsampling then kernel size
        self.input_windows_padded = self.input_windows + (self.padding * max_scale)
        for s, scale in enumerate(self.scales):
            self.input_windows_padded[s] = (self.input_windows_padded[s] / scale).int()
            self.input_windows_padded[s, :, :2] -= self.padding
            self.input_windows_padded[s, :, 2:] += self.padding
        self.input_window_sizes = (
            self.input_windows_padded[:, :, 2:] - self.input_windows_padded[:, :, :2])

        # layers
        self.pad = nn.ConstantPad2d(self.padding, 0)
        self.conv = nn.Conv2d(3, self.oc, kernel_size=kernel_size,
                              padding=0, stride=stride, bias=False)
        self.norm1 = nn.BatchNorm2d(self.oc)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.oc, self.oc, kernel_size=3, stride=1, padding=1,
                             bias=False)
        self.norm2 = nn.BatchNorm2d(self.oc)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.output = Identity()
        

    def forward(self, inp):

        out = torch.zeros(inp.shape[0], self.oc, self.output_size, self.output_size,
                          device=inp.device, dtype=inp.dtype)
        x = self.pad(inp)
        for s, w in itertools.product(range(self.num_scales), range(4)):

            # downsample input
            ti, li, bi, ri = self.input_windows_padded[s, w]
            input_size = tuple(self.input_window_sizes[s, w].tolist())
            x_downsampled = interp(input=x[..., ti:bi, li:ri], size=input_size,
                                   mode='bilinear', antialias=True)

            # convolution
            f = self.conv(x_downsampled)

            # upsample output (extra dim added then removed to avoid 'interp' 3 chan limit
            to, lo, bo, ro = self.output_windows[s, w]
            output_size = tuple([1] + self.output_window_sizes[s, w].tolist())
            out[..., to:bo, lo:ro] = interp(f[:,:,None,:,:], size=output_size,
                                        mode='nearest').squeeze()

        x = self.norm1(out)
        x = self.nonlin1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlin2(x)

        return self.output(x)


class CORblock_S(nn.Module):

    scale = 4  # scale of the bottleneck convolution channels

    def __init__(self, in_channels, out_channels, times=1):
        super().__init__()

        self.times = times

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.skip = nn.Conv2d(out_channels, out_channels,
                              kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels * self.scale,
                               kernel_size=1, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels * self.scale, out_channels * self.scale,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.nonlin2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels * self.scale, out_channels,
                               kernel_size=1, bias=False)
        self.nonlin3 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

        # need BatchNorm for each time step for training to work well
        for t in range(self.times):
            setattr(self, f'norm1_{t}', nn.BatchNorm2d(out_channels * self.scale))
            setattr(self, f'norm2_{t}', nn.BatchNorm2d(out_channels * self.scale))
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


def CORnet_S():
    model = nn.Sequential(OrderedDict([
        ('V1', COGV1(128)),
        ('V2', CORblock_S(128, 128, times=2)),
        ('V4', CORblock_S(128, 256, times=4)),
        ('IT', CORblock_S(256, 512, times=2)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 1000)),
            ('output', Identity())
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        # nn.Linear is missing here because I originally forgot 
        # to add it during the training of this network
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model
