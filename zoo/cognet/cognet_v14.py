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
removed cortical magnification mechanism as reconstructed feature maps cannot 
    be reconstructed into a smooth feature map
no pooling/unpooling, just stride 2 convolutions
much smaller model to speed up training (< 3m parameters)
leaky relu to allow gradient pass-through
single, wider decoder layer
32 group norm channels
"""

class Identity(nn.Module):
    def forward(self, x):
        return x


class LGN(nn.Module):

    def __init__(self, channels, kernel_size=11, stride=2):
        super().__init__()

        # parameters
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        padding = kernel_size // 2

        # layers
        self.conv = nn.Conv2d(3, channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        #self.norm = nn.GroupNorm(32, channels)
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.f = Identity()  # allow for retrieval with forward hook


    def forward(self, inputs):
        f = self.conv(inputs)
        # f = self.norm(f)
        f = self.nonlin(f)
        f = self.pool(f)
        f = self.f(f)
        return f


class CogBlock(nn.Module):

    """
    Generic model block for CogNet
    """

    def __init__(self, channels, kernel_size=3, stride=2, scale=4, 
                 lat_pad=1, back_pad=3):
        super().__init__()

        self.prev_channels = pc = channels[0]
        self.in_channels = ic = channels[1]
        self.out_channels = oc = channels[2]
        self.stride = stride
        self.scale = scale
        sc = ic * scale

        # feed forward connections
        self.conv_skip = nn.Conv2d(ic, oc, kernel_size=1, stride=2, bias=False)
        self.norm_skip = nn.GroupNorm(32, oc)

        self.conv1 = nn.Conv2d(ic, sc, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(32, sc)
        self.nonlin1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(sc, sc, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.norm2 = nn.GroupNorm(32, sc)
        self.nonlin2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(sc, oc, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(32, oc)
        self.nonlin3 = nn.LeakyReLU()

        # lateral connection
        self.conv_lat = nn.ConvTranspose2d(oc, ic, kernel_size=kernel_size,
            stride=stride, padding=1, output_padding=lat_pad, bias=False)
        self.norm_lat = nn.GroupNorm(32, ic)
        self.nonlin_lat = nn.LeakyReLU()

        # backward connection
        if self.prev_channels != 3:
            self.conv_back = nn.ConvTranspose2d(oc, pc, kernel_size=5, stride=4,
                padding=2, output_padding=back_pad, bias=False)
            self.norm_back = nn.GroupNorm(32, pc)
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
        if self.prev_channels != 3:
            b = self.conv_back(f)
            b = self.norm_back(b)
            b = self.nonlin_back(b)
        else:
            b = 0 

        # combine with skip, final nonlin
        f = f + skip
        f = self.nonlin3(f)
        
        # allow for retrieval with forward hook
        f = self.f(f)
        l = self.l(l)
        b = self.b(b)

        return f, l, b


class CogDecoder(nn.Module):

    def __init__(self, in_channels, out_features, head_depth, head_width):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(head_width)
        self.flatten = nn.Flatten()
        self.head_sizes = torch.linspace(
            in_channels * (head_width ** 2), 1000, head_depth + 1, dtype=int)
        self.f = Identity()

        # flexibly generate decoder based on head_depth
        for layer in range(head_depth):
            setattr(self, f'linear_{layer + 1}',
                    nn.Linear(self.head_sizes[layer],
                              self.head_sizes[layer + 1]))
            if layer < head_depth - 1:
                setattr(self, f'nonlin_{layer + 1}', nn.ReLU())

    def forward(self, inputs):

        f = self.avgpool(inputs)
        f = self.flatten(f)

        for layer in range(self.head_depth):
            f = getattr(self, f'linear_{layer + 1}')(f)
            if layer < self.head_depth - 1:
                f = getattr(self, f'nonlin_{layer + 1}')(f)

        f = self.f(f)

        return f

"""
def count_states(states):
    for layer, cycles in states.items():
        for cycle, connections in cycles.items():
            if type(connections) is dict:
                for connection, tensor in connections.items():
                    if tensor is not None:
                        print(f'{layer}: {cycle}: {connection}: {tensor.shape}')
            else:
                print(f'{layer}: {cycle}: {connections.shape}')
"""

class CogNet(nn.Module):

    def __init__(self, cycles=5):
        super().__init__()

        self.cycles = cycles
        chn = [3, 64, 128, 256, 512, 1024]

        self.LGN = LGN(chn[1])
        self.V1 = CogBlock(channels=chn[0:3])
        self.V2 = CogBlock(channels=chn[1:4])
        self.V4 = CogBlock(channels=chn[2:5])
        self.IT = CogBlock(channels=chn[3:6], lat_pad=0, back_pad=1)
        self.decoder = CogDecoder(in_channels=chn[-1], out_features=1000, 
                                  head_depth=1, head_width=3)
        self.blocks = ['V1', 'V2', 'V4', 'IT', 'decoder']

    def forward(self, inp):

        # adjust cycles if video input is submitted
        cycles = inp.shape[0] if len(inp.shape) == 5 else self.cycles

        # initialize block states
        blocks = ['V1', 'V2', 'V4', 'IT']
        empty_state = {'f': 0, 'l': 0, 'b': 0}
        states = {block: empty_state.copy() for block in blocks}
        movie = len(inp.shape) == 5
        from_lgn = self.LGN(inp[0] if movie else inp)

        for c in range(cycles):

            # if movie, input next frame
            if movie and c > 0:
                from_lgn = LGN(inp[c])

            for blk in blocks:
                prv_blk = blocks[
                    blocks.index(blk) - 1] if blk != 'V1' else None
                nxt_blk = blocks[blocks.index(blk) + 1] if blk != 'IT' else None

                # get feedforward inputs from inp or prev block
                f = from_lgn if blk == 'V1' else states[prv_blk]['f']

                # get lateral inputs from current block
                l = states[blk]['l'] if c > 0 else 0

                # get feedback inputs from next block
                b = states[nxt_blk]['b'] if blk != 'IT' and c != 0 else 0

                # forward pass
                inputs = f + l + b
                outputs = getattr(self, blk)(inputs)

                # store outputs
                states[blk] = {key: item for key, item in zip('flb', outputs)}

            output = self.decoder(states['IT']['f'])

        return output


if __name__ == "__main__":

    version = 'v14'
    import sys
    from PIL import Image
    import os.path as op
    import os
    import glob
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torchinfo


    model = CogNet(cycles=5)


    print(torchinfo.summary(model, input_size=(1, 3, 224, 224)))

    sys.path.append(op.expanduser('~/david/master_scripts/image'))
    from image_processing import tile

    sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
    from utils import plot_conv_filters

    model_dir = op.expanduser(f'~/david/projects/p020_activeVision/models/'
                              f'cognet_v10/{version}')
    params_dir = f'{model_dir}/params'
    params_paths = [f'{params_dir}/012.pt', f'{params_dir}/025.pt']

    for params_path in params_paths:
        epoch = op.basename(params_path)[:-3]

        plot_conv_filters('module.V1.conv.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_f.png')
        plot_conv_filters('module.V1.conv_back2.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_l.png')

        feature_maps_dir = (f'{model_dir}/feature_maps/epoch-{epoch}')
        os.makedirs(feature_maps_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
        ])

        model = nn.DataParallel(CogNet(cycles=10, return_states=True).cuda())
        params = torch.load(params_path)
        model.load_state_dict(params['model'])
        batch_size = 4
        data = ImageFolder(f'~/Datasets/ILSVRC2012/val', transform=transform)
        loader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, num_workers=2)
        model = model.module.cpu()
        for batch, (inputs, targets) in enumerate(loader):
            if batch == 0:
                #inputs.cuda()
                states = model(inputs)
                image_paths = {'V1-f': [], 'V2-b': [], 'sum': []}

                for i in range(batch_size):
                    for cycle in states:
                        images = []
                        for layer, conn in zip(['V1'], ['f', 'b']):
                            image = states[layer][cycle][conn][i].detach().cpu().squeeze()
                            image_array = np.array(image.permute(1, 2, 0))
                            images.append(image_array.copy())
                            central_min = image_array[5:-5,5:-5,:].min()
                            central_max = image_array[5:-5, 5:-5, :].max()
                            image_clip = np.clip(image_array, central_min,
                                                 central_max) - central_min
                            image_scaled = image_clip * (255.0 / central_max)
                            image_PIL = Image.fromarray(image_scaled.astype(np.uint8))
                            outpath = (f'{feature_maps_dir}/{layer}-'
                                       f'{conn}_im{i}_cyc{cycle}.png')
                            image_PIL.save(outpath)
                            image_paths[f'{layer}-{conn}'].append(outpath)

                for maptype, paths in image_paths.items():
                    outpath = f'{feature_maps_dir}/{maptype}_tiled.png'
                    tile(paths, outpath, num_cols=10, base_gap=0,
                         colgap=1, colgapfreq=1,
                         rowgap=8, rowgapfreq=1)
                break


