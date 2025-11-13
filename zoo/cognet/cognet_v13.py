import math
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import torch
import torchvision.transforms as transforms
from itertools import product as itp

"""
notes over previous version
backward convtransposes performed for all layers
single convolutional layer combining f, l, b signals
zero tensors for l and b signals are created if they are not passed
"""

class Identity(nn.Module):
    def forward(self, x):
        return x

class LGN(nn.Module):
    
    """
    Simulates higher visual acuity and smaller receptive field size
    in the fovea with the following process:
    - divide the image into different eccentricity bands with concentric squares
    - each eccentricity region is divided into 4 image windows, arranged in a
      rotationally symmetric manner.
    - each window is widened to account for the kernel size
    - decompose image into windows, downsample depending on eccentricity
    - apply convolution to each window
    - rearrange outputs back into original configuration
    - a separate set of windows is defined for the feature map (output) space,
      in case this is different from the image (input) space
    """

    def __init__(self, channels, kernel_size=7, stride=1, image_size=224,
                 num_scales=7, max_scale=2, min_scale=1):

        super().__init__()

        # parameters for eccentricity-dependent convolution
        self.num_scales = num_scales
        self.max_scale = max_scale
        self.scales = torch.linspace(max_scale, min_scale, num_scales)
        self.borders = torch.linspace(0, image_size // 2, num_scales + 1,
                                      dtype=int)
        self.input_size = image_size
        self.output_size = image_size // stride
        self.padding = kernel_size // 2 * max_scale
        self.ic, self.oc = channels

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
        self.input_windows_padded = self.input_windows + (
                    self.padding )
        #for s, scale in enumerate(self.scales):
        #    self.input_windows_padded[s] = (
        #                self.input_windows_padded[s] / scale).int()
        self.input_windows_padded[:, :, :2] -= self.padding
        self.input_windows_padded[:, :, 2:] += self.padding
        self.input_window_sizes = (
                self.input_windows_padded[:, :, 2:] - self.input_windows_padded[
                                                      :, :, :2])

        # operations
        self.fb = nn.Conv2d(self.ic * 2, self.ic, kernel_size=1, bias=False)
        self.pad = nn.ConstantPad2d(self.padding, 0)
        self.conv = nn.Conv2d(self.ic, self.oc, kernel_size=kernel_size,
                              padding=kernel_size//2, stride=stride, bias=False)
        self.norm = nn.GroupNorm(16, self.oc)
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        # allow for retrieval with forward hook
        self.f = Identity()
        self.i = Identity()


    def forward(self, inputs):

        f, _, b, _ = inputs

        # combine image with feedback signals
        b = torch.zeros_like(f) if b is None else b
        f = self.fb(torch.concat([f, b], dim=1))

        # eccentricity-dependent convolution
        f_ecc = torch.zeros(f.shape[0], self.oc, self.output_size,
            self.output_size, device=f.device, dtype=f.dtype)
        f = self.pad(f)
        for s, w in itp(range(self.num_scales), range(4)):

            # window and downsample input
            ti, li, bi, ri = self.input_windows_padded[s, w]
            input_size = tuple(int(x) for x in
                (self.input_window_sizes[s, w]/self.scales[s]))
            f_ds = F.interpolate(
                input=f[..., ti:bi, li:ri],
                size=input_size,
                mode='bilinear', antialias=True)

            # convolution
            f_w = self.conv(f_ds)


            # upsample output
            # extra dim added then removed to avoid 'interp' 3 chan limit
            to, lo, bo, ro = self.output_windows[s, w]
            output_size = tuple([1] + self.output_window_sizes[s, w].tolist())
            f_ecc[..., to:bo, lo:ro] = F.interpolate(
                f_w[:, :, None, :, :], size=output_size, mode='nearest').squeeze()

        # final nonlin and maxpool
        f = self.nonlin(f_ecc)
        f, i = self.pool(f)
        
        # allow for retrieval with forward hook
        f = self.f(f)
        l = None
        b = None
        i = self.i(i)
        
        return f, l, b, i


class CogBlock(nn.Module):

    """
    Generic model block for CogNet
    """

    def __init__(self, channels, kernel_size=3, stride=1, scale=4):
        super().__init__()

        self.prev_channels = pc = channels[0]
        self.in_channels = ic = channels[1]
        self.out_channels = oc = channels[2]
        self.stride = stride
        self.scale = scale
        sc = oc * scale

        # integration convolutions
        self.flb = nn.Conv2d(ic * 3, ic, kernel_size=1, bias=False)

        # feed forward connections
        self.conv_input = nn.Conv2d(ic, oc, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size//2)
        self.conv_skip = nn.Conv2d(oc, oc, kernel_size=1, stride=1, bias=False)
        self.norm_skip = nn.GroupNorm(16, oc)

        self.conv1 = nn.Conv2d(oc, sc, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(16, sc)
        self.nonlin1 = nn.ReLU()

        self.conv2 = nn.Conv2d(sc, sc,
                               kernel_size=kernel_size, stride=1,
                               padding=kernel_size//2, bias=False)
        self.norm2 = nn.GroupNorm(16, sc)
        self.nonlin2 = nn.ReLU()

        self.conv3 = nn.Conv2d(sc, oc, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(16, oc)
        self.nonlin3 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2,
                                    return_indices=True)

        # backward connection
        self.unpool_back = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_back = nn.ConvTranspose2d(oc, pc, kernel_size=3, stride=1,
                                            padding=1, bias=False)
        self.norm_back = nn.GroupNorm(min(16, pc), pc)
        self.nonlin_back = nn.ReLU()
        
        # allow for retrieval with forward hook
        self.f = Identity()
        self.l = Identity()
        self.b = Identity()
        self.i = Identity()


    def forward(self, inputs):

        f, l, b, i = inputs

        # input convolution
        f = self.conv_input(f)

        # combine with lateral and feed back signals
        l = torch.zeros_like(f) if l is None else l
        b = torch.zeros_like(f) if b is None else b
        f = self.flb(torch.concat([f, l, b], dim=1))

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

        # outgoing lateral connection
        l = f.clone()
        
        # outgoing backward connection
        b = self.unpool_back(f, i)
        b = self.conv_back(b)
        b = self.norm_back(b)
        b = self.nonlin_back(b)

        # combine with skip, final nonlin and avgpool
        f += skip
        f = self.nonlin3(f)
        f, i = self.maxpool(f)
        
        # allow for retrieval with forward hook
        f = self.f(f)
        l = self.l(l)
        b = self.b(b)
        i = self.i(i)

        return f, l, b, i


class CogDecoder(nn.Module):

    def __init__(self, in_channels, out_features, head_depth, head_width):
        super().__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.head_depth = head_depth
        self.avgpool = nn.AdaptiveAvgPool2d(head_width)
        self.flatten = nn.Flatten()
        self.head_sizes = torch.linspace(in_channels * (head_width ** 2), 1000,
                                 head_depth + 1, dtype=int)
        self.f = Identity()

        # flexibly generate decoder based on head_depth
        for layer in range(head_depth):
            setattr(self, f'linear_{layer + 1}',
                    nn.Linear(self.head_sizes[layer],
                              self.head_sizes[layer + 1]))
            if layer < head_depth - 1:
                setattr(self, f'nonlin_{layer + 1}', nn.ReLU())

    def forward(self, inp):

        x = self.avgpool(inp)
        x = self.flatten(x)

        for layer in range(self.head_depth):
            x = getattr(self, f'linear_{layer + 1}')(x)
            if layer < self.head_depth - 1:
                x = getattr(self, f'nonlin_{layer + 1}')(x)

        x = self.f(x)

        return x


class CogNet(nn.Module):

    def __init__(self, cycles=4):
        super().__init__()

        self.cycles = cycles
        chn = [3,64,64,64,64,64]
        hd = 2
        hw = 3

        self.LGN = LGN(channels=chn[:2])
        self.V1 = CogBlock(channels=chn[0:3])
        self.V2 = CogBlock(channels=chn[1:4])
        self.V4 = CogBlock(channels=chn[2:5])
        self.IT = CogBlock(channels=chn[3:6])
        self.decoder = CogDecoder(in_channels=chn[-1], out_features=1000,
                                  head_depth=hd, head_width=hw)

    def forward(self, inp):

        # adjust cycles if video input is submitted
        cycles = inp.shape[0] if len(inp.shape) == 5 else self.cycles

        # initialize block states
        blocks = ['LGN', 'V1', 'V2', 'V4', 'IT']
        states = {b: {k: None for k in 'flbi'} for b in ['input', *blocks]}

        for c in range(cycles):
            
            # if image, reinput at each cycle; if movie, input frame sequence
            states['input']['f'] = inp if len(inp.shape) == 4 else inp[c]
            
            for cur_blk in blocks:

                # get names of previous and next blocks
                prv_blk = ['input', *blocks][blocks.index(cur_blk)]
                nxt_blk = [*blocks, None][blocks.index(cur_blk) + 1]

                # collate inputs (prev f, current l, next b, prev i)
                inputs = [states[prv_blk]['f'], states[cur_blk]['l'],
                          states[nxt_blk]['b'], states[prv_blk]['i']]

                # forward pass
                outputs = getattr(self, cur_blk)(inputs)

                # store outputs
                states[cur_blk] = {k: v for k, v in zip('flbi', outputs)}
            
            output = self.decoder(states['IT']['f'])

        return output


# plot window borders with each scale a different color (rainbow)
def plot_windows(LGN, outdir):

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    os.makedirs(outdir, exist_ok=True)

    # input windows
    fig, ax = plt.subplots()
    image_size = LGN.input_size
    ax.imshow(torch.ones(3, image_size, image_size).permute(1, 2, 0))
    colors = plt.cm.rainbow(torch.linspace(0, 1, LGN.num_scales))
    for win, window in enumerate(LGN.input_windows):
        for w in range(4):
            top, left, bottom, right = window[w].clone()
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     linewidth=1, edgecolor=colors[win],
                                     facecolor=colors[win], alpha=0.5)
            ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top','left','bottom','right']].set_visible(False)
    plt.savefig(op.join(outdir, 'input_windows.png'))
    plt.close()

    # input windows with padding
    fig, ax = plt.subplots()
    image_size = LGN.input_size + LGN.padding * 2
    ax.imshow(torch.ones(3, image_size, image_size).permute(1, 2, 0))
    colors = plt.cm.rainbow(torch.linspace(0, 1, LGN.num_scales))
    for win, window in enumerate(LGN.input_windows_padded):
        for w in range(4):
            top, left, bottom, right = window[w].clone()
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     linewidth=1, edgecolor=colors[win],
                                     facecolor=colors[win], alpha=0.5)
            ax.add_patch(rect)
    # add original image border
    rect = patches.Rectangle((LGN.padding, LGN.padding),
                             LGN.input_size, LGN.input_size,
                             linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)
    plt.savefig(op.join(outdir, 'input_windows_padded.png'))
    plt.close()

    # input windows
    fig, ax = plt.subplots()
    image_size = LGN.input_size
    ax.imshow(torch.ones(3, image_size, image_size).permute(1, 2, 0))
    colors = plt.cm.rainbow(torch.linspace(0, 1, LGN.num_scales))
    for win, window in enumerate(LGN.input_windows):
        for w in range(4):
            top, left, bottom, right = window[w].clone()
            rect = patches.Rectangle((left, top), right - left, bottom - top,
                                     linewidth=1, edgecolor=colors[win],
                                     facecolor=colors[win], alpha=0.5)
            ax.add_patch(rect)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'left', 'bottom', 'right']].set_visible(False)
    plt.savefig(op.join(outdir, 'output_windows.png'))
    plt.close()

if __name__ == "__main__":

    version = 'v13'
    import sys
    from PIL import Image
    import os.path as op
    import os
    import glob
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    model = CogNet(cycles=10)
    plot_windows(model.LGN, op.expanduser(
                 f'~/david/models/cognet_v13/xform-cont/window_plots'))

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
        plot_conv_filters('module.V1.conv_lat2.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_l.png')
        plot_conv_filters('module.V2.conv_back2.weight', params_path,
                          f'{op.dirname(op.dirname(params_path))}/'
                          f'kernel_plots/epoch-{epoch}_b.png')

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
                image_paths = {'V1-l': [], 'V2-b': [], 'sum': []}


                for i in range(batch_size):
                    for cycle in states:
                        images = []
                        for layer, conn in zip(['V1', 'V2'], ['l', 'b']):
                            image = states[cycle][layer][conn][i].detach().cpu().squeeze()
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

                        # sum V1-l and V2-b
                        image_array = np.sum(np.array(images), axis=0)
                        central_min = image_array[5:-5, 5:-5, :].min()
                        central_max = image_array[5:-5, 5:-5, :].max()
                        image_clip = np.clip(image_array, central_min,
                                             central_max) - central_min
                        image_scaled = image_clip * (255.0 / central_max)
                        image_PIL = Image.fromarray(
                            image_scaled.astype(np.uint8))
                        outpath = (f'{feature_maps_dir}/sum_im{i}_{cycle}.png')
                        image_PIL.save(outpath)
                        image_paths[f'sum'].append(outpath)

                for maptype, paths in image_paths.items():
                    outpath = f'{feature_maps_dir}/{maptype}_tiled.png'
                    tile(paths, outpath, num_cols=10, base_gap=0,
                         colgap=1, colgapfreq=1,
                         rowgap=8, rowgapfreq=1)
                break


