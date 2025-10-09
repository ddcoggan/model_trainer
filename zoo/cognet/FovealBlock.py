import math
from collections import OrderedDict
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
from itertools import product as itp


class Identity(nn.Module):
    def forward(self, x):
        return x

class FovealBlock(nn.Module):
    """
    FLabNet foveal block, with eccentricity-dependent receptive field density,
    both in terms of stride and dilation.
    """

    def __init__(self, in_channels=3, out_channels=64, kernel_size=7,
                 image_size=224, log_lattice=True, dilate=True, stride=2,
                 lattice_size=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.image_size = image_size
        self.log_lattice = log_lattice
        self.dilate = dilate
        assert stride or lattice_size, \
            'Either stride or lattice_size must be specified'
        if stride and lattice_size and stride * lattice_size != image_size:
            UserWarning ('Conflict between kwargs "stride" and "lattice_size", '
                  'using "lattice_size" to generate lattice')
        self.lattice_size = lattice_size or image_size // stride
        self.stride = image_size // self.lattice_size

        # layers
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=False, padding=0)
        self.norm = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ReLU(inplace=True)
        self.output = Identity()

        # lattice
        center = image_size // 2
        ls = torch.linspace(-1, 1, self.lattice_size)
        out_grid = torch.stack(torch.meshgrid(ls, ls, indexing='ij'), dim=0)
        angs = torch.atan2(out_grid[0], out_grid[1])
        self.eccs = out_grid.float().norm(dim=0)
        self.log_eccs = torch.exp(self.eccs) - 1
        if not log_lattice:
            self.centers_i, self.centers_j = torch.meshgrid(
                [torch.arange(0, image_size, self.stride)] * 2)
        else:
            in_grid = torch.stack([self.log_eccs * torch.cos(angs),
                                   self.log_eccs * torch.sin(angs)], dim=0)
            self.centers_i, self.centers_j = (
                    in_grid / in_grid.max() * (center - .5) + center).int()

        # dilation (use log_eccs regardless of log_lattice for consistency)
        if dilate:
            self.dils = self.log_eccs.round().int() + 1
        else:
            self.dils = torch.ones_like(self.log_eccs, dtype=torch.int)
        self.padding = kernel_size // 2 * self.dils.max()

        # kernel unit locations relative to center each dilation factor
        self.base_locs = {
            dil: torch.arange(0, kernel_size * dil, dil) for dil in
            torch.arange(self.dils.min(), self.dils.max() + 1)}
        self.base_locs = {k.item(): v - v[len(v) // 2] for k, v in self.base_locs.items()}

        # precalculate image locations for unfolding
        self.locs_i = torch.empty(self.lattice_size, self.lattice_size,
                                  self.kernel_size, self.kernel_size,
                                  dtype=torch.int)
        self.locs_j = torch.empty(self.lattice_size, self.lattice_size,
                                  self.kernel_size, self.kernel_size,
                                  dtype=torch.int)
        for i, j in itp(range(self.lattice_size), range(self.lattice_size)):
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
        f = self.norm(f)
        f = self.nonlin(f)
        f = self.output(f)
        return f

    def unfold(self, inputs):
        """
        Stacks up the input to each convolution operation
        along the batch dimension, following torch.tensor.unfold, and is
        equivalent to:
        f = F.pad(inputs, self.padding, * 4, value=.5)
        f_unfolded = torch.empty(inputs.shape[0], inputs.shape[1],
            self.lattice_size, self.lattice_size, self.kernel_size,
            self.kernel_size)
        for i, j in itp(range(self.lattice_size), range(self.lattice_size)):
            f_unfolded[:, :, i, j, :, :] = f[..., self.locs_i[i, j],
                                             self.locs_j[i, j]]
        return torch.movedim(f_unfolded, 1, 3).flatten(0, 2)
        """
        f = F.pad(inputs, (self.padding,) * 4, value=.5)
        f = f[..., self.locs_i, self.locs_j]
        f = torch.movedim(f, 1, 3).flatten(0, 2)
        return f

    def fold(self, inputs):
        """Folds the conv output into the feature map shape"""
        f = inputs.squeeze(-2, -1)
        f = f.unflatten(0, (-1, self.lattice_size, self.lattice_size))
        f = torch.movedim(f, 3, 1)
        return f




if __name__ == '__main__':

    # test LGN
    from PIL import Image
    import os
    import requests
    from torchvision.io import read_image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import matplotlib.cm
    import time

    non_fovea = torch.nn.Sequential(OrderedDict([
        ('conv', nn.Conv2d(3, 64, 7, stride=2, bias=False)),
        ('norm', nn.BatchNorm2d(64)),
        ('nonlin', nn.ReLU(inplace=True)),
        ('output', Identity())]))

    test_input = torch.rand(8, 3, 224, 224)
    start = time.time()
    test_output = non_fovea(test_input)
    stop = time.time()
    print(f'Non-foveal forward pass took {stop - start:.4f} seconds')

    os.makedirs('plots', exist_ok=True)

    # get and save sample input image
    url = 'https://www.gstatic.com/webp/gallery/5.jpg'
    image = (Image.open(requests.get(url, stream=True).raw)
             .convert('L').resize((224, 224)))
    image.save(f'plots/sample_image.png')

    # plot blank kernel
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    for i in range(7):
        plt.axvline(x=i, color='k', lw=.5, clip_on=False)
        plt.axhline(y=i, color='k', lw=.5, clip_on=False)
    plt.axvline(x=7, color='k', lw=.5, clip_on=False)
    plt.axhline(y=7, color='k', lw=.5, clip_on=False)
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'kernels_dil-1_blank.png', dpi=300)
    plt.show()

    # plot eccentricity map
    fovea = FovealBlock()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(fovea.eccs, cmap='viridis')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'plots/eccentricity.png', dpi=300)
    plt.show()

    # plot log eccentricity map
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(fovea.log_eccs, cmap='viridis')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(f'plots/log_eccentricity.png', dpi=300)
    plt.show()

    # plot kernels with dilation factors
    num_dils = fovea.dils.max()
    cols = [matplotlib.cm.viridis.colors[int(i)] for i in
            np.linspace(0, 255, num_dils)]
    for dil in range(num_dils):
        col = cols[dil]
        kernel_size = 7 + 6 * dil
        kernel_ls = torch.linspace(0, kernel_size - 1, 7)
        kernel_grid = torch.meshgrid(kernel_ls, kernel_ls, indexing='ij')
        fig, ax = plt.subplots(figsize=(kernel_size / 12, kernel_size / 12))
        ax.set_xlim(0, kernel_size)
        ax.set_ylim(0, kernel_size)
        for i in range(kernel_size):
            plt.axvline(x=i, color='k', lw=.5, clip_on=False)
            plt.axhline(y=i, color='k', lw=.5, clip_on=False)
        plt.axvline(x=kernel_size, color='k', lw=.5, clip_on=False)
        plt.axhline(y=kernel_size, color='k', lw=.5, clip_on=False)
        ax.axis('off')
        # fill square with color at kernel locations
        for i, j in zip(kernel_grid[0].flatten(), kernel_grid[1].flatten()):
            rect = plt.Rectangle((i, j), 1, 1, color=col)
            ax.add_patch(rect)
        plt.tight_layout()
        fig.savefig(f'plots/kernels_dil-{dil}.png', dpi=300)
        plt.show()

    # analyse foveal block with and without dilation and log lattice
    for dilate, log_lattice in itp([True, False], [True, False]):

        fovea = FovealBlock(dilate=dilate, log_lattice=log_lattice)
        start = time.time()
        test_output = fovea(test_input)
        stop = time.time()
        print(f'FovealBlock (dilation: {dilate}, log_lattice: {log_lattice}) '
              f'forward pass took {stop - start:.4f} seconds')

        plot_dir = f'plots/dilate-{dilate}_log-lattice-{log_lattice}'
        os.makedirs(plot_dir, exist_ok=True)

        # plot discrete dilation factors
        num_dils = fovea.dils.max()
        cols = [matplotlib.cm.viridis.colors[int(i)] for i in
                np.linspace(0, 255, num_dils)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(fovea.dils, cmap='viridis')
        ax.axis('off')
        for c, col in enumerate(cols):
            ax.scatter([],[], color=col, label=c+1)
        ax.legend(title='Dilation factor', bbox_to_anchor=(1, 1.1),
                  loc='upper left', frameon=False)
        plt.tight_layout()
        fig.savefig(f'{plot_dir}/dilation_factors.png', dpi=300)
        plt.show()

        # plot kernel centers colored by eccentricity in feature map
        fig, ax = plt.subplots(figsize=(5, 5))
        cm = matplotlib.cm.viridis.colors
        log_eccs_scaled = fovea.log_eccs / fovea.log_eccs.max() * 255
        ax.imshow(image, cmap='gray', alpha=.25)
        ax.scatter(x=fovea.centers_i.flatten(),
                   y=fovea.centers_j.flatten(),
                   s=1, c=[cm[int(i)] for i in log_eccs_scaled.flatten()])
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(f'{plot_dir}/kernel_centers.png', dpi=300)
        plt.show()

        # plot kernel centers colored by dilation factor
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap='gray', alpha=.25)
        ax.scatter(x=fovea.centers_i.flatten(),
                   y=fovea.centers_j.flatten(),
                   s=1, c=[cols[i - 1] for i in fovea.dils.flatten()])
        ax.axis('off')
        plt.tight_layout()
        fig.savefig(f'{plot_dir}/kernel_centers_dilation.png', dpi=300)
        plt.show()

        """
        # plot linear kernel locations
        center = 112
        ls = torch.linspace(1, 224, 112)
        out_grid = torch.stack(torch.meshgrid(ls, ls, indexing='ij'), dim=0) - center
        angs = torch.atan2(out_grid[0], out_grid[1])
        eccs = out_grid.float().norm(dim=0)
        in_grid = torch.stack([eccs * torch.cos(angs),
                               eccs * torch.sin(angs)], dim=0)
        centers_i, centers_j = (
                in_grid / in_grid.max() * (center - .5) + center).int()
        eccs_scaled = eccs / eccs.max() * 255

        cm = matplotlib.cm.viridis.colors
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap='gray', alpha=.25)
        ax.scatter(x=centers_i.flatten(),
                   y=centers_j.flatten(),
                   s=1, c=[cm[int(i)] for i in eccs_scaled.flatten()])
        ax.axis('off')
        plt.tight_layout()
        fig.savefig('kernel_centers_linear.png', dpi=300)
        plt.show()
        """
        
    
