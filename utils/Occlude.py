import numpy as np
import os
import os.path as op
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
import math
import itertools
from types import SimpleNamespace
import torchvision.transforms.functional as F


class Occlude(torch.nn.Module):

    """
    This class is used to augment images with occlusion. It can be used in
    the same way as any torchvision transform, and is compatible with the
    Compose class. The occluder images are loaded from disk and a separate
    transform is applied before they are superimposed on the input image.
    args: A SimpleNamespace object containing the dataset configuration.
        This must contain the following attributes:
            image_size: The size of the input images
            Occlusion: A SimpleNamespace object containing the occlusion
                configuration. This must contain the following attributes:
                    occluder_dir: top-level directory for occluders dataset
                    form: The form of the occluder as a string, or list
                        thereof. Strings must match one of the directory names
                        in the occlusion dataset, e.g. 'mudSplash'.
                    probability: The probability (0.to 1.) that the occlusion
                        transform is applied to an image, e.g 0.8 will
                        occlude ~80% of the inputs.
                    visibility: The approximate proportion of the image that is
                        to remain visible, i.e. 1. - occluder coverage,
                        or list thereof. 'all' will use all levels available. 
                    color: The color of the occluder, or list thereof. This can
                        be a tuple array of RGB values (0-255) or 'random' to
                        select a random uniform color for each occluder.
                Note: if a list is submitted, the occluder for each input image
                will be configured by sampling randomly from the list.
        preload: If 'paths', occluder images are loaded from disk at runtime.
            If 'tensors', occluder images are loaded from memory at runtime.
    """

    def __init__(self, args, preload='tensors'):

        self.args = args
        self.preload = preload
        self.Occlusion = O = SimpleNamespace(**args.Occlusion)
        self.random_batch = torch.Generator()
        occluder_dir = O.occluder_dir

        # transform for occluder only
        init_occluder_size = {224: 256, 384: 432}[args.image_size]
        if hasattr(O, 'random_resize'):
            occluder_resize = transforms.RandomResizedCrop(
                init_occluder_size, scale=(0.8, 1.0), antialias=True)
        else:
            occluder_resize = transforms.Resize(init_occluder_size, Image.NEAREST)
        self.occluder_transform = transforms.Compose([
            occluder_resize,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(args.image_size)])

        # ensure occluders and visibilities are lists
        occ_forms = [O.form] if isinstance(O.form, str) else O.form
        visibilities = O.visibility if \
            type(O.visibility) in [list, tuple, np.array] else [O.visibility]
            
        # visibility and probability pairings
        if hasattr(O, 'vis_probs'):
            vis_probs = np.array(O.vis_probs)
            vis_probs /= vis_probs.sum()
            self.vis_probs_cumsum = list(np.cumsum(vis_probs))
            
        # specify occluders at instantiation for better training speed
        occ_dirs = {}
        for visibility in visibilities:
            if visibility < 1:
                occ_dirs[visibility] = [f'{occluder_dir}/{form}/{int(visibility*100)}'
                    for form in occ_forms]

        """occluders are stored as a list of tensors, one per occluder 
        directory, allowing mix of L and RGBA"""
        self.occluders = {}
        xform = transforms.PILToTensor()
        for vis, occ_dirs_vis in occ_dirs.items():
            self.occluders[vis] = []
            for occ_dir in occ_dirs_vis:
                if preload == 'images': # method 1: load separate image files
                    paths = glob.glob(f'{occ_dir}/*.png')
                    occs = torch.stack([xform(Image.open(i)) for i in paths], 0)
                elif preload == 'tensors':  # method 2: load single tensor file
                    occs = torch.load(f'{occ_dir}/occluders.pt')
                else:
                    Exception, 'preload option not one of ["images", "tensors"]'
                self.occluders[vis].append(occs)

    # squeeze extra dimension from textured occluders
    # occs = [i.squeeze(0) if i.shape[1] == 4 else i for i in occs]
    
    def set_seed(self, seed):
        """
        Specifying a seed allows for the same occluder to be applied to
        multiple images in a batch. To make random selections consistent
        across a batch, use the self.random_batch generator when calling
        torch.randint or torch.rand. If the seed is not set, self.random_batch
        behaves exactly like the default generator.
        """
        self.random_batch.set_state(seed)

    def __call__(self, image):
    
        O = self.Occlusion

        vis = [v for v, p in zip(O.visibility,
                  self.vis_probs_cumsum) if torch.rand(1) < p][0]

        if vis < 1:

            # select occluder and probability
            occ_type = torch.randint(
                len(self.occluders[vis]), (1,), generator=self.random_batch)
            num_occs = self.occluders[vis][occ_type].shape[0]
            occluder = self.occluders[vis][occ_type][torch.randint(
                num_occs, (1,), generator=self.random_batch)]
            if len(occluder.shape) == 4:  # remove extra dim for textured
                occluder = occluder.squeeze(0)
            
            # ensure range [0,1]
            if occluder.max() > 1.1:
                occluder = occluder / 255
            occluder = occluder.clip(0,1)

            # set occluder color unless texture is used
            if occluder.shape[0] == 1:

                # if multiple colors requested, select one at random
                if type(O.color) == list:
                    fill_col = O.color[torch.randint(
                        len(O.color),(1,), generator=self.random_batch)]
                else:
                    fill_col = O.color

                # if color is specified as RGB, convert to tensor and normalise
                if fill_col == 'random':
                    fill_rgb = torch.rand((3,), generator=self.random_batch)
                else:
                    fill_rgb = torch.tensor(fill_col)
                    if max(fill_rgb) > 1:
                        fill_rgb /= 255

                # colorize
                rgb = torch.tile(fill_rgb[:, None, None], occluder.shape)
                occluder = torch.cat([rgb, occluder], 0)

            # transform
            occluder = self.occluder_transform(occluder)

            # get object and occluder RGB masks from occluder alpha channel 
            occluded_pixels = torch.tile(occluder[3, :, :], dims=(3, 1, 1))
            visible_pixels = 1 - occluded_pixels
            
            # zero occluded pixels in object and visible pixels in occluder
            image *= visible_pixels
            occluder[:3] *= occluded_pixels  # need for untextured

            # replace occluded pixels with occluder (dropping alpha channel)
            image += occluder[:3]

        return image

    """
    # Code for viewing tensor as image
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1,2,0))
    plt.show()
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

if __name__ == "__main__":
    
    import time
    args = SimpleNamespace(**{
        "Occlusion": {
            "color": "random",
            "form": ["barAll04", 'naturalTexturedCropped'],
            "occluder_dir": "/home/tonglab/david/datasets/images/occluders",
            "probability": 1,
            "views": [
                0
            ],
            "visibility": 0.2,
        },
        "image_size": 224,
        "num_epochs": 90,
        "num_views": 1,

    })
    occlude = Occlude(args, preload='images')
    for _ in range(8):
        image = torch.rand(3, 224, 224)
        image_occ = occlude(image)
        image_occ_pil = transforms.ToPILImage()(image_occ)
        image_occ_pil.show()
        time.sleep(2)

        
        
        
