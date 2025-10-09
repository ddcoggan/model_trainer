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

class Blur:

    """
    This class is used to augment images with Gaussian blur.
        sigmas: sigma or list thereof for the Gaussian blur (in pixels)
        probs: probability of applying blur or list thereof for per-sigma probs
    If both sigmas and probs are lists, they should be the same length
    """

    def __init__(self, sigmas, probs):
        super().__init__()
        self.sigmas = sigmas if not hasattr(sigmas, 'len') else sigmas
        self.probs = probs
        self.blur_transforms = {sigma: transforms.GaussianBlur(
            kernel_size=sigma*6+1, sigma=sigma) for sigma in sigmas if sigma > 0}
        self.blur_transforms[0] = lambda x: x
        if type(probs) is list:
            self.cum_prob = torch.cumsum(torch.tensor(probs), 0)

    def __call__(self, image):
        rand = torch.rand(1)
        if hasattr(self, 'cum_prob'):
            sigma = self.sigmas[torch.searchsorted(self.cum_prob, rand)]
        elif rand < self.probs:
            sigma = self.sigmas[torch.randint(len(self.sigmas), (1,))]
        else:
            return image
        return self.blur_transforms[sigma](image)

    """
    # Code for viewing tensor as image
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1,2,0))
    plt.show()
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

        
if __name__ == '__main__':

    sigmas = [0, 1, 2, 3]
    probs = 0.5
    blur1 = Blur(sigmas, probs)
    blur2 = blur1
    setattr(blur2, 'sigmas', [1])
    print(blur1.sigmas)
    print(blur2.sigmas)
    print('sdf')
