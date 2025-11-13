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

class Noise(torch.nn.Module):

    """
    This class is used to augment images with Gaussian or Fourier Noise with a
    given SSNR(s) and probability(s).
    """

    def __init__(self, noise_types, ssnrs, probs, mean_magnitude_path=None):

        self.noise_types = noise_types
        self.ssnrs = ssnrs
        self.probs = probs
        if 'fourier' in noise_types:
            self.mean_fourier_magnitude = torch.load(mean_magnitude_path)
        if type(probs) is list:
            self.cum_prob = torch.cumsum(torch.tensor(probs), 0)

    def __call__(self, image):

        # select ssnr
        rand = torch.rand(1)
        if hasattr(self, 'cum_prob'):
            ssnr = self.ssnr[torch.searchsorted(self.cum_prob, rand)]
        elif rand < self.probs:
            ssnr = self.ssnrs[torch.randint(len(self.ssnrs), (1,))]
        else:
            return image

        assert image.max() < 1.0001 and image.min() >= 0, (
            f'Value range outside [0, 1]: min: {image.min()},'
            f' max: {image.max()}')

        # select noise type
        if type(self.noise_types) is str:
            noise_type = self.noise_types
        else:
            noise_type = self.noise_types[torch.randint(len(
                self.noise_types), (1,))]

        # apply noise
        if noise_type == 'gaussian':
            signal = (image - 0.5) * ssnr + 0.5
            noise = torch.normal(0, (1 - ssnr) / 6, image.shape)
            image = torch.clamp(signal + noise, 0, 1)

        elif noise_type == 'fourier':
            noised_image = torch.zeros_like(image)
            for i, image_channel in enumerate(image):
                image_fft = np.fft.fft2(image_channel)
                image_fft_phase = np.angle(image_fft)
                # np.random.shuffle(image_fft_phase) # wrong! only does 1st dim
                np.random.shuffle(image_fft_phase.flat)  # correct way to do it
                image_fft_shuffled = np.multiply(
                    self.mean_fourier_magnitude[i],
                    np.exp(1j * image_fft_phase))
                image_recon = abs(np.fft.ifft2(image_fft_shuffled))
                image_recon = (image_recon - np.min(image_recon)) / (
                            np.max(image_recon) - np.min(image_recon))
                signal = (image_channel - 0.5) * ssnr + 0.5
                noise = (image_recon - 0.5) * (1 - ssnr)
                noise = torch.tensor(noise).float().to(image.device)
                noised_image[i] = torch.clamp(signal + noise, 0, 1)
            image = noised_image

        return image

    """
    # Code for viewing tensor as image
    import matplotlib.pyplot as plt
    plt.imshow(image.permute(1,2,0))
    plt.show()
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

        
        
        
