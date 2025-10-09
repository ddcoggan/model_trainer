from PIL import Image
import os
from torch.utils.data import Dataset
from torch import stack
import natsort
import numpy as np
import glob


class CustomDataset(Dataset):
    def __init__(self, main_dir, transform=None, targets=None):
        self.main_dir = main_dir
        self.transform = transform
        self.targets = targets

        # if list of image paths is submitted, use that
        if len(main_dir[0]) > 1:
            self.image_paths = main_dir

        # otherwise, get all images in directory
        else:
            self.image_paths = sorted(glob.glob(f'{main_dir}/*'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor_image = self.transform(image)
        if self.targets is not None:
            return tensor_image, self.targets[idx]
        else:
            return tensor_image

