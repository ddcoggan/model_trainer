import os
import os.path as op
import numpy as np
from PIL import Image
from argparse import Namespace
from torchvision import transforms
from torchvision.utils import save_image
import torch
import sys
from itertools import product as itp
import shutil

from image_processing import tile


def save_image_batch(inputs, out_dir, labels=None, num_views=1, max_images=64):

    os.makedirs(out_dir, exist_ok=True)

    # normalize
    inputs_norm = (inputs - inputs.min()) / (inputs.max() - inputs.min())

    # separate into different views
    if num_views > 1:
        if len(inputs_norm.shape) == 4:
            inputs_norm = inputs_norm.chunk(num_views)
        else:
            inputs_norm = [inputs_norm[:, i].squeeze() for i in range(num_views)]
    else:
        inputs_norm = [inputs_norm]

    # save individual inputs
    batch_size = inputs_norm[0].shape[0]
    out_paths = []
    for i, v in itp(range(min(batch_size, max_images)), range(num_views)):
        label_string = f'_{labels[int((v*batch_size)+i)]}' if labels else ''
        view_string = f'_view-{v}' if num_views > 1 else ''
        out_path = op.join(out_dir, f'{i:04}{view_string}{label_string}.png')
        save_image(inputs_norm[v][i, :, :, :].squeeze(), out_path)
        out_paths.append(out_path)
    
    # tile inputs into a single image
    colgap, colgapfreq = (12, num_views) if num_views > 1 else (None, None)
    num_cols = num_views if num_views > 3 else 8
    tile(out_paths, out_dir + '.png', num_cols=num_cols, base_gap=4,
         colgap=colgap, colgapfreq=colgapfreq)
    #shutil.rmtree(tmp_dir)

