import os
import os.path as op
import glob
import sys
import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def plot_conv_filters(layer=None, params_path=None, outpath='filters.png',
                      filters=None):

    os.makedirs(op.dirname(outpath), exist_ok=True)

    if filters is None:
        params = torch.load(params_path, map_location='cpu')
        for key in ['model', 'state_dict']:
            if key in params:
                params = params[key]
                break
        if type(layer) == int:
            layer_key = list(params.keys())[layer]
        else:
            layer_key = layer

        # in case of wrapped model, find the right key
        variations = [f'module.{layer_key}', layer_key[7:]]
        var_counter = 0
        while not layer_key in params:
            layer_key = variations[var_counter]
            var_counter += 1
        filters = params[layer_key].cpu()

            
    num_chan_a,num_chan_b,x,y = filters.shape
    num_filters = num_chan_a if num_chan_b == 3 else num_chan_b if num_chan_a == 3 else num_chan_a*num_chan_b
    grid_size = math.ceil(np.sqrt(num_filters))
    montage_size = (x*grid_size, y*grid_size)
    montage = Image.new(size=montage_size, mode='RGB')

    for i in range(num_filters):
        if num_chan_a == 3:
            image_array = np.array(filters[:, i, :, :].detach().permute(0, 2,
                                                                        1))
        elif num_chan_b == 3:
            image_array = np.array(filters[i, :, :, :].detach().permute(1, 2,
                                                                        0))
        else:
            image_array = filters.flatten(end_dim=1)[i].tile((3,1,1)).permute(1,2,0).numpy()
        image_pos = image_array - image_array.min() # rescale to between 0,255 for PIL
        image_scaled = image_pos * (255.0 / image_pos.max())
        image = Image.fromarray(image_scaled.astype(np.uint8))
        offset_x = (i % grid_size) * x
        offset_y = int(i / grid_size) * y
        montage.paste(image, (offset_x, offset_y))
    montage.save(outpath)

if __name__ == "__main__":
	params_path = (f'/home/tonglab/david/models'
                 f'/cognet_v11/xform-cont/params/015.pt')
	plot_conv_filters('module.V1.conv.weight', params_path,
                      f'{op.dirname(op.dirname(params_path))}/'
                      f'kernel_plots/{op.basename(params_path)[:-3]}.png')
        
        
        
        
