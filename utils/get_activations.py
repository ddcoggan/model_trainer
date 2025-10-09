import glob
import torch
import torch.nn as nn
import os
import os.path as op
import numpy as np
import torchvision.transforms as transforms
from argparse import Namespace
import sys
from torch.utils.data import DataLoader, TensorDataset
import shutil
from types import SimpleNamespace
import itertools
from tqdm import tqdm
import gc

from .save_images import save_image_batch
from .predict import predict
from .get_model import get_model
from .CustomDataset import CustomDataset


def dict_torch_to_numpy(obj):
    for k, v in obj.items():
        if isinstance(v, dict):
            dict_torch_to_numpy(v)
        else:
            obj[k] = v.numpy()


@torch.no_grad()
def get_activations(model=None, architecture=None, image_dir=None, inputs=None,
                    num_workers=2, batch_size=32, layers=None, sampler=None,
                    norm_minibatch=False, save_input_samples=False,
                    sample_input_dir=None, transform=None, shuffle=False, 
                    array_type='numpy'):

    # hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    if model is None:
        model = get_model(architecture, **{'pretrained': True})
    model.to(device)

    # set batch norm behaviour
    model.train() if norm_minibatch else model.eval()

    # image transforms
    if transform is None:
        from .get_transforms import get_transforms
        _, transform = get_transforms()

    # image loader
    if image_dir is None:
        dataset = TensorDataset(inputs, transform=transform)
    else:
        dataset = CustomDataset(image_dir, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, sampler=sampler, pin_memory=True)


    # initialize activations dict
    layers = [layers] if type(layers) is str else layers
    activations = {**{l: [] for l in layers}}

    # define forward hook
    def get_activation(layer):
        def hook(model, input, output):
            # CORnet RT outputs a list of two identical tensors
            if architecture.startswith('cornet_rt'):
                output = output[0]
            activations[layer].append(output.detach().cpu())
        return hook

    # register forward hook
    layers_hidden = [l for l in layers if l not in ['input', 'output']]
    for layer in layers_hidden:

        # get module by decomposing layer string
        layer_path = layer.split('.')
        module = model
        for l in layer_path:
            module = module[int(l)] if l.isnumeric() else getattr(module, l)

        # only add hook if not already registered
        if not len(module.__dict__['_forward_hooks']):
            module.register_forward_hook(get_activation(layer))

    #if T.nGPUs > 1: # TODO: fix this code to allow for data parallelism
    #    model = nn.DataParallel(model)

    # loop through batches
    for batch, inputs in enumerate(tqdm(loader, unit=f"batch({batch_size})")):

        inputs = inputs.to(device)

        # automatic mixed precision
        with torch.autocast(device_type=device.type,
                            dtype=torch.float16):
            outputs = model(inputs)

        # inputs and outputs need to be manually appended
        if 'input' in layers:
            activations['input'].append(inputs.detach().cpu())
        if 'output' in layers:
            activations['output'].append(outputs.detach().cpu())

        # save some input images with class estimates
        if batch == 0 and save_input_samples:
            try:
                responses = predict(outputs, 'ILSVRC2012')
            except:
                responses=None
            if op.isdir(sample_input_dir):
                shutil.rmtree(sample_input_dir)
            os.makedirs(sample_input_dir, exist_ok=True)
            save_image_batch(inputs.detach().cpu(), sample_input_dir,
                              max_images=128, labels=responses)
            

    # post processing (dont overwrite 'activations' as this somehow screws
    # with the hook even though it is not used after this point)
    activations_post = {}

    # post processing
    for layer, acts in activations.items():

        # special handling for vision transformer
        if architecture.startswith('vit') and layer in layers_hidden:
            patch_size = int(np.sqrt(acts.size(-2)))
            activations_post[layer] = acts[:, 1:, :].reshape(
                acts.size(0), patch_size, patch_size, acts.size(2))

        # handle 1-batch loader
        if len(loader) == 1:
            activations_post[layer] = acts[0]

        # normal case, feedforward models, multiple batches
        elif len(acts) == len(loader) and len(acts[-1]) == len(inputs):
            # if responses are in a list, concatenate into a tensor
            activations_post[layer] = torch.concat(acts, dim=0)

        # recurrent models where different cycles are in different list items
        elif len(acts) != len(loader):
            cycles = len(acts) // len(loader)
            activations_post[layer] = {f'cyc{c:02}': torch.concat(
                acts[c::cycles], dim=0) for c in range(cycles)}

        # recurrent models where different cycles are mixed in each list item
        elif type(acts) is list and len(acts[0]) > len(loader):
            cycles = len(acts[0]) // batch_size
            temp_acts = [torch.split(a, a.shape[0] // cycles) for a in acts]
            activations_post[layer] = {f'cyc{c:02}': torch.concat(
                [a[c] for a in temp_acts], dim=0) for c in range(cycles)}



    # numpy conversion
    if array_type == 'numpy':
        dict_torch_to_numpy(activations_post)


    return activations_post


