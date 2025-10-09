import torch.nn
import torchvision.transforms as transforms
import sys
import os
import os.path as op
import shutil
from types import SimpleNamespace
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .Occlude import Occlude
from .get_transforms import get_transforms

def get_loaders(num_workers, args):

    if not hasattr(args, 'transform'):
        transform_train, transform_val = get_transforms(args)
        batch_size_adjusted = args.batch_size // args.num_views
    else:
        transform_train = args.transform
        transform_val = args.transform
        batch_size_adjusted = args.batch_size

    if hasattr(args, 'dataset_path_train'):
        path_train = args.dataset_path_train
    else:
        path_train = op.expanduser(f'~/Datasets/{args.dataset}/train')
    if hasattr(args, 'dataset_path_val'):
        path_val = args.dataset_path_val
    else:
        path_val = op.expanduser(f'~/Datasets/{args.dataset}/val')

   # if op.isdir(path_train):
    data_train = ImageFolder(path_train, transform=transform_train)
    #if op.isdir(path_val):
    data_val = ImageFolder(path_val, transform=transform_val)
    
    if hasattr(args, 'class_subset'):
        from torch.utils.data import Subset
        idxs = [i for i, image_data in enumerate(data_val.imgs) if (
                image_data[1] in args.class_subset)]
        data_val = Subset(data_val, idxs)

    loader_train = DataLoader(data_train, batch_size=batch_size_adjusted,
                              shuffle=True, num_workers=num_workers,
                              drop_last=True)
    loader_val = DataLoader(data_val, batch_size=batch_size_adjusted,
                            shuffle=True, num_workers=num_workers)

    return loader_train, loader_val




