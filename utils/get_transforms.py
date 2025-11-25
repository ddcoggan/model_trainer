import torch.nn
import torchvision.transforms.v2 as transforms
import sys
import os
import shutil
from argparse import Namespace
from .Occlude import Occlude
from .Blur import Blur
from .Noise import Noise

CustomCompose = transforms.Compose
def custom_compose_call(self, img, seed=None):
    for t in self.transforms:
        if seed is not None and hasattr(t, 'set_seed'):
            t.set_seed(seed)
        img = t(img)
    return img
CustomCompose.__call__ = custom_compose_call


class MultipleViews:

    def __init__(self, transforms, seed=False):
        self.transforms = transforms
        self.num_views = len(self.transforms)
        self.seed = seed

    def __call__(self, inputs):
        seed = torch.Generator().get_state() if self.seed else None
        outputs = torch.stack([t(inputs, seed) for t in self.transforms])
        return outputs


class BlurVideo:

    def __init__(self, transform):
        blur_stage = [type(t) for t in transform].index(Blur)
        self.sigmas = transform[blur_stage].sigmas
        self.preblur = transforms.Compose(transform[:blur_stage])
        self.postblur = transforms.Compose(transform[blur_stage+1:])
        blurs = []
        for sigma in self.sigmas:
            if sigma == 0:
                blurs.append(lambda x: x)
            else:
                blurs.append(transforms.GaussianBlur(kernel_size=sigma*6+1,
                                                          sigma=sigma))
        self.blurs = blurs

    def __call__(self, inputs):
        outputs = self.preblur(inputs)
        outputs = torch.stack([self.postblur(blur(outputs)) for blur in
                               self.blurs])
        return outputs


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_transforms(args):

    resize = (0.8, 1.0) if 'weak-resize' in args.transform_type else (
        0.08, 1.0)

    transforms_train = []
    transforms_val = []
    for view in range(args.num_views):

        # standard train transform (normalization is added later)
        transform_train = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomResizedCrop(
                args.image_size, scale=resize, antialias=True),
            transforms.RandomHorizontalFlip(),
        ]

        # add occlusion immediately before random horizontal flip
        if hasattr(args, 'Occlusion') and (
                view in args.Occlusion['views'] or args.num_views == 1):
            idx = [i for i, t in enumerate(transform_train) if isinstance(
                t, transforms.RandomHorizontalFlip)][0]
            transform_train.insert(idx, Occlude(args))

        # contrastive learning transforms
        if 'contrastive' in args.transform_type:
            transform_train.extend([
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)],
                    p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ])
            if 'no-blur' not in args.transform_type:
                transform_train.append(
                    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))

        # add blur transform
        if hasattr(args, 'Blur'):
            transform_train.append(Blur(**args.Blur))

        # add noise transform
        if hasattr(args, 'Noise'):
            transform_train.append(Noise(**args.Noise))

        # normalize
        transform_train.append(normalize)

        # random erase
        if hasattr(args, 'randomerase') and args.randomerase is not False:
            transform_train.append(
                transforms.RandomErasing(value=args.randomerase, inplace=True))

        # add to transform list
        transforms_train.append(transform_train)

        # standard validation transform
        transform_val = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(args.image_size, antialias=True),
            transforms.CenterCrop(args.image_size),
            normalize
        ]
        transforms_val.append(transform_val)

    # special processing for videos (currently works with blur only)
    if args.video:
        transforms_train = BlurVideo(transforms_train[0])
        transforms_val = CustomCompose(transforms_val[0])
        return transforms_train, transforms_val

    # compose transforms
    transforms_train = [CustomCompose(t) for t in transforms_train]
    transforms_val = [CustomCompose(t) for t in transforms_val]

    # wrap in MultipleViews object, if necessary
    if len(transforms_train) == 1:
        transforms_train = transforms_train[0]
        transforms_val = transforms_val[0]
    else:
        transforms_train = MultipleViews(
            transforms_train, seed=args.batch_dependence)
        transforms_val = MultipleViews(
            transforms_val, seed=args.batch_dependence)

    return transforms_train, transforms_val
