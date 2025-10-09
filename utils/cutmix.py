import numpy as np
import torch

def rand_bbox(size, lam):

    H, W = size[2:4]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(inputs, targets, **kwargs):

    """
    Applies cutmix to a batch of images and labels.
    parameters alpha and beta are hyperparameters of beta distribution
    """

    lam = np.random.beta(kwargs['alpha'], kwargs['beta'])
    rand_index = torch.randperm(inputs.size()[0])
    targets_frgrnd = targets[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
    inputs[:, :, bbx1:bbx2, bby1:bby2] = \
        inputs[rand_index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (
                    inputs.size()[-1] * inputs.size()[-2]))

    return inputs, targets_frgrnd, lam
