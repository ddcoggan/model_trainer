"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
Adapted by David Coggan on 06/21/2022
Responses are normalized here in the loss function, not within the model. This requires no modification to the model
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRLoss(nn.Module):
    """Supervised and self-supervised contrastive learning: https://arxiv.org/pdf/2004.11362.pdf. """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR self-supervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:  # feature dimensions: [images x views x channels x width x height]
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # normalise using l2 norm (puts features on a hypersphere)
        features = F.normalize(features, p=2, dim=2)  # make output vector unit length (divide by l2 norm)

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

if __name__ == "__main__":

    """
    The code below is a very quick and dirty demonstration of equivalence in the implementation of contrastive loss
    from SupContrast (https://github.com/HobbitLong/SupContrast) and this adaptation.
    The advantage of this adaptation is that the normalization of features is performed 
    within the loss function, allowing models to be trained without any augmentation to
    the original architecture code. A disadvantage is that outputs must be normalized separately if 
    they are used for any purpose other than measuring contrastive loss.
    """

    from types import SimpleNamespace
    import sys
    import os.path as op
    import torch
    import time
    import torch.optim as optim

    sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
    from utils import get_model

    sys.path.append(op.expanduser('~/david/repos'))
    from SupContrast.losses import SupConLoss

    inputs = torch.rand(32, 3, 224, 224)

    # my approach uses the base model, initialise and get parameters
    model_base = get_model(SimpleNamespace(model_name='cornet_s'))
    params = model_base.state_dict()

    input(params) # screen grab some of the weights

    # get contrastive loss (new adaptation) and perform an optimization step
    model_base.train()
    optimizer = optim.SGD(params=model_base.parameters(), lr=0.1, momentum=0.9)
    outputs = model_base(inputs)
    features = torch.stack(torch.split(outputs, [16, 16], dim=0), dim=1)
    loss_dave = ContrastiveLoss()(features)
    loss_dave.backward()
    optimizer.step()
    params_dave = model_base.state_dict()
    input(params_dave)  # screen grab some of the new model weights, should be different to original weights

    # initialize a model with normalization built in, use original loss method, optimization step
    model_cont = get_model(SimpleNamespace(model_name='cornet_s_cont'))
    model_cont.load_state_dict(params)
    model_cont.train()
    optimizer = optim.SGD(params=model_cont.parameters(), lr=0.1, momentum=0.9)
    outputs = model_cont(inputs)
    features = torch.stack(torch.split(outputs, [16, 16], dim=0), dim=1)
    loss_orig = SupConLoss()(features)
    loss_orig.backward()
    params_orig = model_cont.state_dict()
    print(params_orig) # these weights should be identical to params_dave
