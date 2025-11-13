"""
configure a model for training
"""
import os
import os.path as op
import glob
from types import SimpleNamespace
import pickle as pkl

# model
M = SimpleNamespace(

    architecture = 'cornet_s_custom',  # used to load architecture and name of top-level results directory
    #architecture_args = dict(),  # key word args when initializing the model

    # cornet_s_custom params
    architecture_args = dict(M=SimpleNamespace(
        R = (1,2,4,2),          # recurrence, default = (1,2,4,2),
        K = (3,3,3,3),          # kernel size, default = (3,3,3,3),
        F = (128,128,256,512),   # feature maps, default = (64,128,256,512)
        S = 4,                  # feature maps scaling, default = 4
        out_channels = 1,       # number of heads, default = 1
        head_depth = 2,         # multi-layer head, default = 1
        head_width = 3,         # gridsize of adaptive avgpool layer,
    )),

    model_dir = op.expanduser(f'~/david/models/cornet_s_plus/occ-nat1-untex-weak_task_cont'), #None, # manual override for results directory
    finetune = False,  # starting a fine-tuning regime from pretrained weights?
    #finetune_args = dict(),
        #starting_params=None, #'/home/tonglab/david/masterScripts/DNN/zoo/pretrained_weights/cornet_s-1d3f7974.pth',  # starting params for fine-tuning
        #finetune_dir='finetune_unocc',  # third-level directory for finetuned model results
        #freeze_modules=['V1', 'V2', 'V4','IT'],  # freeze the weights of these modules during finetuning
)

# dataset
D = SimpleNamespace(
    dataset = 'ILSVRC2012',
    num_views = 2,  # number of views to generate per example, default = 1
    transform_type = 'contrastive',  # 'contrastive' or 'default'
    cutmix = None,  # dict(
        #prob=1.0,
        #alpha=1,
        #beta=1,
        #frgrnd=True),
)

D.Occlusion = SimpleNamespace(
    form = 'naturalUntexturedCropped1',       # occluder type or list thereof
    probability = 5/6,                                               # probability that each image will be occluded, range(0,1)
    visibility = [.5, .6, .7, .8, .9],                              # image visibility or list thereof, range(0,1)
    colour = 'random',                                               # occluder colours (RGB uint8 / 'random' or list thereof. Ignored for 'textured' forms)
    views = [0],  # views to apply occlusion to, e.g. [0,2] will apply occluders to 1st and 3rd views
)

# optimization
O = SimpleNamespace(
    num_epochs = 43,  # number of epochs to train for
    batch_size = 256,  # minibatch size
    num_workers = 16,
    optimizer = 'SGD',  # any optimizer from torch.optim
    optimizer_args = dict(
        lr=0.1,  # set float or one of the following strings: 'LRfinder', 'batch_linear', 'batch_nonlinear'
        weight_decay=1e-4,
        momentum=.9,
    ),
    scheduler = 'StepLR',#'ReduceLROnPlateau', #  # any attribute of torch.optim.lr_scheduler
    scheduler_args = dict(
        gamma=0.1, step_size = 20,  # StepLR
        #patience = 10, factor = .1, mode='max',  # ReduceLROnPlateau
    ),
    criteria = dict(SimCLRLoss=dict(views=[0,1],weight=1)),  # loss function and, if multiple views per image, which views to apply loss to
    primary_metric = 'SimCLRLoss',  # metric to guide "best" params and lr scheduler
    save_interval = None,  # preserve params at every n epochs
    checkpoint = None,  # integer, resume training from this epoch (default: most recent, -1: from scratch)
    swa = None, # dict(start=8, anneal_epochs=4, swa_lr=.05)  # type: dict
    amp = True,  # automatic mixed precision. Can speed up optimization but can also cause over/underflow
)

CFG = SimpleNamespace(M=M,D=D,O=O)

if __name__ == "__main__":

    # complete configuration
    from complete_config import complete_config
    CFG = complete_config(CFG, resolve='resume')

    # output directory
    if not hasattr(CFG.M, 'model_dir'):
        CFG.M.model_dir = op.expanduser(f'~/david/projects/p022_occlusion/in_silico/models/{CFG.M.model_name}/{CFG.M.identifier}')

    import sys
    sys.path.append(op.expanduser('~/david/master_scripts/DNN/utils'))

    from train_model import train_model
    train_model(CFG)
