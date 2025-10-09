"""
train models in queue
"""
import os
import os.path as op
import glob
import socket
import sys
import shutil
import itertools
from argparse import Namespace
import json

# set cuda GPU visibility
gpus = input(f'Which GPU(s)? E.g., 0 or 0,1 :')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # converts to nvidia-smi order
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

# configure torch settings AFTER setting CUDA_VISIBLE_DEVICES
import torch
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(False)
# force cuda initialization
# torch.nn.functional.conv2d(
#    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')),
#    torch.zeros(32, 32, 32, 32, device=torch.device('cuda')))

# machines
machines = ['finn','rey','padme','leia','solo','luke','yoda']
machine = socket.gethostname()

# model queue
queue_dir = op.expanduser('~/david/master_scripts/DNN/training_queue')
sys.path.append(queue_dir)
os.chdir(queue_dir)


def find_configs(machine, gpus):
    configs = sorted(glob.glob(f'{machine}-{gpus}_*.json'))
    if not configs:
        configs = sorted(glob.glob(f'*.json'))
        for config, machine in itertools.product(configs, machines):
            if machine in config:
                configs.remove(config)
    return configs


# get model list
configs = find_configs(machine, gpus)
while configs:

    # find next model and claim it by renaming file
    config = configs[0]
    if not config.startswith(machine):
        claimed_config = f'{machine}-{gpus}_{op.basename(config)}'
        shutil.move(config, claimed_config)
        orig_config = config
        config = claimed_config
    else:
        orig_config = '_'.join(config.split('_')[2:])

    # load config, create model directory
    with open(config, 'r') as f:
        args = Namespace(**json.load(f))
    if args.model_dir is None:
        args.model_dir = op.expanduser(
            f'~/david/models/{args.architecture}')
    if args.finetune and not args.model_dir.endswith(
            args.finetune_args['finetune_dir']):
        args.model_dir += f'/{args.finetune_args["finetune_dir"]}'
    os.makedirs(args.model_dir, exist_ok=True)
    """
    # copy over all utilities for reproducibility (now done in optimize_model.py)
    utils_dir = f'{args.model_dir}/utils'
    if op.isdir(utils_dir): # if starting from scratch, remove existing utils
        saved_epoch_paths = sorted(glob.glob(f'{args.model_dir}/params/???.pt'))
        if len(saved_epoch_paths):
            last_saved_epoch = int(op.basename(saved_epoch_paths[-1])[:3])
            if last_saved_epoch == 0:
                shutil.rmtree(utils_dir)
    if not op.exists(utils_dir):
        utils_dir_orig = op.expanduser('~/david/master_scripts/DNN/utils')
        shutil.copytree(utils_dir_orig, utils_dir)
    sys.path.append(utils_dir)
    """

    utils_dir = f'{args.model_dir}/utils'
    if not op.isdir(utils_dir):  # if starting from scratch, remove
        utils_dir = op.expanduser('~/david/master_scripts/DNN/utils')
    sys.path.append(utils_dir)

    # calculate / set missing values in config
    from complete_args import complete_args
    args = complete_args(args, update=False)

    # train model
    from optimize_model import optimize_model
    optimize_model(args, verbose=True)

    # clean up after training
    shutil.move(config, f'done/{orig_config}')

    # refresh model configs
    configs = find_configs(machine, gpus)


