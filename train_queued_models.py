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
machines = ['finn','rey','padme','leia','solo','luke','yoda', 'chewie', 'Mando']
machine = socket.gethostname()

# job number
job_id = input(f'Which job number for this GPU set? E.g., 0 :')

def find_configs(machine, gpus, job_id):
    configs = sorted(glob.glob(f'training_queue/{machine}_gpu-{gpus}_job'
                               f'-{job_id}__*.json'))
    if not configs:
        configs = sorted(glob.glob(f'training_queue/*.json'))
        for config, machine in itertools.product(configs, machines):
            if machine in config:
                configs.remove(config)
    if not configs:
        print('no (more) configs found')
    return configs


# get model list
configs = find_configs(machine, gpus, job_id)
while configs:

    # find next model and claim it by renaming file
    config = configs[0]
    if not op.basename(config).startswith(machine):
        claimed_config = (f'training_queue/{machine}_gpu-{gpus}_job'
                          f'-{job_id}__{op.basename(config)}')
        shutil.move(config, claimed_config)
        orig_config = config
        config = claimed_config
    else:
        orig_config = op.basename(config).split('__')[-1]

    # load config, create model directory
    with open(config, 'r') as f:
        args = Namespace(**json.load(f))
    if args.model_dir is None:
        args.model_dir = op.expanduser(
            f'models/{args.architecture}')
    if args.finetune and not args.model_dir.endswith(
            args.finetune_args['finetune_dir']):
        args.model_dir += f'/{args.finetune_args["finetune_dir"]}'
    os.makedirs(args.model_dir, exist_ok=True)

    utils_dir = f'{args.model_dir}/utils'
    if not op.isdir(utils_dir):
        utils_dir = 'utils'
    sys.path.append(utils_dir)

    # calculate / set missing values in config
    from complete_args import complete_args
    args = complete_args(args, update=False)

    # train model
    from optimize_model import optimize_model
    optimize_model(args, verbose=True)

    # clean up after training
    if op.isfile(f'{args.model_dir}/done'):
        shutil.move(config, f'training_queue/done/{orig_config}')
    sys.path.remove(utils_dir)

    # refresh model configs
    configs = find_configs(machine, gpus)


