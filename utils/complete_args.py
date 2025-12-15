import os
import os.path as op
import glob
import sys
import datetime
import numpy as np
import pickle as pkl
import shutil
from types import SimpleNamespace
import pandas as pd
import json


def complete_args(args, update=False):

    """
    This function fills in an optimization configuration with default settings,
    including the calculation of basic optimization parameters prior to calling the
    optimization script. It also applies updates to the optimization if new 
    parameters are requested mid-way through an optimization regime.

    update = False to continue optimizing with no changes to the config
    update = True to merge new parameters into config
    """

    # look for previous configs
    config_file = f"{args.model_dir}/args.json"
    if not op.isfile(config_file):
        config_file = f"{args.model_dir}/config.json"
    orig_config_exists = op.isfile(config_file)
    if orig_config_exists:
        with open(config_file, 'r') as f:
            args_orig = SimpleNamespace(**json.load(f))
        if update == False:
            print(f'Resuming optimization of model at {args.model_dir} with '
                  f'original args')
            args = args_orig
        else:
            print(f'Resuming optimization of model at {args.model_dir} with '
                  f'potential new args')
            for key, value in vars(args).items():
                if key in vars(args_orig) and value != vars(args_orig)[key]:
                    print(f'Replacing parameter "{key}" with "{vars(args_orig)[key]}')
            for key, value in vars(args_orig).items():
                if key not in vars(args):
                    setattr(args, key, value)

    # unless starting new optimization regime or resuming with no changes, merge
    elif args.finetune:
        if args.finetune_args['starting_params'] is None:
            try:
                args.starting_params = sorted(glob.glob(
                    f'{op.dirname(args.model_dir)}/params/*.pt'))[-1]
                print(f'Finetuning regime initiated using params at '
                      f'{args.starting_params}')
            except:
                ValueError, ('Finetuning regime requested but starting '
                             'weights were not found')
    else:
        print(f'Starting optimization of new model at {args.model_dir}')

    # add any missing default parameters
    default_args = dict(
        architecture_args=dict(),
        dataset='ILSVRC2012',
        num_views=1,
        transform_type='default',
        image_size=224,
        cutmix=False,
        mixup=False,
        randomerase=False,
        num_epochs=100,
        criterion='CrossEntropyLoss',
        optimizer='SGD',
        optimizer_args=dict(lr=.05, momentum=0.9, weight_decay=1e-4),
        scheduler='StepLR',
        scheduler_args=dict(step_size=30, gamma=0.1),
        save_interval=None,
        swa=None,
        amp=True,
        batch_dependence=False,
        view_dependence=False,
        video=False,
    )
    for key, value in default_args.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # calculate last epoch if not set or resume optimization with no changes
    if 'checkpoint' not in vars(args) or args.checkpoint is None or not update:
        params_paths = sorted(glob.glob(f"{args.model_dir}/params/???.pt"))
        if params_paths and int(op.basename(params_paths[-1])[:-3]):
            print(f'Most recent params found at {params_paths[-1]}')
            setattr(args, 'checkpoint', int(op.basename(params_paths[-1])[:-3]))
        else:
            setattr(args, 'checkpoint', -1)

    # determine whether model has finished optimizing
    performance_path = f'{args.model_dir}/performance.csv'
    if op.isfile(performance_path):
        performance = pd.read_csv(open(performance_path, 'r+'))
        if performance.epoch.min() == 1:
            performance.epoch = performance.epoch - 1
            performance.to_csv(performance_path, index=False)
        if args.checkpoint > -1:
            performance = performance[performance.epoch <= args.checkpoint]
            performance.to_csv(performance_path, index=False)
        optimize = performance.epoch.max() < args.num_epochs - 1
    else:
        optimize = True

    if optimize:

        import torch
        utils_dir = f'{args.model_dir}/utils'
        if not op.isdir(utils_dir):  # if starting from scratch, remove
            utils_dir = 'utils'
        sys.path.append(utils_dir)
        from get_model import get_model
        from change_output_size import change_output_size

        # loading model necessary for hyperparameter calculation
        def _load_model(args):

            # configure hardware
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # load model
            model = get_model(args.architecture, args.architecture_args)

            # adapt output size of model based on number of classes in dataset
            num_classes = len(
                glob.glob(op.expanduser(f'~/Datasets/{args.dataset}/train/*')))
            if num_classes != 1000 and 'CrossEntropyLoss' in args.criterion:
                model = change_output_size(model, args, num_classes)
            return model, device

        # calculate optimal batch size
        if 'batch_size' not in vars(args):
            from calculate_batch_size import calculate_batch_size
            model, device = _load_model(args)
            setattr(args, 'batch_size', calculate_batch_size(
                model, args, device))
            print(f'optimal batch size calculated at {args.batch_size}')

        # calculate optimal learning rate
        if optimize and type(args.optimizer_args['lr']) == str:
            model, device = _load_model(args)
            if args.optimizer_args['lr'] == 'batch_nonlinear':
                args.optimizer_args['lr'] = 2**-7 * np.sqrt(args.batch_size/2**5)
            elif args.optimizer_args['lr'] == 'batch_linear':
                args.optimizer_args['lr'] = args.batch_size / 2**5
            elif args.optimizer_args['lr'] == 'LRfinder':
                from find_lr import find_lr
                args.optimizer_args['lr'] = find_lr(model, device, args)

    return args


