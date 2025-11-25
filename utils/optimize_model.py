import sys
import pickle as pkl
import pandas as pd
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
import math
import pprint
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.autograd.variable import Variable
import torch.nn.functional as F
import copy
import json
import multiprocessing as mp
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from utils.accuracy import accuracy
from utils.select_outputs_targets import select_outputs_targets
from utils.AverageMeter import AverageMeter
from utils.cutmix import cutmix
from utils.plot_performance import plot_performance
from utils.save_images import save_image_batch
from utils.get_loaders import get_loaders
from utils.get_criterion import get_criterion
from utils.get_model import get_model
from utils.load_params import load_params

torch.random.manual_seed(42)


def optimize_model(args, verbose=False):

    # configure hardware
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = torch.cuda.device_count() * 8

    # loss functions and performance metrics
    # send criteria to device before constructing optimizer
    criterion, metrics = get_criterion(args.criterion)
    if args.cutmix or args.mixup:
        metrics.remove('acc1')
        metrics.remove('acc5')
    criterion.to(device)
    performance_path = f'{args.model_dir}/performance.csv'
    if op.isfile(performance_path):
        performance = pd.read_csv(open(performance_path, 'r+'))
    else:
        performance = pd.DataFrame()

    # image processing
    loader_train, loader_val = get_loaders(num_workers, args)

    # model
    model = get_model(args.architecture, args.architecture_args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # optimizer
    optimizer = getattr(torch.optim, args.optimizer)(
        params=model.parameters(), **args.optimizer_args)
    scheduler = getattr(torch.optim.lr_scheduler, args.scheduler)(
        optimizer, **args.scheduler_args)
    if args.swa is not None:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, last_epoch=args.checkpoint, **args.swa)
    if hasattr(args, 'warmup_scheduler'):
        warmup_scheduler = getattr(torch.optim.lr_scheduler, args.warmup_scheduler)(
            optimizer, **args.warmup_scheduler_args)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], 
            milestones=[args.warmup_scheduler_args['total_iters']])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)  # mixed precision

    # find checkpoint params
    if args.finetune and args.checkpoint == -1:
        params_path = args.finetune_args['starting_params']
    elif args.checkpoint > -1:
        params_path = f'{args.model_dir}/params/{args.checkpoint:03}.pt'
    else:
        params_path = None

    # load states
    if params_path:
        print(f'Loading model and optimizer states from {params_path}')
        params = torch.load(params_path, map_location=device)
        if args.finetune and 'freeze_modules' in args.finetune_args:
            modules = args.finetune_args['freeze_modules']
        else:
            modules = 'all'
        model = load_params(params, model, 'model', modules=modules)
        if not (args.finetune and args.checkpoint == -1):
            optimizer = load_params(params, optimizer, 'optimizer')
            scheduler = load_params(params, scheduler, 'scheduler')
        if args.finetune and 'freeze_modules' in args.finetune_args:  # freeze weights (turn off gradients)
            for module in args.finetune_args['freeze_modules']:
                print(f'Freezing layer: {module}')
                getattr(model, module).requires_grad_(False)
            if verbose:
                for p, param in enumerate(model.parameters()):
                    print(f'param: {p} | shape: {param.shape} | '
                          f'requires_grad: {param.requires_grad}')

    # save initial states
    if args.checkpoint == -1:
        print('Saving initial model and optimizer states')
        optimizer.param_groups[0]['initial_lr'] = args.optimizer_args['lr']
        args.checkpoint = 0
        params_path = f'{args.model_dir}/params/000.pt'
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
        if args.swa:
            params['swa_model'] = swa_model.state_dict()
        os.makedirs(op.dirname(params_path), exist_ok=True)
        torch.save(params, params_path)
        with open(op.join(args.model_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    # print some useful information
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f'Model directory: {args.model_dir}')
    print(f'Parameters with gradients on: {num_params}')
    print(f'Workers: {num_workers}')
    print(f'Metrics: {metrics}')

    def _one_epoch(model, loader, optimizer, train_eval, performance):

        model.train() if train_eval == 'train' else model.eval()
        performance_tracker = {metric: AverageMeter() for metric in metrics}
        step = (epoch + int(train_eval == 'eval')) * len(loader_train)

        # loop through batches
        with tqdm(loader, unit=f"batch({args.batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):

                batch_time = datetime.now().strftime("%y/%m/%d %H:%M:%S")
                tepoch.set_description(
                    f'{batch_time} | {train_eval.ljust(5)} | epoch'
                    f' {epoch + 1}/{args.num_epochs}')

                # after successful batch has been run, 
                # copy over all utilities for reproducibility and save sample inputs
                if epoch == 0 and batch == 1 and train_eval == 'train':
                    save_image_batch(inputs, num_views=args.num_views,
                        out_dir=f'{args.model_dir}/sample_{train_eval}_inputs')
                    shutil.copytree('utils', f'{args.model_dir}/utils')

                # flatten inputs for SimCLR
                if args.criterion == 'SimCLRLoss':
                    assert len(inputs.shape) == 5, ('SimCLRLoss requires '
                        'inputs with shape [batch, num_views, C, H, W]')
                    inputs = inputs.flatten(start_dim=0, end_dim=1)

                # put inputs on device
                non_blocking = 'SimCLRLoss' in metrics
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)

                # automatic mixed precision
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=args.amp):

                    outputs = model(inputs)  # pass through model
                    outputs, targets = select_outputs_targets(
                        outputs, targets, args)  # select outputs and targets
                    loss = criterion(outputs, targets)   # get loss

                    # classification accuracy
                    if 'acc1' in metrics:
                        acc1, acc5 = [x.detach().cpu().item() for x in
                                      accuracy(outputs, targets, (1, 5))]
                        performance_tracker['acc1'].update(acc1)
                        performance_tracker['acc5'].update(acc5)

                    # add final loss value to performance tracker
                    loss_value = loss.clone().detach().cpu().numpy()
                    performance_tracker[args.criterion].update(loss_value)

                # update model
                if train_eval == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    lr = optimizer.param_groups[0]['lr']
                else:
                    lr = 0

                # display performance metrics for this batch
                tepoch.set_postfix_str(' '.join(
                    [f'{metric}={value.val:.4f}({value.avg_epoch:.4f}) ' for
                     metric, value in performance_tracker.items()]) +
                                       f' lr={lr:.5f}')

                # every n training steps and at end of eval epoch,
                # update performance, reset tracker batches, plot
                if (train_eval == 'train' and step % 1024 == 0) or (
                        train_eval == 'eval' and batch == len(loader) - 1):
                    performance_batches = pd.DataFrame(dict(
                        time=[batch_time], step=[step], epoch=[epoch],
                        train_eval=[train_eval], lr=[lr]))
                    for key, value in performance_tracker.items():
                        performance_batches[key] = [value.avg_batches]
                        value.reset_batches()
                    performance = pd.concat([performance, performance_batches])
                    plot_performance(performance, metrics, args.model_dir)

        return model, optimizer, scaler, performance

    # train / eval loop
    for epoch in range(args.checkpoint, args.num_epochs):

        model, optimizer, scaler, performance = _one_epoch(
            model, loader_train, optimizer, 'train', performance)  # train
        with torch.no_grad():
            model, optimizer, scaler, performance = _one_epoch(
                model, loader_val, optimizer, 'eval', performance)  # eval

        # update schedulers
        if args.swa and epoch >= args.swa_args['start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            if epoch == args.num_epochs:
                update_bn(loader_train, swa_model)
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(performance[args.primary_metric].values[-1])
        else:
            scheduler.step()

        # save new states
        params_path = f"{args.model_dir}/params/{epoch + 1:03}.pt"
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()}
        if args.swa:
            params['swa_model'] = swa_model.state_dict()
        torch.save(params, params_path)

        # delete previous states
        if args.save_interval is None or epoch % args.save_interval:
            last_save_path = f"{args.model_dir}/params/{epoch:03}.pt"
            if op.exists(last_save_path):
                os.remove(last_save_path)

        # save if best eval performance
        perf = performance[args.primary_metric][
            performance['train_eval'] == 'eval'].to_numpy()
        func = np.greater if 'acc' in args.primary_metric else np.less
        if func(perf[-1], perf[:-1]).all():
            for prev_best in glob.glob(f'{args.model_dir}/params/best*.pt'):
                os.remove(prev_best)
            torch.save(params,f'{args.model_dir}/params/best_{epoch + 1:03}.pt')

        # save performance
        performance.to_csv(performance_path, index=False)

    print('Optimization complete.')
    with open(f'{args.model_dir}/done', 'w') as _: 
        pass 
    
    
