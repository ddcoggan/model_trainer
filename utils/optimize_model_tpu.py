import sys
import pickle as pkl
import pandas as pd
import numpy as np
import glob
import os
import os.path as op
from PIL import Image
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import Namespace
import math
import pprint

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch.nn.functional as F

import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental import pjrt
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.utils.utils as xu
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.test.test_utils as test_utils
torch.set_default_tensor_type('torch.FloatTensor')

sys.path.append(op.expanduser('~/david/master_scripts/DNN'))
from utils import accuracy

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

def train_model_tpu(CFG):

    # unpack config
    M, D, T = CFG.M, CFG.D, CFG.T

    # initialise model directory and save config
    os.makedirs(f'{M.model_dir}/params', exist_ok=True)
    pkl.dump(CFG, open(f'{CFG.M.model_dir}/config.pkl', 'wb'))
    from utils import config_to_text
    config_to_text(CFG)

    # hardware
    device = xm.xla_device()

    # image processing
    from utils import get_transforms
    train_path = op.expanduser(f'~/Datasets/{D.dataset}/train')
    val_path = op.expanduser(f'~/Datasets/{D.dataset}/val')
    transform_train, transform_val = get_transforms(D, T)
    train_data = ImageFolder(train_path, transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=T.batch_size, shuffle=True, num_workers=T.num_workers)
    train_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_data = ImageFolder(val_path, transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=T.batch_size, shuffle=True, num_workers=T.num_workers)
    val_device_loader = pl.MpDeviceLoader(val_loader, device)

    # loss functions
    if 'classification' in T.learning:
        loss_class = nn.CrossEntropyLoss().to(device)  # send to device before constructing optimizer
    if 'contrastive' in T.learning:
        from utils import ContrastiveLoss
        loss_contr = ContrastiveLoss().to(device)  # can do both supervised and unsupervised contrastive loss

    # performance metrics
    metrics = []
    if 'classification' in T.learning:
        metrics += ['acc1', 'acc5', 'loss_class']
        loss_metric = 'loss_class'
    if 'contrastive' in T.learning:
        metrics += ['loss_contr']
        loss_metric = 'loss_contr'

    # epoch-wise stats file
    epoch_stats_path = f'{M.model_dir}/epoch_stats.csv'
    if op.isfile(epoch_stats_path):
        epoch_stats = pd.read_csv(open(epoch_stats_path, 'r+'), index_col=0)
    else:
        epoch_stats = {'epoch': [], 'train_eval': []}
        for metric in metrics:
            epoch_stats[f'{metric}'] = []
            epoch_stats[f'{metric}'] = []
        epoch_stats['time'] = []
        epoch_stats = pd.DataFrame(epoch_stats)


    # model

    # put model on device
    if hasattr(M, 'model'):
        model = M.model
    else:
        from utils import get_model
        model = get_model(M)
    model.to(device)
    print(model)

    if not hasattr(T, 'checkpoint') or T.checkpoint is None:

        # set starting point
        params_paths = sorted(glob.glob(f'{M.model_dir}/params/*.pt'))
        if params_paths:
            T.checkpoint = int(params_paths[-1][-6:-3])
        else:
            print('New model state created')
            print('New optimizer state created')
            T.checkpoint = None  # make sure attribute exists
            next_epoch = 1
            train_evals = ['train', 'eval']

    if T.checkpoint is not None:

        print('Loading previous model state')

        # if training interrupted during eval of last epoch, finish this before continuing
        if not len(epoch_stats[(epoch_stats['epoch'] == T.checkpoint) & (epoch_stats['train_eval'] == 'eval')]):
            next_epoch = T.checkpoint
            train_evals = ['eval']
        else:
            next_epoch = T.checkpoint + 1
            train_evals = ['train', 'eval']

        # load model parameters
        params_path = f'{M.model_dir}/params/{T.checkpoint:03}.pt'
        params = torch.load(params_path)
        from utils import load_params
        model = load_params(params, model=model)

        # freeze weights if transfer learning
        if T.freeze_weights:
            for p, param in enumerate(model.parameters()):
                if p > T.freeze_weights:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # view param items
            # for p, param in enumerate(model.parameters()):
            #    print(f'item {p}, shape = {param.shape}, requires_grad = {param.requires_grad}')

    # optimizer
    if T.optimizer_name == 'SGD':
        optimizer = optim.SGD(params=model.parameters(), lr=T.learning_rate, momentum=T.momentum)
        optimizer.param_groups[0]['initial_lr'] = T.learning_rate
    elif T.optimizer_name == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=T.learning_rate)

    # load optimizer state
    if T.checkpoint is not None:
        from utils import load_params
        optimizer = load_params(params, optimizer=optimizer)
    else:
        T.checkpoint = 0  # scheduler errors if this is None

    # scheduler to adapt optimizer parameters throughout training
    if T.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=T.step_size, last_epoch=T.checkpoint)
    if T.scheduler == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # save initial model and optimizer parameters
    if T.checkpoint is None:
        print('Saving model and optimizer parameters')
        epoch_save_path = f'{M.model_dir}/params/000.pt'
        params = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
        torch.save(params, epoch_save_path)

    # train / eval loop
    for epoch in list(range(next_epoch, T.num_epochs + 1)) + [0]: # do epoch zero last

        # load epoch 0 params at the end
        if epoch == 0:
            from utils import load_params
            params_path = f'{M.model_dir}/params/000.pt'
            model = load_params(params_path, model=model)

        for train_eval in train_evals:

            # train/eval specific settings
            if train_eval == 'train':
                loader = train_device_loader
                log_string = 'Training'.ljust(10)
                model.train() if epoch != 0 else model.eval()
            else:
                loader = val_device_loader
                log_string = 'Evaluating'
                model.eval()

            # initialize cumulative performance stats for this epoch
            epoch_tracker = {metric: 0 for metric in metrics}

            # loop through batches
            with tqdm(loader, unit=f"batch({T.batch_size})") as tepoch:

                for batch, (inputs, targets) in enumerate(tepoch):

                    tepoch.set_description(
                        f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | {log_string} | Epoch {epoch}/{T.num_epochs}')

                    if train_eval == 'train' and epoch > 0:
                        optimizer.zero_grad(set_to_none=True)  # set_to_none saves even more memory1

                    # initialize performance stats for this batch (used for print out)
                    batch_stats = {}

                    # for contrastive learning, alter images, perform remaining image transform
                    if 'contrastive' in T.learning:
                        from utils import AlterImages, get_remaining_transform
                        inputs = AlterImages(D, T)(inputs)
                        remaining_transform = get_remaining_transform(train_eval)
                        inputs = remaining_transform(inputs)

                    # save some input images
                    if epoch == 1 and batch == 0:
                        from utils import save_image_custom
                        sample_input_dir = f'{M.model_dir}/sample_training_inputs'
                        os.makedirs(sample_input_dir, exist_ok=True)
                        save_image_custom(inputs, T, sample_input_dir, max_images=64)

                    # put inputs on device
                    non_blocking = 'contrastive' in T.learning
                    inputs = inputs.to(device, non_blocking=non_blocking)
                    targets = targets.to(device, non_blocking=non_blocking)

                    # pass through model
                    outputs = model(inputs)

                    # separate outputs by classification/contrastive
                    if M.model_name in ['cornet_s_custom', 'cornet_st'] and M.out_channels == 2:
                        outputs_class = outputs[:, :, 0]
                        outputs_contr = outputs[:, :, 1]
                        if 'contrastive' in T.learning:
                            targets_class = torch.cat([targets, targets], dim=0)
                        else:
                            targets_class = targets
                    elif T.learning == 'supervised_classification':
                        outputs_class = outputs
                        targets_class = targets
                    else:
                        outputs_contr = outputs

                    # classification accuracy and loss
                    if 'classification' in T.learning:
                        batch_stats['acc1'], batch_stats['acc5'] = [x.detach().cpu() for x in
                                                                    accuracy(outputs_class, targets_class, (1, 5))]
                        epoch_tracker[f'acc1'] = ((epoch_tracker[f'acc1'] * batch) +
                                                                  batch_stats['acc1']) / (batch + 1)
                        epoch_tracker[f'acc5'] = ((epoch_tracker[f'acc5'] * batch) +
                                                                  batch_stats['acc5']) / (batch + 1)
                        loss_cl = loss_class(outputs_class, targets_class)  # leave this on gpu for back prop
                        batch_stats['loss_class'] = loss_cl.detach().cpu().item()  # store copy of loss value on cpu
                        epoch_tracker[f'loss_class'] = ((epoch_tracker[f'loss_class'] * batch) + batch_stats[
                                                                            'loss_class']) / (batch + 1)

                    # contrastive accuracy and loss
                    if 'contrastive' in T.learning:
                        features = torch.stack(torch.split(outputs_contr, [targets.shape[0], targets.shape[0]], dim=0),
                                               dim=1)  # unstack and combine along new (contrastive) dimension
                        if 'unsupervised_contrastive' in T.learning:
                            loss_co = loss_contr(features)  # leave this on gpu for back prop
                            batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                        elif 'supervised_contrastive' in T.learning:
                            loss_co = loss_contr(features, targets)  # leave this on gpu for back prop
                            batch_stats['loss_contr'] = loss_co.detach().cpu().item()  # store copy of loss value on cpu
                        epoch_tracker[f'loss_contr'] = ((epoch_tracker[f'loss_contr'] * batch) + batch_stats[
                                                                            'loss_contr']) / (batch + 1)

                    # display performance metrics for this batch
                    postfix_string = ''
                    for metric in metrics:
                        postfix_string += f"{metric}={batch_stats[metric]:.4f}({epoch_tracker[metric]:.4f}) | "
                    postfix_string += f"lr={optimizer.param_groups[0]['lr'] * (train_eval == 'train' and epoch != 0):.5f}"
                    tepoch.set_postfix_str(postfix_string)


                    # compute gradients and optimize parameters
                    if train_eval == 'train' and epoch > 0:
                        if M.model_name in ['cornet_s_custom', 'cornet_st'] and M.out_channels == 2:
                            if hasattr(T, 'loss_ratio'):
                                loss_unit = 1 / sum(T.loss_ratio)
                                loss_cl_weight, loss_co_weight = [T.loss_ratio[l] * loss_unit for l in
                                                                  range(len(T.loss_ratio))]
                            else:
                                loss_cl_weight, loss_co_weight = (.5, .5)
                            loss = loss_cl * loss_cl_weight + loss_co * loss_co_weight
                        elif 'classification' in T.learning:
                            loss = loss_cl
                        else:
                            loss = loss_co
                        loss.backward()
                        xm.optimizer_step(optimizer, barrier=True)

            # save epoch stats
            new_stats = {'time': [datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")], 'epoch': [epoch], 'train_eval': [train_eval]}
            new_stats = {**new_stats, **{key: np.array(item, dtype="float16") for key, item in epoch_tracker.items()}}
            epoch_stats = pd.concat([epoch_stats, pd.DataFrame(new_stats)]).reset_index(drop=True)
            epoch_stats.to_csv(epoch_stats_path)

            # plot performance
            if epoch != 1:
                fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics * 4), 4))
                train_vals = epoch_stats[epoch_stats['train_eval'] == 'train']
                eval_vals = epoch_stats[epoch_stats['train_eval'] == 'eval']
                for m, metric in enumerate(metrics):
                    ax = axes[m]
                    train_x, train_y = train_vals['epoch'].values, train_vals[metric].values
                    eval_x, eval_y = eval_vals['epoch'].values, eval_vals[metric].values
                    ax.plot(train_x, train_y, label='train')
                    ax.plot(eval_x, eval_y, label='eval')
                    ax.set_xlabel('epoch')
                    ax.set_ylabel(metric)
                    if metric == 'loss_contr':
                        ax.yscale('log')
                plt.legend()
                plt.grid(True)
                plt.savefig(f'{M.model_dir}/performance.png')
                plt.close()

            # update LR scheduler
            if epoch > 0 and train_eval == 'train':
                if T.scheduler == 'StepLR':
                    scheduler.step()
                elif T.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(epoch_stats[loss_metric].values[-2])

            # record model and optimizer state
            if train_eval == 'train':

                # save new state
                epoch_save_path = f"{M.model_dir}/params/{epoch:03}.pt"
                params = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
                torch.save(params, epoch_save_path)

                # delete old state
                if epoch > 1 and M.save_interval > 1 and (epoch - 1) % M.save_interval != 0:
                    last_save_path = f"{M.model_dir}/params/{epoch - 1:03}.pt"
                    os.remove(last_save_path)

        # ensure next epoch does both train and eval
        train_evals = ['train', 'eval']

    print('Training complete.')


if __name__ == '__main__':

    from types import SimpleNamespace

    # model
    M = SimpleNamespace(
        model_name='cornet_s',
        identifier='base_model',  # used to name model directory, required
        save_interval=4,  # preserve params at every n epochs
    )


    # dataset
    D = SimpleNamespace(
        dataset='ILSVRC2012',
        transform='alter',
    )

    # training
    T = SimpleNamespace(
        optimizer_name='SGD',  # SGD or ADAM
        batch_size=128,
        learning_rate=.01,
        momentum=.9,
        scheduler='StepLR',
        step_size=16,
        num_epochs=36,  # number of epochs to train for
    )


    CFG = SimpleNamespace(M=M, D=D, T=T)

    # output directory
    if not hasattr(CFG.M, 'model_dir'):
        CFG.M.model_dir = op.expanduser(
            f'models/{CFG.M.model_name}/{CFG.M.identifier}')

    train_model(CFG)
