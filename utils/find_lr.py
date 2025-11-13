"""
This script determines basic training parameters prior to calling the training script.
Specifically, it calculates the maximum batch size and reasonable learning rate, and
resolves conflicts when a training regime is resumed with new settings
"""

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
from ignite.handlers import FastaiLRFinder
from ignite.engine import create_supervised_trainer
from .utils import get_loaders, get_optimizer, get_criterion

def find_lr(model, device, args):
    
    print('Calculating initial learning rate using LRfinder...')
    
    learning_rate = 0.1
    train_loader, _ = get_loaders(args, num_workers=4)
    lr_finder = FastaiLRFinder()
    optimizer = getattr(torch.optim, args.optimizer)(
        params=model.parameters(), **args.optimizer_args)
    criterion = get_criterion(args.criterion)
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device)
    model.to(device)
    to_save = {"model": model, "optimizer": optimizer}
    with lr_finder.attach(trainer, to_save=to_save) as finder:
        finder.run(train_loader)
        lr = lr_finder.suggestion()
    
    print(f'Initial learning rate set to {lr:.08}')
    
    return lr

