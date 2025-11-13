"""
Create model optimization configuration files for training.
"""

import os
import os.path as op
import glob
from argparse import Namespace
from itertools import product as itp
from copy import deepcopy
import json

queue_dir = op.expanduser('~/david/master_scripts/DNN/training_queue')
os.chdir(queue_dir)
models_dir = op.expanduser('~/david/models')

architectures = ['resnet101', 'cornet-s', 'efficientnet_b1']
occluders = {
    'unoccluded': None,
    'artificial-shape': [
        'ellipses_filled', 'ellipses_empty',
        'triangles_filled', 'triangles_empty',
        'rectangles_filled', 'rectangles_empty',
        'curved_lines', 'straight_lines'],
    'natural-shape': 'naturalUntexturedCropped1',
    'natural-shape-and-texture': 'naturalTexturedCropped1'
}
occlusion_levels = {'weak': [.5, .6, .7, .8, .9],
                    'strong': [.1, .2, .3, .4, .5, .6, .7, .8, .9]}

with open('templates/template.json') as f:
    args = json.load(f)

p = 1  # priority_counter
for (occluder, occluder_form), arch in itp(occluders.items(), architectures):

    tasks = ['classification']
    if arch == 'resnet101':
        tasks.append('contrastive')
    for task in tasks:

        occ_levels = ['weak']
        if (arch == 'resnet101' and task == 'classification' and
                occluder != 'unoccluded'):
            occ_levels.append('strong')
        for occ_level in occ_levels:

            label_items = [task, occluder, occ_level]
            if occluder == 'unoccluded':
                label_items.remove(occ_level)
            label = '_'.join(label_items)

            args2 = Namespace(**args)
            args2.architecture = arch
            args2.model_dir = f'{models_dir}/{arch}/{label}'
            if occluder_form is None:
                delattr(args2, 'Occlusion')
            else:
                args2.Occlusion['form'] = occluder_form
                args2.Occlusion['visibility'] = occlusion_levels[occ_level]

            if arch != 'cornet_s':
                args2.batch_size = 512
                args2.num_epochs = 100
                args2.scheduler_args['step_size'] = 30

            if task == 'contrastive':
                args2.num_views = 2
                args2.criterion = 'SimCLRLoss'
                args2.primary_metric = 'SimCLRLoss'

            filename = f'{arch}_{label}.json'
            if len(glob.glob(op.join(queue_dir, f'*{filename}'))) == 0:

                path = op.join(queue_dir, f'unclaimed_{p:02}_{filename}')
                args2_dict = vars(args2)
                print(path)
                with open(path, 'w+') as f:
                    json.dump(args2_dict, f, sort_keys=True, indent=4)
                p += 1
