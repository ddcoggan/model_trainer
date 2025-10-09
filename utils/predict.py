import pandas as pd
import os.path as op
from torch import Tensor

def predict(output, dataset, label_type='label', afc=None):

    """returns class prediction (based on directory label) for all
    classes or within alternate forced choice"""
    if dataset == 'ILSVRC2012':
        label_data = pd.read_csv(open(op.expanduser('~/david/datasets/images/ILSVRC2012/labels.csv'), 'r+'))
    if afc:
        output = output[:,afc]
    if type(output) == Tensor:
        class_idx = output.argmax(dim=1)
    else:
        class_idx = output.argmax(axis=1)
    if afc:
        class_idx = [afc[idx] for idx in class_idx]

    responses = [label_data[label_type][int(idx)] for idx in class_idx]
    return responses
