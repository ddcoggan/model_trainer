# Created by David Coggan on 2024 12 20
import os.path as op
import glob
import pandas as pd
import pickle
import json
import numpy as np
from types import SimpleNamespace
from itertools import product as itp
import sys
sys.path.append(op.expanduser('~/david/master_scripts/DNN'))

MODEL_DIR = op.expanduser('~/david/models')
IGNORE_PARAMS = ['device', 'num_workers', 'workers', 'model', 'save_interval',
                 'checkpoint', 'model_dir', 'identifier', 'label',  'nGPUs',
                 'GPUids', 'overwrite_optimizer', 'visLabel', 'force_lr',
                 'reinitialize_layers', 'Blur', 'Noise', 'invert']
IGNORE_MODELS = ['pix2pix', 'pretrained']
PARAMS = dict(
    amp=dict(aliases=['AMP'], dtype=bool),
    architecture=dict(aliases=['model_name'], dtype=str),
    architecture_args=dict(aliases=[], dtype=object, sub_params=dict(
        cycles=dict(aliases=[], dtype=int),
        out_channels=dict(aliases=[], dtype=int),
        F=dict(aliases=[], dtype=object),
        head_depth=dict(aliases=[], dtype=int),
        head_width=dict(aliases=[], dtype=int),
        K=dict(aliases=[], dtype=object),
        log_lattice=dict(aliases=[], dtype=bool),
        num_classes=dict(aliases=[], dtype=int),
        num_layers=dict(aliases=[], dtype=int),
        pretrained=dict(aliases=[], dtype=bool),
        R=dict(aliases=[], dtype=object),
        recurrent_steps=dict(aliases=[], dtype=int),
        return_cycles=dict(aliases=[], dtype=object),
        S=dict(aliases=['scale'], dtype=int),
    )),
    batch_size=dict(aliases=[], dtype=int),
    batch_dependence=dict(aliases=[], dtype=bool),
    blur_sigmas=dict(aliases=[], dtype=object),
    checkpoint=dict(aliases=[], dtype=int),
    contrast=dict(aliases=[], dtype=str),
    criterion=dict(aliases=['criteria'], dtype=str),
    cutmix=dict(aliases=[], dtype=bool),
    cutmix_args=dict(aliases=[], dtype=object, sub_params=dict(
        alpha=dict(aliases=['cutmix_alpha'], dtype=float),
        prob=dict(aliases=['cutmix_prob'], dtype=float),
        beta=dict(aliases=['cutmix_beta'], dtype=float),
        frgrnd=dict(aliases=['cutmix_frgrnd'], dtype=bool),
    )),
    dataset=dict(aliases=[], dtype=str),
    dataset_path=dict(aliases=[], dtype=str),
    finetune=dict(aliases=['transfer'], dtype=bool),
    finetune_dir=dict(aliases=['transfer_dir'], dtype=str),
    freeze_modules=dict(aliases=['freeze_layers', 'freeze_weights'],
                        dtype=object),
    image_size=dict(aliases=[], dtype=int),
    last_epoch=dict(aliases=[], dtype=int),
    loss_cycle=dict(aliases=[], dtype=object),
    model_dir=dict(aliases=['outdir'], dtype=str),
    num_epochs=dict(aliases=[], dtype=int),
    num_views=dict(aliases=['views'], dtype=int),
    Occlusion=dict(aliases=[], dtype=object, sub_params=dict(
        color=dict(aliases=['colour', 'colours'], dtype=object),
        form=dict(aliases=['type', 'occluder'], dtype=object),
        occluder_dir=dict(aliases=[], dtype=str),
        probability=dict(aliases=['prop_occluded'], dtype=float),
        views=dict(aliases=['views_occluded', 'views_altered'], dtype=int),
        visibility=dict(aliases=['coverage'], dtype=object),
    )),
    optimizer=dict(aliases=['optimizer_name'], dtype=str),
    optimizer_args=dict(aliases=[], dtype=object, sub_params=dict(
        lr=dict(aliases=['learning_rate'], dtype=float),
        weight_decay=dict(aliases=[], dtype=float),
        momentum=dict(aliases=[], dtype=float),
    )),
    primary_metric=dict(aliases=['metric'], dtype=str),
    save_interval=dict(aliases=[], dtype=int),
    scheduler=dict(aliases=[], dtype=str),
    scheduler_args=dict(aliases=[], dtype=object, sub_params=dict(
        gamma=dict(aliases=[], dtype=float),
        patience=dict(aliases=[], dtype=int),
        factor=dict(aliases=[], dtype=float),
        last_epoch=dict(aliases=[], dtype=int),
        step_size=dict(aliases=[], dtype=int),
        T_max=dict(aliases=[], dtype=int),
        mode=dict(aliases=[], dtype=str),
    )),
    starting_params=dict(aliases=[], dtype=object),
    swa=dict(aliases=['SWA'], dtype=bool),
    swa_args=dict(aliases=['SWA_args'], dtype=object, sub_params=dict(
        freq=dict(aliases=['swa_freq', 'SWA_freq'], dtype=int),
        lr=dict(aliases=['swa_lr', 'SWA_lr'], dtype=float),
        mode=dict(aliases=['swa_mode', 'SWA_mode'], dtype=str),
        start=dict(aliases=['swa_start', 'SWA_start'], dtype=int),
    )),
    transform_type=dict(aliases=['transform'], dtype=str),
    view_resample=dict(aliases=[], dtype=bool),
    views_class=dict(aliases=[], dtype=object),
    views_contr=dict(aliases=[], dtype=object),
)

def get_params(args, columns=[], values=[], prefix=''):
    ''' recursively unpack each base args item, making a string of
    combined keys '''
    if type(args) not in [SimpleNamespace, dict]:
        columns.append(prefix)
        values.append(args)
        return columns, values
    else:
        if type(args) is SimpleNamespace:
            args = vars(args)
        for k, v in args.items():
            if k.startswith('_'):
                continue
            column = '__'.join([prefix, k]) if prefix else k
            get_params(v, columns, values, column)
    return columns, values


def make_archive():
    model_dirs = sorted(glob.glob(f'{MODEL_DIR}/*/**/params', recursive=True))
    model_dirs = [op.dirname(i) for i in model_dirs if not
                  any([j in i for j in IGNORE_MODELS])]
    archive = pd.DataFrame()
    for m, model_dir in enumerate(model_dirs):

        # find args
        args = glob.glob(f'{model_dir}/args.*')
        if not len(args):
            args = glob.glob(f'{model_dir}/config.json')
        if not len(args):
            args = glob.glob(f'{model_dir}/config.pkl')
        if not len(args):
            args = glob.glob(f'{model_dir}/config.py')
        assert len(args) == 1, f'{len(args)} args found for {model_dir}'
        args = args[0]

        # load args
        if args.endswith('.pkl'):
            import pickle
            with open(args, 'rb') as f:
                args = pickle.load(f)
        elif args.endswith('.json'):
            import json
            with open(args, 'r') as f:
                args = json.load(f)
        elif args.endswith('.py'):
            import sys
            sys.path.append(op.dirname(args))
            from config import args
            sys.path.remove(op.dirname(args))

        # unpack args
        if type(args) is not dict:
            args = vars(args)
        columns_orig, values_orig = [], []
        for k, v in args.items():
            columns_orig, values_orig = get_params(v, columns_orig, values_orig, k)

        # convert to current format
        columns, values, dtypes = [], [], []
        for c_orig, v_orig in zip(columns_orig, values_orig):

            # skip ignored params
            ignore=False
            for param in IGNORE_PARAMS:
                if param in c_orig:
                    ignore=True
            if ignore:
                continue

            # remove M,D,T wrappers
            column = (c_orig
                      .replace('M__', '')
                      .replace('D__', '')
                      .replace('T__', '')
                      .replace('O__', ''))

            # convert coverage to visibility
            if 'coverage' in c_orig:
                column = column.replace('coverage', 'visibility')
                value = [1-j for j in value]
            else:
                value = v_orig

            # convert criterion format
            if column in ['classification', 'contrastive',
                          'contrastive_supervised', 'supervised_contrastive']\
                    and value == False:
                continue
            elif column == 'classification'  or (column == \
                    'learning' and value == 'supervised_classification'):
                column = 'criterion'
                value = 'CrossEntropyLoss'
            elif column == 'contrastive' or (column == 'learning' and value == \
                    'unsupervised_contrastive'):
                column = 'criterion'
                value = 'SimCLRLoss'
            elif column in ['contrastive_supervised', 'supervised_contrastive'] or (
                    column == 'learning' and value == 'supervised_contrastive'):
                column = 'criterion'
                value = 'SimCLRSupLoss'

            # find current parameter name
            current_format = []
            orig_format = column.split('__')
            if orig_format[0] in PARAMS:
                current_format.append(orig_format[0])
            else:
                for param, param_info in PARAMS.items():
                    if orig_format[0] in param_info['aliases']:
                        current_format.append(param)
                        break
                    if 'sub_params' in param_info:
                        for sub_param, sub_param_info in param_info['sub_params'].items():
                            if orig_format[0] in [sub_param] + sub_param_info['aliases']:
                                current_format.append(param)
                                current_format.append(sub_param)
                                break
            assert(len(current_format)), f'{orig_format[0]} not found'
            if len(orig_format) == 2:
                if orig_format[1] in PARAMS[current_format[0]]['sub_params']:
                    current_format.append(orig_format[1])
                else:
                    for sub_param, sub_param_info in PARAMS[current_format[0]][
                                                     'sub_params'].items():
                        if orig_format[1] in sub_param_info['aliases']:
                            current_format.append(sub_param)
                            break
                assert(len(current_format) == 2), f'{orig_format[1]} not found'
            column = '__'.join(current_format)

            # new contrastive transform formatting
            if column == 'transform_type' and 'contrastive' in value:
                if 'weak-resize' in model_dir:
                    value = 'contrastive-weak-resize'
                else:
                    value = 'contrastive'

            # get dtype (also checks that current format exists in PARAMS)
            temp_dict = PARAMS[current_format[0]]
            if len(current_format) > 1:
                for j in current_format[1:]:
                    temp_dict = temp_dict['sub_params'][j]
            dtype = temp_dict['dtype']

            # use defualt values when int parameter is missing
            value = 999 if dtype == int and value is None else value

            columns.append(column)
            values.append(value)
            dtypes.append(dtype)

        # add params not stored in args
        epochs = sorted(glob.glob(f'{model_dir}/params/???.pt'))
        last_epoch = int(op.basename(epochs[-1]).split('.')[0]) if len(epochs) \
            else -1
        columns.append('last_epoch')
        values.append(last_epoch)
        dtypes.append(int)

        best_epoch = sorted(glob.glob(f'{model_dir}/params/best*.pt'))
        best_epoch = op.basename(best_epoch[-1]) if len(best_epoch) else last_epoch
        if type(best_epoch) is str:
            if len(best_epoch) == 11:
                best_epoch = int(best_epoch[5:8])
            else:
                best_epoch = -1
        columns.append('best_epoch')
        values.append(best_epoch)
        dtypes.append(int)

        done = True if op.isfile(f'{model_dir}/done') else False
        columns.append('done')
        values.append(done)
        dtypes.append(bool)

        columns.append('model_dir')
        model_dir_short = model_dir.split('models/')[-1]
        values.append(model_dir_short)
        dtypes.append(str)

        if 'architecture' not in columns:
            columns.append('architecture')
            values.append(model_dir_short.split('/')[0])
            dtypes.append(str)

        if last_epoch not in [0, 1]:
            if op.isfile(f'{model_dir}/performance.csv'):
                perf_data = pd.read_csv(f'{model_dir}/performance.csv')
            elif op.isfile(f'{model_dir}/epoch_stats.pkl'):
                with open(f'{model_dir}/epoch_stats.pkl', 'rb') as f:
                    perf_dict = pickle.load(f)
                metrics = list(perf_dict['train'].keys())
                epochs = sorted(list(perf_dict['train'][metrics[0]].keys()))
                perf_data = pd.DataFrame()
                for epoch, train_eval in itp(
                        epochs, ['train', 'eval']):
                    perf_data = pd.concat([perf_data, pd.DataFrame({
                        **{'epoch': [epoch], 'train_eval': [train_eval]},
                        **{metric: [perf_dict[train_eval][metric][epoch]]
                           for metric in metrics}})])
                perf_data.to_csv(f'{model_dir}/performance.csv', index=False)
            perf_data = perf_data[perf_data.train_eval == 'eval']
            if 'acc1' in perf_data.columns:
                performance = perf_data.acc1.max()
            else:
                metric = [i for i in perf_data.columns if 'loss' in i.lower()][0]
                performance = perf_data[metric].min()
            if type(performance) is str and 'tensor' in performance:
                performance = float(performance[7:-1])
            columns.append('performance')
            values.append(performance)
            dtypes.append(float)

        # create dataframe for model
        unpacked_args = pd.DataFrame({
            c: pd.Series([v], dtype=d.__name__) for c, v, d in zip(columns, values,
                                                             dtypes)
        })
        assert not all(unpacked_args.iloc[0].isnull()), f'null values'
        
        # store in model_dir
        unpacked_args.to_csv(f'{model_dir}/args_modern_format.csv', index=False)

        # add to archive
        archive = pd.concat([archive, unpacked_args]).reset_index(drop=True)

    archive.to_csv(f'{MODEL_DIR}/model_summary.csv', index=False)


if __name__ == '__main__':

    #make_archive()

    """
    Use the code below to search for models with specific parameters
    """
    archive = pd.read_csv(f'{MODEL_DIR}/model_summary.csv')

    architectures = ['cornet_s_plus', 'resnet101', 'efficientnet_b1']
    min_vis = 0.5
    transform_type = 'contrastive-weak-resize'
    columns = ['architecture', 'transform_type', 'Occlusion__visibility',
               'Occlusion__form', 'model_dir', 'done', 'criterion',
               'performance']

    results = archive[
        (archive.architecture.isin(architectures)) &
        (archive.transform_type == transform_type)
    ][columns]

    results = results[results.Occlusion__visibility.apply(
        lambda x: np.min(x) >= min_vis)]

