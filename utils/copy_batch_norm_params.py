# /usr/bin/python
# Created by David Coggan on 2022 10 21

# copies batch_norm params to transfer learning params
# in case they were left tracking during transfer learning
import os, glob, torch
import os.path as op
os.chdir('/home/tonglab/david/projects/p022_occlusion')
for model_dir in glob.glob(f'{os.getcwd()}/in_silico/models/cornet_s/*'):
    transfer_dirs = glob.glob(f'{model_dir}/transfer*')
    if transfer_dirs:
        params_path_orig = sorted(glob.glob(f'{model_dir}/params/*.pt'))[-1]
        params_orig = torch.load(params_path_orig)
        for transfer_dir in transfer_dirs:
            params_paths_transfer = sorted(glob.glob(f'{transfer_dir}/params/*.pt'))
            for params_path_transfer in params_paths_transfer:
                print(params_path_transfer)
                params_transfer = torch.load(params_path_transfer)
                for key in params_transfer['model']:
                    if 'norm' in key:
                        try:
                            params_transfer['model'][key] = params_orig['model'][key]
                        except:
                            params_transfer['model'][key] = params_orig['model'][f'module.{key}']
                torch.save(params_transfer, params_path_transfer)
