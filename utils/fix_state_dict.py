# /usr/bin/python
# Created by David Coggan on 2022 10 21


# renames state_dict items, replacing any string instance with new instance

import os, glob, torch
import os.path as op
renaming_config = {
    'decoder.linear.': 'decoder.linear_1_1.'
}
os.chdir('/home/tonglab/david/projects/p022_occlusion')
for folder, subfolders, subfiles in os.walk(f'{os.getcwd()}/in_silico/data/cornet_s_custom'):
    for file in sorted(subfiles):
        if file.endswith('.pt') and len(file) == 6:
            Exception()
            params_path = f'{folder}/{file}'
            if not op.isfile(f'{params_path[:-3]}_old.pt'):
                params = torch.load(params_path)
                new_params = {}
                altered_any = False
                for dict_name in ['model','optimizer']:
                    new_dict = {}
                    for old_key, item in params[dict_name].copy().items():
                        altered = False
                        for old_string, new_string in renaming_config.items():
                            if old_string in old_key:
                                new_key = old_key.replace(old_string,new_string)
                                new_dict[new_key] = item
                                altered_any = True
                                altered = True
                        if not altered:
                            new_dict[old_key] = item
                    new_params[dict_name] = new_dict

                if altered_any:
                    print(f'changes made to {params_path}')
                    os.rename(params_path, f'{params_path[:-3]}_old.pt')
                    torch.save(new_params, params_path)
                else:
                    print(f'no changes made to {params_path}')

