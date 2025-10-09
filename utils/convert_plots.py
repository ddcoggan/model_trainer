# /usr/bin/python
# Created by David Coggan on 2023 02 09

# overwrites contrastive loss plots in a model directory with a log scale version of the plot

import glob
import pickle as pkl
import os
import os.path as op
import matplotlib.pyplot as plt

params_dirs = sorted(glob.glob(f'in_silico/data/cornet_s_custom*/**/*/params', recursive=True))
for params_dir in params_dirs:

    stats_path = glob.glob(f'{op.dirname(params_dir)}/*tats.pkl')
    if not len(stats_path):
        stats_path = glob.glob(f'{op.dirname(params_dir)}/plots/*tats.pkl')

    assert len(stats_path) == 1
    stats_path = stats_path[0]

    stats = pkl.load(open(stats_path, 'rb'))

    for key in stats['train']:
        if 'loss' in key:
            epochs_train = sorted(stats['train'][key].keys())
            epochs_eval = sorted(stats['eval'][key].keys())
            plt.plot(epochs_train, [stats['train'][key][epoch] for epoch in epochs_train], label='train')
            plt.plot(epochs_eval, [stats['eval'][key][epoch] for epoch in epochs_eval], label='eval')
            plt.xlabel('epoch')
            plt.yscale('log')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(f'{op.dirname(params_dir)}/plots/{key}.png'))
            plt.show()
            plt.close()
