import matplotlib.pyplot as plt
import numpy as np

def plot_performance(performance, metrics, outdir):

    num_plots = len(metrics) + 1
    fig, axes = plt.subplots(nrows=num_plots,
                             figsize=(4, num_plots * 2), sharex=True)
    train_vals = performance[performance['train_eval'] == 'train']
    eval_vals = performance[performance['train_eval'] == 'eval']
    if not eval_vals.empty:
        steps_per_epoch = eval_vals.step[eval_vals.epoch == 0].values[0]
        step_to_epoch = lambda step: step / steps_per_epoch
        epoch_to_step = lambda epoch: epoch * steps_per_epoch
    for m, metric in enumerate(metrics + ['lr']):
        ax = axes.flatten()[m]
        train_x, train_y = (train_vals['step'].values,
                            train_vals[metric].values)
        eval_x, eval_y = (eval_vals['step'].values,
                          eval_vals[metric].values)
        ax.plot(train_x, train_y, label='train', zorder=2)
        ax.plot(eval_x, eval_y, label='eval', zorder=3)
        
        ax.set_ylabel(metric)
        if 'acc' not in metric:
            ax.set_yscale('log')
        if eval_vals.epoch.max() > 1 :
            if m == 0:
                ax2 = ax.secondary_xaxis('top', functions=(step_to_epoch, epoch_to_step))
                ax2.set_xlabel('epoch')
                ax2.grid(axis='x', linestyle='solid', alpha=.25, zorder=1)
        if m == len(metrics):
            ax.set_xlabel('optimization step')
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 2))
            ax.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/performance.pdf')
    plt.close()
    
if __name__ == '__main__':
    
    import pandas as pd
    import os.path as op
    performance_path = ('/mnt/HDD2_16TB/models/cornet_s/classification_natural'
                        '-shape/performance.csv')
    performance = pd.read_csv(performance_path)
    outdir = op.dirname(performance_path)
    metrics = ['CrossEntropyLoss', 'acc1', 'acc5']
    plot_performance(performance, metrics, outdir)
