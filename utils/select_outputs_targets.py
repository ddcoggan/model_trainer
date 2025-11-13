import numpy as np
import torch

def select_outputs_targets(outputs, targets, args):
    
    views = np.arange(args.num_views)

    # handling for model with multiple different-sized outputs in list form
    if 'output_avpool' in args.architecture:
        if 'SimCLR' in args.criterion:
            outputs = outputs[0]
            outputs = torch.stack(torch.split(
                outputs, [targets.shape[0]] * args.num_views, dim=0), dim=1)
            outputs = outputs[:, views]
        elif args.criterion == 'CrossEntropyLoss':
            outputs = outputs[1]
            outputs = torch.stack(torch.split(
                outputs, [targets.shape[0]] * args.num_views, dim=0), dim=1)
            outputs = outputs[:, views]
            outputs = torch.concat(torch.split(
                outputs, [1] * outputs.shape[1], dim=1), dim=0).squeeze()

    # handling for model with multiple same-sized outputs in tensor form
    elif args.architecture in ['cornet_s_custom', 'cornet_st'] and \
            args.architecture_args['M'].out_channels == 2:
        if args.criterion.startswith('SimCLR'): 
            outputs = outputs[:, :, 1]
        elif args.criterion == 'CrossEntropyLoss':
            outputs = outputs[:, :, 0]
        
    # for SimCLR, ensure view dimension is separated from batch dimension
    elif 'SimCLR' in args.criterion and len(outputs.shape) == 2:
        outputs = torch.unflatten(outputs, 0, sizes=(outputs.shape[0] // args.num_views, args.num_views))

    # for supervised recurrent models, ensure cycle and batch dims are combined
    elif 'SimCLR' not in args.criterion and len(outputs.shape) == 3:
        targets = torch.stack([targets] * outputs.shape[1]) \
            .transpose(0, 1).flatten()
        outputs = outputs.reshape(-1, outputs.shape[-1])
        
    # to prevent SimCLR from using supervised method, do not provide targets
    if args.criterion == 'SimCLRLoss':
        targets = None

    return outputs, targets
