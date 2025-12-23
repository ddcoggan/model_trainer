This repository contains a pytorch pipeline for training DNN models, which I have used to train all models in my work. 
It is capable of training any model I have used in my work.
Generally, it works by configuring a training job with a config file. Running 'train_queued_models.py' then iterates through the training job, managing hardware according to user input at the beginning.

// model
"architecture": "cornet_s_custom",  // used to load architecture and name of top-level results directory
//architecture_args = dict(),  // key word args when initializing the model

// cornet_s_custom params
//architecture_args = dict(
//    "R": (1,2,4,2),          // recurrence, default = (1,2,4,2),
//    "K": (3,3,3,3),          // kernel size, default = (3,3,3,3),
//    "F": (128,128,256,512),   // feature maps, default = (64,128,256,512)
//    "S": 4,                  // feature maps scaling, default = 4
//    "out_channels": 1,       // number of heads, default = 1
//    "head_depth": 2,         // multi-layer head, default": 1
//    "head_width": 3,         // gridsize of adaptive avgpool layer,
//)),

"model_dir": "~/david/models/cornet_s_plus/occ-nat1-untex-weak_xform_cont", //null, // manual override for results directory
"finetune": False,  // starting a fine-tuning regime from pretrained weights?
"finetune_args": {
    "starting_params: null, //"/home/tonglab/david/masterScripts/DNN/zoo/pretrained_weights/cornet_s-1d3f7974.pth",  // starting params for fine-tuning
    "finetune_dir: "finetune_unocc",  // third-level directory for finetuned model results
    "freeze_modules: ["V1", "V2", "V4","IT"],  // freeze the weights of these modules during finetuning

// dataset
"dataset": "ILSVRC2012",
"class_subset": null,  // list of class indices to include in the dataset
"num_views": 1,  // number of views to generate per example, default = 1
"transform_type": "contrastive",  // "contrastive", "contrastive-weak-resize" or "default"
"cutmix": null,  // dict(prob=1.0, alpha=1, beta=1, frgrnd=True),
"Occlusion": {
    "form": "naturalUntexturedCropped1",       // occluder type or list thereof
    "probability": 0.8,                        // probability that each image will be occluded, range(0,1)
    "visibility": [.5, .6, .7, .8, .9],        // image visibility or list thereof, range(0,1), or 'all'
    "color": "random",                         // occluder colours (RGB uint8 / "random" or list thereof. Ignored for "textured" forms)
    "views": [0],  				   // views to apply occlusion to, e.g. [0,2] will apply occluders to 1st and 3rd views
},

// optimization
"num_epochs": 43,  // number of epochs to train for
"batch_size": 256,  // minibatch size
"optimizer": "SGD",  // any optimizer from torch.optim
"optimizer_args": {
    "lr": 0.1,  // set float or one of the following strings: "LRfinder",
    // "batch_linear", "batch_nonlinear"
    "weight_decay": 1e-4,
    "momentum": .9,
},
"scheduler": "StepLR",  //"ReduceLROnPlateau", //  // any attribute of torch.optim.lr_scheduler
"scheduler_args": {
    "gamma": 0.1, "step_size": 20,  // StepLR
    //"patience": 10, "factor": .1, "mode": "max",  // ReduceLROnPlateau
},
"criterion": "CrossEntropyLoss",
"primary_metric": "acc1",  // metric for "best" params and lr scheduler
"save_interval": null,  // preserve params at every n epochs
"checkpoint": null,  // integer, resume training from this epoch (default: most recent, -1: from scratch)
"swa": null, // dict(start=8, anneal_epochs=4, swa_lr=.05)  // type: dict
"amp": True,  // automatic mixed precision. Can speed up optimization but can also cause over/underflow
"loss_cycle": "uniform",  // any string value applies loss to all output cycles with particular weight distribution, int returns that one cycle, default (i.e. no attribute) is last cycle
}