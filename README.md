# Model Training Pipeline

This repository contains a pytorch pipeline for training DNN models, which I 
have used to train all models in my work. Generally, it works by configuring 
a training job in a json file (see below) and placing this in the 
`training_queue` directory. Running `train_queued_models.py` then 
iterates through the training jobs, managing hardware according to user input 
requested at the beginning. 

Below is a list of parameters for the config file with brief documentation, 
broken down into broad categories and in alphabetical order. Not all 
parameters must be specified, and default values for those that are required 
can be found in `utils/complete_config.py`.

### Model

architecture
: string name of model architecture to use. Must be in 
the zoo or the torchvision library (lower-case string, e.g., `"resnet101"`)

architecture_args
: dictionary containing any kwargs to pass to the model at 
instantiation

model_dir
: path to the top-level output directory

save_interval
: how often to save model parameters during training. E.g. 
`10` will save weights at epochs 10, 20, 30, etc. Set to 
`null` to only save the best and final parameters.


### Data

batch_dependence
: boolean determining whether the same (`true`) or different (`false`) 
random augmentations are applied to all images in a batch.

class_subset
: list of class indices to include in the dataset. Set to `null` 
to include all classes.

cutmix
: boolean specifying whether to use cutmix data augmentation. Currently, can 
only use the default parameters.

dataset
: string specifying directory basename for the dataset.

image_size
: a single integer specifying the height and width of input images.

mixup
: boolean specifying whether to use mixup data augmentation. Currently, can 
only use the default parameters.

Noise:
: dictionary containing the parameters for applying Fourier or Gaussian 
noise to images (see `utils/Noise.py`).

num_views
: integer specifying the number of views to generate per example (e.g., for 
SimCLR).

Occlusion
: dictionary containing the parameters for applying the [Visual Occluders 
Dataset](https://github.com/ddcoggan/VisualOccludersDataset) to training 
images (see `utils/Occlude.py`).

randomerase
: the `value` parameter for the random erase data augmentation. Set to 
`false` to disable.

transform_type
: base dataset augmentations to apply (not including occlusion, noise etc). See 
`utils/get_transforms.py`.

video
: boolean determing whether different views constitute different frames 
of a video (e.g., gradual deblurring) or different views of the same image (e.
g., SimCLR).

view_dependence
: boolean determining whether different views of an image should have the 
same (`true`) or different (`false`) random augmentations applied.


### Optimization

amp
: boolean determining whether automatic mixed precision. Can speed up 
optimization but can also cause over/underflow.

batch_size
: integer specifying minibatch size. This accounts for multiple 
views of an image, e.g. a batch size of 256 with 2 views means 128 base 
images will be used per minibatch.

checkpoint
: integer specifying which epoch to resume training from. `-1` starts from 
random parameters, '0' starts from a saved initialization ('000.pt').

criterion
: string specifying loss function as named in pytorch, e.g. `"CrossEntropyLoss"` 
or, for SimCLR, use `"SimCLRLoss"`.

finetune
: boolean determining whether to begin training from a pretrained model.

finetune_args
: dictionary containing the following arguments for fine-tuning regime:

  * starting_params: string path to the pretrained weights.
  * finetune_dir: string name of third-level results directory for fine-tuned 
    models.
  * freeze_modules: list of strings specifying model modules to freeze during 
    fine-tuning.

loss_cycle
: for recurrent models that produce multiple outputs across 
different processing cycles, this determines which cycle to obtain the loss 
from (int) or set to `uniform` to apply loss to all output cycles 
equally.

num_epochs
: integer specifying number of training epochs.

optimizer
: string specifying optimizer as named in `torch.optim`, e.g., `"SGD"` or 
`"Adam"`.

optimizer_args
: dictionary containing kwargs to pass to the optimizer at initialization.

primary_metric
: string specifying the validation metric to use for learning rate scheduling 
and saving the "best" model parameters (e.g., `acc1`).

scheduler
: string specifying learning rate scheduler as named in `torch.optim.
lr_scheduler`, e.g., `"StepLR"` or `"CosineAnnealingLR"`.

scheduler_args
: dictionary containing kwargs to pass to the scheduler at initialization.

swa
: dictionary containing keyword arguments for stochastic weight averaging. 
Set to `null` to disable.

warmup_scheduler
: string specifying the scheduler as named in `torch.optim.lr.scheduler` 
to use for warm-up, e.g., `"LinearLR"`.

warmup_scheduler_args
: dictionary containing keyword arguments passed to the warm-up scheduler at 
initialization.