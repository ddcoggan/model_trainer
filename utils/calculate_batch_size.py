import torch
from torchinfo import summary
import math

def calculate_batch_size(model, device):

    mod_sum = str(summary(model, input_size=(1,3, 224, 224), device=device)).split('\n')[-4:-2]
    image_mem = float(mod_sum[0].split(' ')[-1])
    image_mem_unit = mod_sum[0].split(' ')[-2][1:3]
    if image_mem_unit == 'KB':
        image_mem *= 1e3
    elif image_mem_unit == 'MB':
        image_mem *= 1e6
    elif image_mem_unit == 'GB':
        image_mem *= 1e9
    device_mem = (torch.cuda.mem_get_info(device)[0] *
                  torch.cuda.device_count()) # model is already on device, so get remaining memory
    max_batch = device_mem / image_mem
    batch_size = math.floor(pow(2, int(math.log(max_batch, 2)))) # largest power of 2

    return batch_size
