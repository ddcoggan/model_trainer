from collections import OrderedDict
import torch
from torch import nn


HASH = '933c001c'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x


class CORblock_RT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp, state=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
        else:
            inp = self.conv_input(inp)
            inp = self.norm_input(inp)
            inp = self.nonlin_input(inp)

        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        x = self.norm1(x)
        x = self.nonlin1(x)

        state = self.output(x)
        output = state
        return output, state


class CORnet_RT(nn.Module):
    
    def __init__(self, hw=3, num_cycles=5):
        super().__init__()

        # parameters
        self.blocks = ['V1', 'V2', 'V4', 'IT']
        self.num_cycles = num_cycles

        # architecture
        self.V1 = CORblock_RT(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RT(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RT(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RT(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(hw)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(hw**2 * 512, 1000))
        ]))

    # version that stores all states and returns all at once 
    def forward(self, inp):

        num_cycles = inp.shape[0] if len(inp.shape) == 5 else self.num_cycles

        # initialize block states
        states = {block: 0 for block in self.blocks + ['output']}

        for c in range(num_cycles):
            
            # if image, reinput at each cycle; if movie, input frame sequence
            inp_c = inp if len(inp.shape) == 4 else inp[c]
            
            for b, block in enumerate(self.blocks):

                prv_block = ([None] + self.blocks)[b]

                # get feedforward inputs from inp or prev block
                f = inp_c if block == 'V1' else states[prv_block]

                # get lateral inputs from current block
                l = states[block] if c > 0 else None

                # forward pass
                outputs = getattr(self, block)(f, l)

                # store outputs
                states[block] = outputs[1]

                
        return self.decoder(outputs[0])


if __name__ == "__main__":

    model = CORnet_RT()
    params = torch.load('/mnt/HDD2_16TB/projects/p022_occlusion/in_silico/models/cornet_rt_hw3/transform-contrastive/params/best.pt')
    model.load_state_dict(params['model'])
    inputs = torch.rand([5, 3, 224, 224])
    for cycle in model(inputs):
        print(cycle['V1']['f'].shape)

