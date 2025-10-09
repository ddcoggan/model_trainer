from collections import OrderedDict
import torch
from torch import nn
import numpy as np

HASH = '933c001sflowtohigh'

class filtering(nn.Module):
    def __init__(self, sfCutOffCPD,isLo,ppd):
        self.sfCutOffCPD = sfCutOffCPD
        self.isLo = isLo
        self.ppd = ppd
        super().__init__()
    def forward(self, x):
        self.s = x.size()[2]
        self.sfCutOff = self.sfCutOffCPD * self.s/self.ppd
        self.fNyquist = self.s/2
        self.sfCutOff = self.sfCutOff/self.fNyquist
        self.t = (torch.range(1,self.s)-(self.s/2))*2/self.s
        self.t1,self.t2 = torch.meshgrid(self.t,self.t)
        self.t3 = self.t1**2 + self.t2**2
        d = x.device
        self.w = torch.ones([self.s,self.s]).to(d)
        if self.isLo == 1:
            self.w[self.t3>self.sfCutOff**2] = 0
        else:
            self.w[self.t3<=self.sfCutOff**2] = 0
        self.fftx = torch.fft.fftn(x)
        self.fftshiftx = torch.fft.fftshift(self.fftx)
        self.fftfilteredx = self.fftshiftx * self.w
        self.ifftshiftx = torch.fft.ifftshift(self.fftfilteredx)
        self.ifftx = torch.real(torch.fft.ifftn(self.ifftshiftx))
        return self.ifftx

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

## for model training
class CORblock_RTSFLOWTOHIGH(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=kernel_size // 2)
        self.norm_input = nn.GroupNorm(32, out_channels)
        self.nonlin_input = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.nonlin1 = nn.ReLU(inplace=True)

        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp=None, state=None, batch_size=None):
        if inp is None:  # at t=0, there is no input yet except to V1
            inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
            if self.conv_input.weight.is_cuda:
                inp = inp.cuda()
            # conv_input = np.array(inp.detach().to('cpu'))
            # norm_input = np.array(inp.detach().to('cpu'))
            # nonlin_input = np.array(inp.detach().to('cpu'))

        else:
            inp = self.conv_input(inp)
            # conv_input = np.array(inp.detach().to('cpu'))
            inp = self.norm_input(inp)
            # norm_input = np.array(inp.detach().to('cpu'))
            inp = self.nonlin_input(inp)
            # nonlin_input = np.array(inp.detach().to('cpu'))


        if state is None:  # at t=0, state is initialized to 0
            state = 0
        skip = inp + state

        x = self.conv1(skip)
        # conv1 = np.array(x.detach().to('cpu'))
        x = self.norm1(x)
        # norm1 = np.array(x.detach().to('cpu'))
        x = self.nonlin1(x)
        # nonlin1 = np.array(x.detach().to('cpu'))
        state = self.output(x)
        output = state
        return output, state


class CORnet_RTSFLOWTOHIGH(nn.Module):

    def __init__(self, times=5,ppd = 4, cpdCutOff = 1,timeStepCutoff = 2):
        super().__init__()
        self.times = times
        self.timeStepCutoff = timeStepCutoff
        self.filterLo = filtering(sfCutOffCPD=cpdCutOff,isLo = 1,ppd = ppd)
        self.filterHi = filtering(sfCutOffCPD=cpdCutOff,isLo = 0,ppd = ppd)
        self.V1 = CORblock_RTSFLOWTOHIGH(3, 64, kernel_size=7, stride=4, out_shape=56)
        self.V2 = CORblock_RTSFLOWTOHIGH(64, 128, stride=2, out_shape=28)
        self.V4 = CORblock_RTSFLOWTOHIGH(128, 256, stride=2, out_shape=14)
        self.IT = CORblock_RTSFLOWTOHIGH(256, 512, stride=2, out_shape=7)
        self.decoder = nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, 16))
        ]))

    def forward(self, inp):
        outputs = {'inp': inp}
        states = {}
        blocks = ['inp', 'V1', 'V2', 'V4', 'IT']
        all_states = []
        # temp_states = []
        # conv_inputs, norm_inputs, nonlin_inputs, conv1s, norm1s, nonlin1s = [], [], [], [], [], []
        # temp_conv_inputs, temp_norm_inputs, temp_nonlin_inputs, temp_conv1s, temp_norm1s, temp_nonlin1s = [], [], [], [], [], []
        for block in blocks[1:]:
            if block == 'V1':  # at t=0 input to V1 is the image
                this_inp = self.filterLo(outputs['inp'])
            else:  # at t=0 there is no input yet to V2 and up
                this_inp = None
            new_output, new_state= getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
            outputs[block] = new_output
            states[block] = new_state

        for t in range(1, self.times):
            new_outputs = {'inp': inp}
            for block in blocks[1:]:
                prev_block = blocks[blocks.index(block) - 1]
                prev_output = outputs[prev_block]
                if np.logical_and(block == 'V1',t <= self.timeStepCutoff):
                    prev_output = self.filterLo(prev_output)
                elif np.logical_and(block == 'V1',t > self.timeStepCutoff):
                    prev_output = self.filterHi(prev_output)
                prev_state = states[block]
                new_output, new_state = getattr(self, block)(prev_output, prev_state)
                new_outputs[block] = new_output
                states[block] = new_state

            outputs = new_outputs

        out = self.decoder(outputs['IT'])
        return out


## for test surround supp
# class CORblock_RTSFLOWTOHIGH(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, out_shape=None):
#         super().__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.out_shape = out_shape
#
#         self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
#                                     stride=stride, padding=kernel_size // 2)
#         self.norm_input = nn.GroupNorm(32, out_channels)
#         self.nonlin_input = nn.ReLU(inplace=True)
#
#         self.conv1 = nn.Conv2d(out_channels, out_channels,
#                                kernel_size=3, padding=1, bias=False)
#         self.norm1 = nn.GroupNorm(32, out_channels)
#         self.nonlin1 = nn.ReLU(inplace=True)
#
#         self.output = Identity()  # for an easy access to this block's output
#
#     def forward(self, inp=None, state=None, batch_size=None):
#         if inp is None:  # at t=0, there is no input yet except to V1
#             inp = torch.zeros([batch_size, self.out_channels, self.out_shape, self.out_shape])
#             if self.conv_input.weight.is_cuda:
#                 inp = inp.cuda()
#             conv_input = np.array(inp.detach().to('cpu'))
#             norm_input = np.array(inp.detach().to('cpu'))
#             nonlin_input = np.array(inp.detach().to('cpu'))
#
#         else:
#             inp = self.conv_input(inp)
#             # conv_input = inp
#             conv_input = np.array(inp.detach().to('cpu'))
#             inp = self.norm_input(inp)
#             # norm_input = inp
#             norm_input = np.array(inp.detach().to('cpu'))
#             inp = self.nonlin_input(inp)
#             # nonlin_input = inp
#             nonlin_input = np.array(inp.detach().to('cpu'))
#
#
#         if state is None:  # at t=0, state is initialized to 0
#             state = 0
#         skip = inp + state
#
#         x = self.conv1(skip)
#         # conv1 = x
#         conv1 = np.array(x.detach().to('cpu'))
#         x = self.norm1(x)
#         # norm1 = x
#         norm1 = np.array(x.detach().to('cpu'))
#         x = self.nonlin1(x)
#         # nonlin1 = x
#         nonlin1 = np.array(x.detach().to('cpu'))
#         state = self.output(x)
#         output = state
#         return output, state, conv_input, norm_input, nonlin_input, conv1, norm1, nonlin1
#
#
# class CORnet_RTSFLOWTOHIGH(nn.Module):
#
#     def __init__(self, times=5):
#         super().__init__()
#         self.times = times
#         self.filterLo = filtering(sfCutOffCPD=1,isLo = 1,ppd = 4)
#         self.filterHi = filtering(sfCutOffCPD=1,isLo = 0,ppd = 4)
#         self.V1 = CORblock_RTSFLOWTOHIGH(3, 64, kernel_size=7, stride=4, out_shape=56)
#         self.V2 = CORblock_RTSFLOWTOHIGH(64, 128, stride=2, out_shape=28)
#         self.V4 = CORblock_RTSFLOWTOHIGH(128, 256, stride=2, out_shape=14)
#         self.IT = CORblock_RTSFLOWTOHIGH(256, 512, stride=2, out_shape=7)
#         self.decoder = nn.Sequential(OrderedDict([
#             ('avgpool', nn.AdaptiveAvgPool2d(1)),
#             ('flatten', Flatten()),
#             ('linear', nn.Linear(512, 1000))
#         ]))
#
#     def forward(self, inp):
#         outputs = {'inp': inp}
#         states = {}
#         blocks = ['inp', 'V1', 'V2', 'V4', 'IT']
#         all_states = []
#         temp_states = []
#         conv_inputs, norm_inputs, nonlin_inputs, conv1s, norm1s, nonlin1s = [], [], [], [], [], []
#         temp_conv_inputs, temp_norm_inputs, temp_nonlin_inputs, temp_conv1s, temp_norm1s, temp_nonlin1s = [], [], [], [], [], []
#         for block in blocks[1:]:
#             if block == 'V1':  # at t=0 input to V1 is the image
#                 this_inp = self.filterLo(outputs['inp'])
#             else:  # at t=0 there is no input yet to V2 and up
#                 this_inp = None
#             new_output, new_state, conv_input, norm_input, nonlin_input, conv1, norm1, nonlin1= getattr(self, block)(this_inp, batch_size=len(outputs['inp']))
#             outputs[block] = new_output
#             states[block] = new_state
#
#             temp_states.append(np.array(new_state.detach().to('cpu')))
#
#             temp_conv_inputs.append(np.array(conv_input))
#             temp_norm_inputs.append(np.array(norm_input))
#             temp_nonlin_inputs.append(np.array(nonlin_input))
#             temp_conv1s.append(np.array(conv1))
#             temp_norm1s.append(np.array(norm1))
#             temp_nonlin1s.append(np.array(nonlin1))
#
#         all_states.append(temp_states)
#         conv_inputs.append(temp_conv_inputs)
#         norm_inputs.append(temp_norm_inputs)
#         nonlin_inputs.append(temp_nonlin_inputs)
#         conv1s.append(temp_conv1s)
#         norm1s.append(temp_norm1s)
#         nonlin1s.append(temp_nonlin1s)
#         for t in range(1, self.times):
#             new_outputs = {'inp': inp}
#             temp_conv_inputs, temp_norm_inputs, temp_nonlin_inputs, temp_conv1s, temp_norm1s, temp_nonlin1s = [], [], [], [], [], []
#             temp_states = []
#             for block in blocks[1:]:
#                 prev_block = blocks[blocks.index(block) - 1]
#                 prev_output = outputs[prev_block]
#                 if block == 1 & t <=2:
#                     prev_output = self.filterLo(prev_output)
#                 elif block == 1 & t > 2:
#                     prev_output = self.filterHi(prev_output)
#                 prev_state = states[block]
#                 new_output, new_state, conv_input, norm_input, nonlin_input, conv1, norm1, nonlin1 = getattr(self, block)(prev_output, prev_state)
#                 new_outputs[block] = new_output
#                 states[block] = new_state
#
#                 temp_states.append(np.array(new_state.detach().to('cpu')))
#
#                 temp_conv_inputs.append(np.array(conv_input))
#                 temp_norm_inputs.append(np.array(norm_input))
#                 temp_nonlin_inputs.append(np.array(nonlin_input))
#                 temp_conv1s.append(np.array(conv1))
#                 temp_norm1s.append(np.array(norm1))
#                 temp_nonlin1s.append(np.array(nonlin1))
#
#             outputs = new_outputs
#
#             all_states.append(temp_states)
#             conv_inputs.append(temp_conv_inputs)
#             norm_inputs.append(temp_norm_inputs)
#             nonlin_inputs.append(temp_nonlin_inputs)
#             conv1s.append(temp_conv1s)
#             norm1s.append(temp_norm1s)
#             nonlin1s.append(temp_nonlin1s)
#
#         out = self.decoder(outputs['IT'])
#         return all_states,conv_inputs, norm_inputs, nonlin_inputs, conv1s, norm1s, nonlin1s,out


