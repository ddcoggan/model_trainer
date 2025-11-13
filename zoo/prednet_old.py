'''PredNet in PyTorch.'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class features2(nn.Module):
    def __init__(self, inchan, outchan, kernel_size=7, stride=2, padding=3, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inchan, outchan, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.GN = nn.GroupNorm(1,outchan)

    def forward(self, x):
        y = self.GN(self.relu(self.conv(x)))
        return y

# Feedforeward module
class FFconv2d(nn.Module):
    def __init__(self, inchan, outchan, downsample=False):
        super().__init__()
        self.conv2d = nn.Conv2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        if self.downsample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv2d(x)
        if self.downsample:
            x = self.Downsample(x)
        return x


# Feedback module
class FBconv2d(nn.Module):
    def __init__(self, inchan, outchan, upsample=False):
        super().__init__()
        self.convtranspose2d = nn.ConvTranspose2d(inchan, outchan, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample = upsample
        if self.upsample:
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        if self.upsample:
            x = self.Upsample(x)
        x = self.convtranspose2d(x)
        return x

   
# FFconv2d and FBconv2d that share weights
class Conv2d(nn.Module):
    def __init__(self, inchan, outchan, sample=False):
        super().__init__()
        self.kernel_size = 3
        self.weights = nn.init.xavier_normal(torch.Tensor(outchan,inchan,self.kernel_size,self.kernel_size))
        self.weights = nn.Parameter(self.weights, requires_grad=True)
        self.sample = sample
        if self.sample:
            self.Downsample = nn.MaxPool2d(kernel_size=2, stride=2)
            self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, feedforward=True):
        if feedforward:
            x = F.conv2d(x, self.weights, stride=1, padding=1)
            if self.sample:
                x = self.Downsample(x)
        else:
            if self.sample:
                x = self.Upsample(x)
            x = F.conv_transpose2d(x, self.weights, stride=1, padding=1)
        return x

# PredNet
class PredNet(nn.Module):

    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics) #number of layers

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(ics[i],ocs[i],downsample=sps[i]) for i in range(self.nlays)])
        # Feedback layers
        if cls > 0:
            self.FBconv = nn.ModuleList([FBconv2d(ocs[i],ics[i],upsample=sps[i]) for i in range(self.nlays)])

        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])

        #Group Normalization with group = 1
        self.GN = nn.ModuleList([nn.GroupNorm(1,ocs[i]) for i in range(self.nlays)])

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):

        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        xr[0] = self.GN[0](xr[0])
        for i in range(1,self.nlays):            
            xr.append(F.relu(self.FFconv[i](xr[i-1])))          
            xr[i] = self.GN[i](xr[i])

        # Dynamic process 
        for t in range(self.cls):

            # Feedback prediction
            xp = []
            for i in range(self.nlays-1,0,-1):
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a0 + xr[i-1]*(1-a0))
                xr[i-1] = self.GN[i-1](xr[i-1])

            # Feedforward prediction error
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            xr[0] = self.GN[0](xr[0])
            for i in range(1, self.nlays):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-xp[i-1])*b0 + xr[i])
                xr[i] = self.GN[i](xr[i])

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
 
        return out


# PredNet
class PredNetTied(nn.Module):
    def __init__(self, num_classes=10, cls=3):
        super().__init__()
        ics = [3,  64, 64,  128, 128, 256, 256, 256] # input chanels
        ocs = [64, 64, 128, 128, 256, 256, 256, 256] # output chanels
        sps = [False, False, True, False, True, False, False, False] # downsample flag
        self.cls = cls # num of circles
        self.nlays = len(ics) # num of circles

        #Group Normalization with group = 1
        self.GN = nn.ModuleList([nn.GroupNorm(1,ocs[i]) for i in range(self.nlays)])

        # Convolutional layers
        self.conv = nn.ModuleList([Conv2d(ics[i],ocs[i],sample=sps[i]) for i in range(self.nlays)])

        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ics[i],1,1)+0.5) for i in range(1,self.nlays)])
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1,ocs[i],1,1)+1.0) for i in range(self.nlays)])

        # Linear layer
        self.linear = nn.Linear(ocs[-1], num_classes)

    def forward(self, x):

        # Feedforward
        xr = [F.relu(self.conv[0](x))]        
        xr[0] = self.GN[0](xr[0])
        for i in range(1,self.nlays):
            xr.append(F.relu(self.conv[i](xr[i-1])))     
            xr[i] = self.GN[i](xr[i])

        # Dynamic process 
        for t in range(self.cls):

            # Feedback prediction
            xp = []
            for i in range(self.nlays-1,0,-1):
                xp = [self.conv[i](xr[i],feedforward=False)] + xp
                a = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a + xr[i-1]*(1-a))
                xr[i-1] = self.GN[i-1](xr[i-1])

            # Feedforward prediction error
            b = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.conv[0](x - self.conv[0](xr[0],feedforward=False))*b + xr[0])
            xr[0] = self.GN[0](xr[0])   
            for i in range(1, self.nlays):
                b = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.conv[i](xr[i-1]-xp[i-1])*b + xr[i])  
                xr[i] = self.GN[i](xr[i])

        # classifier                
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# PredNet ImageNet
class PredNetImageNet(nn.Module):

    def __init__(self, num_classes=1000, cls=3, pretrained=False):
        assert pretrained is False
        super().__init__()
        self.ics =     [    3,   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512] # input chanels (x, xr0, xr1, ..., xr10)
        self.ocs =     [   64,   64,  128,  128,  128,  128,  256,  256,  256,  512,  512,  512] # output chanels (x, xp0, xp1, ..., xp9, None)
        self.sps = [False,False, True,False, True,False, True,False,False, True,False,False] # downsample flag
        self.cls = cls  # num of circles
        self.nlays = len(self.ics)  # number of layers

        self.baseconv = features2(self.ics[0], self.ocs[0])

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(self.ics[i], self.ocs[i], downsample=self.sps[i]) for i in range(1, self.nlays)]) # (None, FF0, FF1, ..., FF10)
        # Feedback layers
        # if cls > 0:
        #     self.FBconv = nn.ModuleList([FBconv2d(self.ocs[i], self.ics[i], upsample=self.sps[i]) for i in range(1, self.nlays)]) # (None, FB0, FB1, ..., FB10)
        self.FBconv = nn.ModuleList([FBconv2d(self.ocs[i], self.ics[i], upsample=self.sps[i]) for i in range(1, self.nlays)]) # (None, FB0, FB1, ..., FB10)

        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ics[i], 1, 1) + 0.5) for i in range(2, self.nlays)]) # (None, None, a0_0, ..., a0_9)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ocs[i], 1, 1) + 1.0) for i in range(1, self.nlays)]) # (None, b0_0, b0_1, ..., b0-10)

        # Group Normalization with group = 1
        self.GN = nn.ModuleList([nn.GroupNorm(1, self.ocs[i]) for i in range(1, self.nlays)]) # (None, GN0, GN1, ..., GN10)

        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)

    def forward(self, x): # why self.nlays-1? because this includes features2() as the first layer like PCN-with-Local

        x = self.baseconv(x)

        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        xr[0] = self.GN[0](xr[0])
        # for i in range(1,self.nlays):
        for i in range(1,self.nlays-1):
            xr.append(F.relu(self.FFconv[i](xr[i-1])))
            xr[i] = self.GN[i](xr[i])

        # Dynamic process
        for t in range(self.cls):

            # Feedback prediction
            xp = []
            # for i in range(self.nlays-1,0,-1):
            for i in range(self.nlays-2,0,-1):
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i-1]).expand_as(xr[i-1])
                xr[i-1] = F.relu(xp[0]*a0 + xr[i-1]*(1-a0))
                xr[i-1] = self.GN[i-1](xr[i-1])

            # Feedforward prediction error
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xr[0] = F.relu(self.FFconv[0](x-self.FBconv[0](xr[0]))*b0 + xr[0])
            xr[0] = self.GN[0](xr[0])
            # for i in range(1, self.nlays):
            for i in range(1, self.nlays-1):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xr[i] = F.relu(self.FFconv[i](xr[i-1]-xp[i-1])*b0 + xr[i])
                xr[i] = self.GN[i](xr[i])

        # classifier
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
        

# PredNet ImageNet
class PredNetImageNet_detailedOutput(nn.Module):

    def __init__(self, num_classes=1000, cls=3):
        super().__init__()
        self.ics = [3, 64, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512]  # input chanels (x, xr0, xr1, ..., xr10)
        self.ocs = [64, 64, 128, 128, 128, 128, 256, 256, 256, 512, 512,
                    512]  # output chanels (x, xp0, xp1, ..., xp9, None)
        self.sps = [False, False, True, False, True, False, True, False, False, True, False, False]  # downsample flag
        self.cls = cls  # num of circles
        self.nlays = len(self.ics)  # number of layers

        self.baseconv = features2(self.ics[0], self.ocs[0])

        # Feedforward layers
        self.FFconv = nn.ModuleList([FFconv2d(self.ics[i], self.ocs[i], downsample=self.sps[i]) for i in
                                     range(1, self.nlays)])  # (None, FF0, FF1, ..., FF10)
        # Feedback layers
        # if cls > 0:
        #     self.FBconv = nn.ModuleList([FBconv2d(self.ocs[i], self.ics[i], upsample=self.sps[i]) for i in range(1, self.nlays)]) # (None, FB0, FB1, ..., FB10)
        self.FBconv = nn.ModuleList([FBconv2d(self.ocs[i], self.ics[i], upsample=self.sps[i]) for i in
                                     range(1, self.nlays)])  # (None, FB0, FB1, ..., FB10)

        # Update rate
        self.a0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ics[i], 1, 1) + 0.5) for i in
                                    range(2, self.nlays)])  # (None, None, a0_0, ..., a0_9)
        self.b0 = nn.ParameterList([nn.Parameter(torch.zeros(1, self.ocs[i], 1, 1) + 1.0) for i in
                                    range(1, self.nlays)])  # (None, b0_0, b0_1, ..., b0-10)

        # Group Normalization with group = 1
        self.GN = nn.ModuleList(
            [nn.GroupNorm(1, self.ocs[i]) for i in range(1, self.nlays)])  # (None, GN0, GN1, ..., GN10)

        # Linear layer
        self.linear = nn.Linear(self.ocs[-1], num_classes)

    def forward(self, x):

        # x_ff, x_fb = [], []
        x_ff, x_fb, x_pred, x_err = [], [], [], []
        x_ff_beforeGN, x_fb_beforeGN = [], []
        x = self.baseconv(x)

        # Feedforward
        xr = [F.relu(self.FFconv[0](x))]
        xr_beforeGN = [F.relu(self.FFconv[0](x))]
        xr[0] = self.GN[0](xr[0])
        # for i in range(1,self.nlays):
        for i in range(1, self.nlays - 1):
            xr.append(F.relu(self.FFconv[i](xr[i - 1])))
            xr_beforeGN.append(F.relu(self.FFconv[i](xr[i - 1])))
            xr[i] = self.GN[i](xr[i])

        # Save feedforward
        x_ff_cycle = [x]
        x_ff_cycle.extend(xr)
        x_ff.append(x_ff_cycle[:])  # save feedforward

        x_ff_beforeGN_cycle = [x]
        x_ff_beforeGN_cycle.extend(xr_beforeGN)
        x_ff_beforeGN.append(x_ff_beforeGN_cycle[:])  # save feedforward

        # Dynamic process - version 1.
        for t in range(self.cls):

            # Feedback prediction
            xp = []
            # for i in range(self.nlays-1,0,-1):
            for i in range(self.nlays - 2, 0, -1):
                xp = [self.FBconv[i](xr[i])] + xp
                a0 = F.relu(self.a0[i - 1]).expand_as(xr[i - 1])
                xr[i - 1] = F.relu(xp[0] * a0 + xr[i - 1] * (1 - a0))
                xr_beforeGN[i - 1] = F.relu(xp[0] * a0 + xr[i - 1] * (1 - a0))
                xr[i - 1] = self.GN[i - 1](xr[i - 1])

            x_pred_cycle = [self.FBconv[0](xr[0])]
            x_pred_cycle.extend(xp)
            x_pred.append(x_pred_cycle[:])  # save prediction

            x_fb_cycle = []
            x_fb_cycle.extend(xr)
            x_fb.append(x_fb_cycle[:])  # save feedback

            x_fb_beforeGN_cycle = []
            x_fb_beforeGN_cycle.extend(xr_beforeGN)
            x_fb_beforeGN.append(x_fb_beforeGN_cycle[:])  # save feedback without GN

            # Feedforward prediction error
            xe = []
            b0 = F.relu(self.b0[0]).expand_as(xr[0])
            xe.append(x - self.FBconv[0](xr[0]))
            xr[0] = F.relu(self.FFconv[0](x - self.FBconv[0](xr[0])) * b0 + xr[0])
            xr_beforeGN[0] = F.relu(self.FFconv[0](x - self.FBconv[0](xr[0])) * b0 + xr[0])
            xr[0] = self.GN[0](xr[0])
            # for i in range(1, self.nlays):
            for i in range(1, self.nlays - 1):
                b0 = F.relu(self.b0[i]).expand_as(xr[i])
                xe.append(xr[i - 1] - xp[i - 1])
                xr[i] = F.relu(self.FFconv[i](xr[i - 1] - xp[i - 1]) * b0 + xr[i])
                xr_beforeGN[i] = F.relu(self.FFconv[i](xr[i - 1] - xp[i - 1]) * b0 + xr[i])
                xr[i] = self.GN[i](xr[i])

            # Save feedforward
            x_ff_cycle = [x]
            x_ff_cycle.extend(xr)
            x_ff.append(x_ff_cycle[:])  # save feedforward

            x_ff_beforeGN_cycle = [x]
            x_ff_beforeGN_cycle.extend(xr_beforeGN)
            x_ff_beforeGN.append(x_ff_beforeGN_cycle[:])  # save feedforward

            x_err_cycle = []
            x_err_cycle.extend(xe)
            x_err.append(x_err_cycle[:])
        # classifier
        out = F.avg_pool2d(xr[-1], xr[-1].size(-1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        # return out, (x_ff, x_fb)
        return out, (x_ff, x_fb, x_pred, x_err, x_ff_beforeGN, x_fb_beforeGN)

