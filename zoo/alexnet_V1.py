import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
import torchvision.transforms as transforms

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Identity(nn.Module):
    def forward(self, x):
        return x


class LGN(nn.Module):
    
    """
    This is designed to model foveal and peripheral vision separately.
    In order to be treated like a CogBlock, it accepts maxpooling indices in 
    the forward pass, and returns feedback signals in the backward pass, 
    but these are ignored.
    """

    def __init__(self, channels=[3,64], kernel_size=11, image_size=224, num_scales=5,
                 max_scale=2, min_scale=1):

        super().__init__()

        self.in_channels = ic = channels[0]
        self.out_channels = oc = channels[1]
        self.num_scales = num_scales
        self.max_scale = max_scale

        # integration convolutions
        self.fl = nn.Conv2d(ic * 2, ic, kernel_size=1, bias=False)
        self.fb = nn.Conv2d(ic * 2, ic, kernel_size=1, bias=False)

        # feed forward convolutions at various eccentricities
        scales = torch.linspace(max_scale, min_scale, num_scales)
        self.xforms = nn.ModuleList([transforms.Resize(
                int(image_size // scale), antialias=True) for scale in scales])
        self.conv = nn.Conv2d(ic, oc, kernel_size=kernel_size, 
                              padding=kernel_size // 2, bias=False)
        self.resize = transforms.Resize(image_size, antialias=True)
        self.windows = torch.linspace(0, image_size / 2, num_scales+1,
                                      dtype=torch.int)[:-1]
        self.norm = nn.GroupNorm(16, oc)
        self.nonlin = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=8, stride=8, return_indices=False)
        
        # allow for retrieval with forward hook
        self.f = Identity()


    def forward(self, inputs):


        # feed forward
        for transform, w in zip(self.xforms, self.windows):
            temp = transform(inputs)  # shrink image depending on eccentricity
            temp = self.conv(temp)  # apply convolution
            temp = self.resize(temp)  # grow back to original size
            if w == 0:
                f_out = temp
            else:
                f_out[..., w:-w, w:-w] = temp[..., w:-w, w:-w]

        # final nonlin and maxpool
        f = self.nonlin(f_out)
        f = self.pool(f)
        
        # allow for retrieval with forward hook
        f = self.f(f)
        
        return f


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            LGN(),
            #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            #nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
