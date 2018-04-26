import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CrowdNet(nn.Module):
    def __init__(self, pretrained=False):
        super(CrowdNet, self).__init__()

        vgg16 = models.vgg16(pretrained=pretrained)
        self.deep_network = nn.Sequential(
            *[
                *list(vgg16.features.children())[:23], # to relu4_3
                *[nn.MaxPool2d(2, stride=1)], # new pooling on 4th layer - kernel 2x2 with stride 1
                *list(vgg16.features.children())[24:-1] # from relu5_1 to relu5_3
            ]
        )
        
        self.shallow_network = nn.Sequential(
            nn.Conv2d(3, 24, 5, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2),
            nn.Conv2d(24, 24, 5, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2),
            nn.Conv2d(24, 24, 5, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2)
        )
        
        self.conv1d = nn.Conv2d(536, 1, 1)

    def forward(self, x):
        do = self.deep_network(x)
        so = self.shallow_network(x)
        co = torch.cat((do, so), 1)
        out = self.conv1d(co)
        return out