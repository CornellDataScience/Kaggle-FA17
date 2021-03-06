import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ResNetTransfer(nn.Module):
    def __init__(self, num_classes, dropRate=0.0):
        super(ResNetTransfer, self).__init__()

        self.resnet = torchvision.models.densenet(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
