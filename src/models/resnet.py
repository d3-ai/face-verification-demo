import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from .base_model import Net

class ResNet18(Net):
    def __init__(self, out_dims, pretrained: bool = False) -> None:
        super(ResNet18, self).__init__()
        if pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.resnet.fc = nn.Linear(512, 10)
        else:
            self.resnet = resnet18(norm_layer = lambda x: nn.GroupNorm(2,x), num_classes=out_dims)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.resnet(x)
