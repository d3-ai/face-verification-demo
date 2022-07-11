import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .base_model import Net

class EfficientNetB0(Net):
    def __init__(self, out_dims, pretrained: bool = False) -> None:
        super(EfficientNetB0, self).__init__()
        if pretrained:
            self.resnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.resnet.fc = nn.Linear(512, 10)
        else:
            self.resnet = efficientnet_b0(norm_layer = lambda x: nn.GroupNorm(2,x), num_classes=out_dims)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.resnet(x)
