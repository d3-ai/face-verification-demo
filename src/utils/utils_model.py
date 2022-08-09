import torch.nn as nn

from models.base_model import Net
from models.efficientnet import EfficientNetB0
from models.resnet import resnet18 
from models.tinycnn import tinyCNN

from typing import Tuple

def load_model(name: str, input_spec: Tuple[int, int, int], out_dims: int = 10, pretrained: bool = False)->Net:
    if name == "tinyCNN":
        return tinyCNN(input_spec=input_spec, out_dims=out_dims)
    elif name == "ResNet18":
        return resnet18(input_spec=input_spec, num_classes=out_dims)
    elif name == "GNResNet18":
        return resnet18(input_spec=input_spec, num_classes=out_dims, norm_layer=lambda x: nn.GroupNorm(2,x))
    elif name == "EfficientNetB0":
        return EfficientNetB0(pretrained=pretrained, out_dims=out_dims)
    else:
        raise NotImplementedError(f"model {name} is not implemented.")