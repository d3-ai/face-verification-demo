from typing import Tuple

from models.base_model import Net
from models.efficientnet import EfficientNetB0
from models.resnet import ResNet18
from models.tiny_CNN import tinyCNN

def load_model(name: str, input_spec: Tuple[int, int, int], out_dims: int = 10, pretrained: bool = False)->Net:
    if name == "tiny_CNN":
        return tinyCNN(input_spec=input_spec, out_dims=out_dims)
    elif name == "ResNet18":
        return ResNet18(pretrained=pretrained, out_dims=out_dims)
    elif name == "EfficientNetB0":
        return EfficientNetB0(pretrained=pretrained, out_dims=out_dims)
    else:
        raise NotImplementedError(f"model {name} is not implemented.")