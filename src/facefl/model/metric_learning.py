import math
from logging import WARNING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from flwr.common.logger import log
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

try:
    from torchvision.models.resnet import ResNet18_Weights
except ImportError:
    log(
        WARNING,
        "Import ResNet18_Weights failed, since torchvision version is %s,",
        torchvision.__version__,
    )
    log(WARNING, "If you have some problems, upgrade torchvision>=0.13.0")
    pass
try:
    from torchvision.utils import _log_api_usage_once
except ImportError:
    log(
        WARNING,
        "Import _log_api_usage_once failed, since torchvision version is %s,",
        torchvision.__version__,
    )
    log(WARNING, "If you have some problems, upgrade torchvision>=0.13.0")
    pass
from typing import Any, Callable, List, Optional, Type, Union

from facefl.model.base_model import Net


class ArcFaceResNet(Net):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.arcmarginprod = ArcMarginProduct(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)
        self.features = x
        x = self.arcmarginprod(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class ArcFaceResNetLR(Net):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 2)
        self.arcmarginprod = ArcMarginProduct(2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        self.features = x
        x = self.arcmarginprod(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def get_arcface_resnet18(
    input_spec,
    num_classes: int,
    pretrained: str = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
) -> ArcFaceResNet:
    if input_spec[1] >= 112:
        model = ArcFaceResNet(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            norm_layer=norm_layer,
            **kwargs,
        )
        if pretrained == "IMAGENET1K_V1":
            model.load_state_dict(
                ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True),
                strict=False,
            )
        elif pretrained == "CelebA":
            if norm_layer is not None:
                pretrained_weight = torch.load("./models/GNResNet18.pth")
            else:
                pretrained_weight = torch.load("./models/ResNet18.pth")
            # pretrained_weight['arcmarginprod.weight'] = ArcMarginProduct(512,num_classes)
            pretrained_weight["arcmarginprod.weight"] = torch.FloatTensor(
                num_classes, 512
            )
            nn.init.xavier_normal_(pretrained_weight["arcmarginprod.weight"])
            model.load_state_dict(pretrained_weight, strict=False)
    else:
        model = ArcFaceResNetLR(
            BasicBlock,
            [2, 2, 2, 2],
            num_classes=num_classes,
            norm_layer=norm_layer,
            **kwargs,
        )
    return model


# Copied and modified from
# https://github.com/pytorch/vision/issues/2391


def batchnorm_to_groupnorm(net: Net):
    def get_layer(model, name):
        layer = model
        for attr in name.split("."):
            layer = getattr(layer, attr)
        return layer

    def set_layer(model, name, layer):
        try:
            attrs, name = name.rsplit(".", 1)
            model = get_layer(model, attrs)
        except ValueError:
            pass
        setattr(model, name, layer)

    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn = get_layer(net, name)
            gn = nn.GroupNorm(2, bn.num_features)
            print("Swapping {} with {}".format(bn, gn))

            set_layer(net, name, gn)


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)

    Copied and modified from:
        https://github.com/ChristofHenkel/kaggle-landmark-2021-1st-place/blob/034a7d8665bb4696981698348c9370f2d4e61e35/models/ch_mdl_dolg_efficientnet.py
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
    ):
        super(ArcMarginProduct, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_dims, in_dims))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine


class ArcFaceLoss(nn.modules.Module):
    def __init__(
        self, s: float = 45.0, m: float = 0.1, weight=None, reduction="mean"
    ) -> None:
        super().__init__()

        self.weight = weight
        self.reduction = reduction

        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.th = math.cos(math.pi - m)
        self.mm = math.cos(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clip(1e-16, 1))

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        onehot = torch.zeros_like(cosine)
        onehot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (onehot * phi) + ((1.0 - onehot) * cosine)

        s = self.s
        output = output * s

        loss = self.criterion(output, labels)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class CosineContrastiveLoss(nn.modules.Module):
    def __init__(self, nu: float = 0.9) -> None:
        super().__init__()
        self.nu = nu

    def forward(self, x, labels):
        loss = torch.pow(torch.max(torch.tensor(0.0), self.nu - x), 2).mean()
        return loss


class SpreadoutRegularizer(nn.modules.Module):
    def __init__(self, nu: float = 0.9) -> None:
        super().__init__()
        self.nu = nu

    def forward(self, w, out_dims):
        w = F.normalize(w)
        loss = 0.0
        for i in range(out_dims - 1):
            for j in range(i + 1, out_dims):
                loss += torch.pow(
                    torch.max(
                        torch.tensor(0.0), self.nu - 1 + torch.dot(w[i, :], w[j, :])
                    ),
                    2,
                )
        return loss
