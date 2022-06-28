from argparse import ArgumentError
import numpy as np
from torchvision.transforms import transforms
from pathlib import Path
from typing import List

from dataset_app.datasets import (
    CelebaDataset,
    FashionMNIST_truncated,
    CIFAR10_truncated,
)
DATA_ROOT = Path("./data")

def load_dataset(name: str, id: str = None, train: bool = True, dataidxs: List[np.ndarray]=None, target: str = None):
    if name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
        dataset = FashionMNIST_truncated(
            root=DATA_ROOT,
            dataidxs=dataidxs,
            train=train,
            transform=transform,
            download=True) 
    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
        if id is not None:
            json_path = DATA_ROOT / "CIFAR10" / "partitions" / target 
            dataset = CIFAR10_truncated(
                id=id,
                json_path=json_path,
                train=train,
                transform=transform,)
        else:
            dataset = CIFAR10_truncated(
                train=train,
                transform=transform,)
    elif name == "CelebA":
        if target is not None:
            dataset = CelebaDataset(
                id = id,
                target=target,
                train=train,
            )
        else:
            raise ArgumentError(f"CelebA should be specified by target attributes")
    else:
        raise NotImplementedError(f"Dataset {name} is not Implemented")
    return dataset