import numpy as np
from torchvision.transforms import transforms
from pathlib import Path
from typing import List

from dataset_app.datasets import (
    FashionMNIST_truncated,
    CIFAR10_truncated,
)
DATA_ROOT = Path("./data")

def load_dataset(name: str, train: bool = True, dataidxs: List[np.ndarray]=None, target: str = None):
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
        root = DATA_ROOT / "CIFAR10" / "raw"
        dataset = CIFAR10_truncated(
            root=root,
            dataidxs=dataidxs,
            train=train,
            transform=transform,
        )
    else:
        raise NotImplementedError(f"Dataset {name} is not Implemented")
    return dataset