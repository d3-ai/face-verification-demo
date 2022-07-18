import os
import json
from argparse import ArgumentError
import numpy as np
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms
from pathlib import Path
from typing import List, Dict, Tuple
from common.typing import Scalar

from dataset_app.datasets import (
    CelebaDataset,
    FashionMNIST_truncated,
    CIFAR10_truncated,
)

def load_dataset(
    name: str,
    id: str = None,
    train: bool = True,
    dataidxs: List[np.ndarray]=None,
    target: str = None,
    download: bool=False):
    DATA_ROOT = Path(os.environ["DATA_ROOT"])
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
                root=DATA_ROOT,
                id=id,
                json_path=json_path,
                train=train,
                transform=transform,
                download=download)
        else:
            dataset = CIFAR10_truncated(
                root=DATA_ROOT,
                train=train,
                transform=transform,
                download=download)
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

def configure_dataset(dataset_name: str)->Dict[str, Scalar]:
    if dataset_name == "CIFAR10":
        input_spec = (3,32,32)
        out_dims = 10
    elif dataset_name == "CelebA":
        input_spec = (3,224,224)
        out_dims =1
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    config = {
        "input_spec" : input_spec,
        "out_dims": out_dims
    }
    return config

def split_validation(dataset: Dataset, split_ratio: float)->Tuple[Dataset, Dataset]:
    num_samples = dataset.__len__()
    num_train = int(num_samples*split_ratio)
    num_val = num_samples - num_train
    trainset, valset = random_split(dataset, [num_train, num_val])
    return trainset, valset

def write_json(dataset: str, target: str, json_data: Dict[str, List[np.ndarray]], train: bool):
    DATA_ROOT = Path(os.environ["DATA_ROOT"])
    if dataset == "CIFAR10":
        save_dir = DATA_ROOT / "CIFAR10" / "partitions" / target
    elif dataset == "CelebA":
        save_dir = DATA_ROOT / "celeba" / "attrs" / target

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if train:
        file_path = save_dir / "train_data.json"
    else:
        file_path = save_dir / "test_data.json"
    
    print("writing {}".format(file_path))
    with open(file_path, "w") as outfile:
        json.dump(json_data, outfile)