import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import Scalar
from torch.utils.data import Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from .federated_dataset import (
    CIFAR10_truncated,
    FederatedCelebaVerification,
    FederatedUsbcamVerification,
)
from .centralized_dataset import (
    CentralizedCelebaAndUsbcamVerification,
    CentralizedCelebaVerification,
)

DATA_ROOT = Path(os.environ["DATA_ROOT"])


def load_centralized_dataset(
    dataset_name: str, train: bool = True, target: str = None, download: bool = False
) -> Dataset:
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        root = DATA_ROOT / "CIFAR10" / "raw"
        dataset = CIFAR10(
            root=root, train=train, transform=transform, download=download
        )
    elif dataset_name == "CelebA":
        assert target is not None
        if target == "mix_usbcam":
            dataset = CentralizedCelebaAndUsbcamVerification()
        else:
            dataset = CentralizedCelebaVerification(train=train, target=target)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
    return dataset


def load_federated_dataset(
    dataset_name: str,
    id: str,
    train: bool = True,
    target: str = None,
    download: bool = False,
) -> Dataset:
    if dataset_name == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        dataset = CIFAR10_truncated(
            root=DATA_ROOT,
            id=id,
            train=train,
            target=target,
            transform=transform,
            download=download,
        )
    elif dataset_name == "CelebA":
        assert target is not None
        dataset = FederatedCelebaVerification(id=id, train=train, target=target)
    elif dataset_name == "usbcam":
        dataset = FederatedUsbcamVerification(id=id, train=train)
    else:
        raise NotImplementedError(f"{dataset_name} is not supported")
    return dataset


def configure_dataset(dataset_name: str, target: str = None) -> Dict[str, Scalar]:
    if dataset_name == "CIFAR10":
        input_spec = (3, 32, 32)
        out_dims = 10
    elif (dataset_name == "CelebA") or (dataset_name == "usbcam"):
        input_spec = (3, 112, 112)
        if (target == "small") or (target == "mix_usbcam"):
            out_dims = 10
        elif target == "medium":
            out_dims = 100
        elif target == "large":
            out_dims = 1000
    else:
        raise NotImplementedError(f"{dataset_name} is not implemented")
    config = {"input_spec": input_spec, "out_dims": out_dims}
    return config


def split_validation(dataset: Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
    num_samples = dataset.__len__()
    num_train = int(num_samples * split_ratio)
    num_val = num_samples - num_train
    trainset, valset = random_split(dataset, [num_train, num_val])
    return trainset, valset


def write_json(
    dataset: str, target: str, json_data: Dict[str, List[np.ndarray]], train: bool
):
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
