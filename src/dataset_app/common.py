import json
import os
import random
import torch
import numpy as np
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import transforms

from typing import List, Dict, Tuple
from pathlib import Path

from .datasets import DATA_ROOT, CIFAR10_truncated
DATA_ROOT = Path(os.environ['DATA_ROOT'])

def load_fmnist():
    transform = transforms.Compose([transforms.ToTensor()])

    traindata = FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    testdata = FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.targets
    X_test, y_test = testdata.data, testdata.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    return (X_train, y_train, X_test, y_test)

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor()])

    traindata = MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    testdata = MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.targets
    X_test, y_test = testdata.data, testdata.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()
    return (X_train, y_train, X_test, y_test)

def load_cifar10():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])

    traindata = CIFAR10_truncated(root=DATA_ROOT,train=True, download=True, transform=transform)
    testdata = CIFAR10_truncated(root=DATA_ROOT,train=False, download=True, transform=transform)

    X_train, y_train = traindata.data, traindata.target
    X_test, y_test = testdata.data, testdata.target
    return (X_train, y_train, X_test, y_test)

def create_iid(
    labels: np.ndarray,
    num_parties: int,
    classes: List[int] = None,
    list_labels_idxes: Dict[int, List[int]] = None):
    if labels.shape[0] % num_parties:
        raise ValueError("Imbalanced classes are not allowed")

    if classes is None and list_labels_idxes is None:
        print("creating label_idxes ...")
        classes = list(np.unique(labels))
        list_labels_idxes = {k: np.where(labels == k)[0].tolist() for k in classes}
    elif classes is None or list_labels_idxes is None:
        raise ValueError("Invalid Argument Error")
    else:
        classes = classes
        list_labels_idxes = list_labels_idxes
    
    net_dataidx_map = {i: [] for i in range(num_parties)}
    id = 0
    for k in classes:
        while(len(list_labels_idxes[k]) > 0):
            label_idx = list_labels_idxes[k].pop()
            net_dataidx_map[id % num_parties].append(label_idx)
            id += 1
    record_net_data_stats(labels,net_dataidx_map)
    return net_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    print(str(net_cls_counts))
    return net_cls_counts

def write_json(json_data: Dict[str, List[np.ndarray]], save_dir: str, file_name: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    file_path = Path(save_dir) / f"{file_name}.json"
    print("writing {}".format(file_name))
    with open(file_path, "w") as outfile:
        json.dump(json_data, outfile)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    dataset = "FashionMNIST"
    X_train, y_train, X_test, y_test = load_cifar10()
    set_seed(1234)
    train_json = create_iid(
        labels=y_train,
        num_parties=1000,
    )
    test_json = create_iid(
        labels=y_test,
        num_parties=1000
    )
    save_dir = "./data/CIFAR10/partitions/iid"
    write_json(train_json, save_dir=save_dir, file_name="train")
    write_json(test_json, save_dir=save_dir, file_name="test")