import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.datasets import (
    FashionMNIST,
    CIFAR10,
)

from PIL import Image

DATA_ROOT = Path("./data")

"""The followings are borrowed from NIID-Bench"""
class FashionMNIST_truncated(Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        mnist_dataobj = FashionMNIST(self.root, self.train, self.transform, self.target_transform, self.download)
        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR10_truncated(Dataset):
    def __init__(self, id:str = None, json_path: str = None,  train=True, transform=None, target_transform=None, download=False):
        self.id = id
        self.json_path = json_path
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        
        self.root = DATA_ROOT / "CIFAR10" / "raw"
        if self.json_path is not None:
            if train:
                self.json_path = Path(json_path) / "train_data.json"
            else:
                self.json_path = Path(json_path) / "test_data.json"

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)

        if self.json_path is not None:
            with open(self.json_path, 'r') as f:
                json_data = json.load(f)
            data_idx = json_data[self.id]
            data = data[data_idx]
            target = target[data_idx]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class CelebaDataset(Dataset):
    def __init__(self, id: str, target: str, train: bool = True, transform=None) -> None:
        self.root = Path("./data/celeba")
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(160),
                transforms.Resize((64,64)),
                transforms.ToTensor()
            ])

        if train:
            self.json_path = self.root / "attrs" / target / "train_data.json"
        else:
            self.json_path = self.root / "attrs" / target / "test_data.json"
        with open(self.json_path, 'r') as f:
            self.json_data = json.load(f)

        if train:
            self.id = int(id)
            self.num_samples = self.json_data['num_samples'][self.id]
            user = self.json_data['users'][self.id]
            self.data = self.json_data['user_data'][user]
        else:
            self.data = {'x': [], 'y': []}
            for _, data in self.json_data['user_data'].items():
                self.data['x'].extend(data['x'])
                self.data['y'].extend(data['y'])

            self.num_samples = len(self.data['x'])
    
    def __getitem__(self, index):
        img_path = self.root / "raw" / "img_align_celeba" / self.data['x'][index]
        img = Image.open(img_path)
        target = self.data['y'][index]
        # target = one_hot(torch.tensor(self.data['y'][index]), num_classes=2).to(torch.float32)
        # target = torch.eye(2)[self.data['y'][index]]

        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return self.num_samples
