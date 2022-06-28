import argparse

import random
import numpy as np
import torch

from pathlib import Path

DATA_ROOT=Path("./data")

parser = argparse.ArgumentParser("Dataset partitioning")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "FashionMNIST"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

if __name__ == "__main__":
    main()