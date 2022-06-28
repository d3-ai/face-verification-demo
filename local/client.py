import argparse
import warnings

import random
import numpy as np
import torch

import flwr as fl
from flwr.client.client import Client

from client_app.base_client import FlowerClient

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Client")
parser.add_argument("--cid", type=str, required=True, help="Client id for data partitioning.")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--target", type=str, required=True, help="FL config: target partitions for common dataset target attributes for celeba")
parser.add_argument("--model", type=str, required=False, choices=["tiny_CNN", "ResNet18"], default="tiny_CNN", help="model name for FL training")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    # Parse command line argument `partition`
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "model_name": args.model, 
    }
    client: Client = FlowerClient(cid=args.cid, config=config)
    fl.client.start_client("0.0.0.0:8080", client=client)

if __name__ == "__main__":
    main()