import argparse
import warnings

import random
import numpy as np
import torch

from client_app.client import Client
from client_app.face_client import FlowerFaceClient
from client_app.app import start_client
from utils.utils_dataset import configure_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Client")
parser.add_argument("--server_address", type=str, required=True, default="0.0.0.0:8080", help="server ipaddress:post")
parser.add_argument("--cid", type=str, required=True, help="Client id for data partitioning.")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA", "usbcam"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--target", type=str, required=True, help="FL config: target partitions for common dataset target attributes for celeba")
parser.add_argument("--model", type=str, required=False, choices=["tinyCNN", "ResNet18", "GNResNet18"], default="tinyCNN", help="model name for FL training")
parser.add_argument("--pretrained", type=str, required=False, choices=["IMAGENET1K_V1", "None"], default="None", help="pretraing recipe")
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
    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "model_name": args.model,
        "pretrained": args.pretrained
    }
    client: Client = FlowerFaceClient(cid=args.cid, config=config)
    start_client(server_address=args.server_address, client=client)

if __name__ == "__main__":
    main()