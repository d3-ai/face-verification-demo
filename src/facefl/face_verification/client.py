import argparse
import random
import warnings

import numpy as np
import torch
from flwr.client import start_client

from facefl.client.face_client import FlowerFaceClient
from facefl.utils.utils_dataset import configure_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower Client")
parser.add_argument(
    "--server_address",
    type=str,
    required=True,
    default="0.0.0.0:8080",
    help="server ipaddress:post",
)
parser.add_argument(
    "--strategy",
    type=str,
    required=True,
    choices=["FedAvg", "FedAwS"],
    default="FedAvg",
    help="FL config: number of clients",
)
parser.add_argument(
    "--cid", type=str, required=True, help="Client id for data partitioning."
)
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=["CIFAR10", "CelebA", "usbcam"],
    default="CIFAR10",
    help="dataset name for FL training",
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    default="tinyCNN",
    help="model name for FL training",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


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
    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    config.update(dataset_config)
    if args.strategy == "FedAvg":
        config["out_dims"] = 10
    elif args.strategy == "FedAwS":
        config["out_dims"] = 1
    else:
        raise NotImplementedError(f"{args.strategy} is not implemented.")
    client = FlowerFaceClient(cid=args.cid, config=config)
    start_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
