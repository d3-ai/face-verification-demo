import argparse
import warnings

import random
import numpy as np
import torch

from dataset_app.common import create_iid, load_cifar10
from utils.utils_dataset import write_json

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Dataset partitioning for federated settings.")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--target", type=str, required=True, help="FL config: target partitions for common dataset target attributes for celeba")
parser.add_argument("--num_clients", type=int, required=False, default=4, help="FL config: number of clients")
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
    dataset = args.dataset
    target = args.target
    num_clients = args.num_clients
    
    if dataset == "CIFAR10":
        _, y_train, _, y_test = load_cifar10()
    else:
        raise NotImplementedError(f"{dataset} is not implemented.")


    if target == "iid":
        train_json = create_iid(
            labels=y_train,
            num_parties=num_clients,
        )
        test_json = create_iid(
            labels=y_test,
            num_parties=num_clients
        )
    else:
        raise NotImplementedError(f"Partitioning {target} is not implemented.")
    
    write_json(dataset=dataset, target=target, json_data=train_json, train=True)
    write_json(dataset=dataset, target=target, json_data=test_json, train=False)
    
if __name__ == "__main__":
    main()