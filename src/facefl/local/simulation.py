import argparse
import gc
import os
import random
import warnings
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from flwr.client import Client
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from torch.utils.data import DataLoader

from facefl.client.base_client import FlowerRayClient
from facefl.dataset import configure_dataset, load_centralized_dataset
from facefl.model.base_model import Net
from facefl.model.driver import test
from facefl.server.wandb_server import WandbServer
from facefl.simulation.app import start_simulation
from facefl.utils.utils_model import load_model
from facefl.utils.utils_wandb import custom_wandb_init

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower simulation")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["CIFAR10", "CelebA"],
    help="FL config: dataset name",
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
    required=True,
    choices=["tinyCNN", "ResNet18", "GNResNet18"],
    help="FL config: model name",
)
parser.add_argument(
    "--num_rounds",
    type=int,
    required=False,
    default=5,
    help="FL config: aggregation rounds",
)
parser.add_argument(
    "--num_clients",
    type=int,
    required=False,
    default=4,
    help="FL config: number of clients",
)
parser.add_argument(
    "--fraction_fit",
    type=float,
    required=False,
    default=1,
    help="FL config: client selection ratio",
)
parser.add_argument(
    "--local_epochs",
    type=int,
    required=False,
    default=5,
    help="Client fit config: local epochs",
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=False,
    default=10,
    help="Client fit config: batchsize",
)
parser.add_argument(
    "--lr",
    type=float,
    required=False,
    default=0.01,
    help="Client fit config: learning rate",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    required=False,
    default=0.0,
    help="Client fit config: weigh_decay",
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)

    net: Net = load_model(
        name=args.model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())

    client_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "model_name": args.model,
    }
    server_config = ServerConfig(num_rounds=args.num_rounds)

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        return config

    def eval_config(server_round: int) -> Dict[str, Scalar]:
        config = {
            "batch_size": args.batch_size,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(
            dataset_name=dataset, train=False, target=target
        )
        testloader = DataLoader(testset, batch_size=1000)

        def evaluate(
            server_round: int, weights: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(weights)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            results = test(model, testloader, device=device)
            torch.cuda.empty_cache()
            gc.collect()
            return results["loss"], {"accuracy": results["acc"]}

        return evaluate

    def client_fn(cid: str) -> Client:
        return FlowerRayClient(cid, client_config)

    strategy = FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=1,
        min_fit_clients=int(args.num_clients * args.fraction_fit),
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        evaluate_fn=get_eval_fn(net, args.dataset, args.target),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        initial_parameters=init_parameters,
    )
    server = RayTuneServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
    )
    client_resources = {"num_cpus": 2, "num_gpus": 1.0}
    ray_config = {"include_dashboard": False, "address": "auto"}
    params_config = {
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_name": args.model,
        "seed": args.seed,
        "num_clients": args.num_clients,
        "fraction_fit": args.fraction_fit,
        "api_key_file": os.environ["WANDB_API_KEY_FILE"],
    }
    custom_wandb_init(config=params_config, project="hoge_baselines", strategy="FedAvg")
    _, _ = start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        server=server,
        config=server_config,
        ray_init_args=ray_config,
        keep_initialised=True,
    )


if __name__ == "__main__":
    main()
