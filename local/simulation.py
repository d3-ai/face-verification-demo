import argparse
import uuid
import warnings
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr.server.strategy import FedAvg
from server_app.app import ServerConfig

from driver import test
from models.base_model import Net
from client_app.client import Client
from client_app.base_client import FlowerClient
from server_app.client_manager import SimpleClientManager
from server_app.raytuning_server import RayTuneServer
from utils.utils_model import load_model
from utils.utils_dataset import configure_dataset, load_dataset
from common import ndarrays_to_parameters
from common import Parameters, Scalar, NDArrays
from simulation_app.app import start_simulation
from typing import Callable, Dict, Optional, Tuple

import wandb

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower simulation")
parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "CelebA"], help="FL config: dataset name")
parser.add_argument("--target", type=str, required=True, help="FL config: target partitions for common dataset target attributes for celeba")
parser.add_argument("--model", type=str, required=True, choices=["tinyCNN", "ResNet18"], help="FL config: model name")
parser.add_argument("--num_rounds", type=int, required=False, default=5, help="FL config: aggregation rounds")
parser.add_argument("--num_clients", type=int, required=False, default=4, help="FL config: number of clients")
parser.add_argument("--local_epochs", type=int, required=False, default=5, help="Client fit config: local epochs")
parser.add_argument("--batch_size", type=int, required=False, default=10, help="Client fit config: batchsize")
parser.add_argument("--lr", type=float, required=False, default=0.01, help="Client fit config: learning rate")
parser.add_argument("--weight_decay", type=float, required=False, default=0.0, help="Client fit config: weigh_decay")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    dataset_config = configure_dataset(dataset_name=args.dataset)

    net: Net = load_model(name=args.model, input_spec=dataset_config["input_spec"], out_dims=dataset_config["out_dims"])
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())

    client_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "model_name": args.model
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
    
    def eval_config(server_round: int)->Dict[str,Scalar]:
        config = {
            "batch_size": args.batch_size,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str)-> Callable:
        testset = load_dataset(name=dataset, train=False, target=target)
        testloader = DataLoader(testset, batch_size=10)
        def evaluate(server_round: int, weights: NDArrays, config: Dict[str, Scalar])-> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(weights)
            results = test(model, testloader)
            return results['loss'], {"accuracy": results['acc']}
        return evaluate

    def client_fn(cid: str)->Client:
        return FlowerClient(cid, client_config)

    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=args.num_clients,
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
    client_resources = {"num_cpus": 1, "num_gpus": 0.1}
    ray_config = {"include_dashboard": False, "address": "auto"}
    params_config = {
        "batch_size": args.batch_size,
        "local_epochs": args.local_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "model_name": args.model,
    }
    trial_id = str(uuid.uuid1().hex)[:8]
    wandb.init(
        dir=f"/home/yamasaki/PythonProject/project/FedAvg_{args.dataset}",
        project="test",
        config=params_config,
        group=f"FedAvg_{args.model}",
        name=f"fedavg_{trial_id}"
    )
    hist = start_simulation(
        client_fn = client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        server=server,
        config=server_config,
        ray_init_args=ray_config,
        keep_initialised=True,
    )

if __name__ == "__main__":
    main()