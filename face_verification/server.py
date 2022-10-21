import argparse
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from flwr.common import NDArray, NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig, start_server
from flwr.server.client_manager import SimpleClientManager
from models.base_model import Net
from models.driver import test
from server_app.custom_server import CustomServer
from torch.utils.data import DataLoader
from utils.utils_dataset import configure_dataset, load_centralized_dataset
from utils.utils_model import load_arcface_model
from utils.utils_server import load_strategy

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Flower face verification server")
parser.add_argument("--server_address", type=str, required=True, default="0.0.0.0:8080", help="server ipaddress:post")
parser.add_argument(
    "--dataset", type=str, required=True, choices=["CIFAR10", "CelebA"], help="FL config: dataset name"
)
parser.add_argument(
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--model", type=str, required=True, choices=["tinyCNN", "GNResNet18"], help="FL config: model name"
)
parser.add_argument(
    "--pretrained",
    type=str,
    required=False,
    choices=["IMAGENET1K_V1", "CelebA", "None"],
    default="None",
    help="pretraing recipe",
)
parser.add_argument("--num_rounds", type=int, required=False, default=5, help="FL config: aggregation rounds")
parser.add_argument("--num_clients", type=int, required=False, default=4, help="FL config: number of clients")
parser.add_argument("--fraction_fit", type=float, required=False, default=1, help="FL config: client selection ratio")
parser.add_argument(
    "--strategy",
    type=str,
    required=False,
    choices=["FedAvg", "FedAwS"],
    default="FedAvg",
    help="FL config: number of clients",
)
parser.add_argument("--local_epochs", type=int, required=False, default=5, help="Client fit config: local epochs")
parser.add_argument("--batch_size", type=int, required=False, default=10, help="Client fit config: batchsize")
parser.add_argument(
    "--criterion",
    type=str,
    required=False,
    default="ArcFace",
    choices=["ArcFace", "CCL"],
    help="Criterion of classification performance",
)
parser.add_argument("--lr", type=float, required=False, default=0.01, help="Client fit config: learning rate")
parser.add_argument("--weight_decay", type=float, required=False, default=0.0, help="Client fit config: weigh_decay")
parser.add_argument("--scale", type=float, required=False, default=0.0, help="scale for arcface loss")
parser.add_argument("--margin", type=float, required=False, default=0.0, help="margin for arcface loss")
parser.add_argument("--nu", type=float, required=False, default=0.9, help="margin for cosine contrastive loss")
parser.add_argument("--lam", type=float, required=False, default=10.0, help="lr for regularizer")
parser.add_argument("--save_model", type=int, required=False, default=0, help="flag for model saving")
parser.add_argument("--save_dir", type=str, required=False, help="save directory for the obtained results")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    # Parse command line argument `partition`
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    params_config = vars(args)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)

    net: Net = load_arcface_model(
        name=args.model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
        pretrained=args.pretrained,
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())
    init_embeddings: NDArray = net.get_weights()[-1]

    server_config = ServerConfig(num_rounds=args.num_rounds)

    def eval_config(server_rnd: int) -> Dict[str, Scalar]:
        config = {
            "val_steps": 5,
            "batch_size": args.batch_size,
        }
        return config

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(dataset_name=dataset, train=False, target=target)
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

    def timestamp_aggregation_fn(fit_metrics: Dict[str, Scalar]) -> Dict[str, Dict[str, Scalar]]:
        timestamps_aggregated: Dict[str, Dict[str, float]] = {}
        for cid, metrics in fit_metrics.items():
            timestamps_aggregated[cid] = {}
            timestamps_aggregated[cid]["comm"] = metrics["total"] - metrics["comp"]
            timestamps_aggregated[cid]["comp"] = metrics["comp"]
        return timestamps_aggregated

    # Create strategy
    strategy = load_strategy(
        strategy_name=args.strategy,
        params_config=params_config,
        init_parameters=init_parameters,
        init_embeddings=init_embeddings,
        evaluate_fn=get_eval_fn(net, args.dataset, args.target),
        fit_metrics_aggregation_fn=timestamp_aggregation_fn,
    )
    client_manager = SimpleClientManager()
    server = CustomServer(
        client_manager=client_manager,
        strategy=strategy,
        save_model=args.save_model,
        save_dir=Path(args.save_model) / "models",
        net=net,
    )

    # Start Flower server for four rounds of federated learning
    hist = start_server(
        server_address=args.server_address,
        server=server,
        config=server_config,
    )

    # Save results
    save_path = Path(args.save_dir) / "config.yaml"
    config = vars(args)
    with open(save_path, "w") as outfile:
        yaml.dump(config, outfile)
    save_path = Path(args.save_dir) / "metrics" / "timestamps_federated.json"
    with open(save_path, "w") as outfile:
        json.dump(hist.timestamps_distributed, outfile)
    save_path = Path(args.save_dir) / "metrics" / "timestamps_centralized.json"
    with open(save_path, "w") as outfile:
        json.dump(hist.timestamps_centralized, outfile)
    save_path = Path(args.save_dir) / "metrics" / "accuracy_centralized.json"
    with open(save_path, "w") as outfile:
        json.dump(hist.metrics_centralized, outfile)


if __name__ == "__main__":
    main()
