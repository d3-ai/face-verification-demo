import argparse
import gc
import json
import os
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from flwr.client import Client
from flwr.common import NDArray, NDArrays, Parameters, Scalar, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from torch.utils.data import DataLoader

from facefl.client.face_client import FlowerFaceRayClient
from facefl.model.base_model import Net
from facefl.model.driver import test, verify
from facefl.server.wandb_server import WandbServer
from facefl.simulation.app import start_simulation
from facefl.utils.utils_dataset import configure_dataset, load_centralized_dataset
from facefl.utils.utils_model import load_arcface_model
from facefl.utils.utils_server import load_strategy
from facefl.utils.utils_wandb import custom_wandb_init

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Federated face verification simulation")
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["CelebA"],
    help="FL config: dataset name",
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
    "--target",
    type=str,
    required=True,
    help="FL config: target partitions for common dataset target attributes for celeba",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["ResNet18", "GNResNet18"],
    help="FL config: model name",
)
parser.add_argument(
    "--pretrained",
    type=str,
    required=False,
    choices=["IMAGENET1K_V1", "None", "CelebA"],
    default=None,
    help="pretraing recipe",
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
    "--criterion",
    type=str,
    required=False,
    default="ArcFace",
    choices=["ArcFace", "CCL"],
    help="Criterion of classification performance",
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
    "--scale", type=float, required=False, default=0.0, help="scale for arcface loss"
)
parser.add_argument(
    "--margin", type=float, required=False, default=0.0, help="margin for arcface loss"
)
parser.add_argument(
    "--nu",
    type=float,
    required=False,
    default=0.9,
    help="margin for cosine contrastive loss",
)
parser.add_argument(
    "--lam", type=float, required=False, default=10.0, help="lr for regularizer"
)
parser.add_argument(
    "--save_model", type=int, required=False, default=0, help="flag for model saving"
)
parser.add_argument(
    "--save_dir",
    type=str,
    required=False,
    help="save directory for the obtained results",
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
    params_config = vars(args)
    params_config["api_key_file"] = os.environ["WANDB_API_KEY_FILE"]
    custom_wandb_init(
        config=params_config, project="hoge_verifications", strategy=args.strategy
    )

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)

    net: Net = load_arcface_model(
        name=args.model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
        pretrained=args.pretrained,
    )
    init_parameters: Parameters = ndarrays_to_parameters(net.get_weights())
    init_embeddings: NDArray = net.get_weights()[-1]

    client_config = {
        "dataset_name": args.dataset,
        "target_name": args.target,
        "model_name": args.model,
    }
    client_config.update(dataset_config)
    if args.strategy == "FedAvg":
        client_config["out_dims"] = 10
    elif args.strategy == "FedAwS":
        client_config["out_dims"] = 1
    else:
        raise NotImplementedError(f"{args.strategy} is not implemented.")
    server_config = ServerConfig(num_rounds=args.num_rounds)

    def get_eval_fn(model: Net, dataset: str, target: str) -> Callable:
        testset = load_centralized_dataset(
            dataset_name=dataset, train=False, target=target
        )
        testloader = DataLoader(testset, batch_size=6)

        def evaluate(
            server_round: int, weights: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            model.set_weights(weights)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            results = verify(model, testloader, device=device)
            results = test(model, testloader, device=device)
            torch.cuda.empty_cache()
            gc.collect()
            return results["loss"], {"accuracy": results["acc"]}

        return evaluate

    def client_fn(cid: str) -> Client:
        return FlowerFaceRayClient(cid, client_config)

    strategy = load_strategy(
        strategy_name=args.strategy,
        params_config=params_config,
        init_parameters=init_parameters,
        init_embeddings=init_embeddings,
        evaluate_fn=get_eval_fn(net, args.dataset, args.target),
    )
    server = WandbServer(
        client_manager=SimpleClientManager(),
        strategy=strategy,
        save_model=args.save_model,
        net=net,
    )
    client_resources = {"num_cpus": 2, "num_gpus": 1.0}
    ray_config = {"include_dashboard": False, "address": "auto"}
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        server=server,
        config=server_config,
        ray_init_args=ray_config,
        keep_initialised=True,
    )
    save_path = Path(args.save_dir) / "metrics" / "accuracy_centralized.json"
    with open(save_path, "w") as outfile:
        json.dump(hist.metrics_centralized, outfile)


if __name__ == "__main__":
    main()
