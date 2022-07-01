import ray
import warnings

import torch
from torch.utils.data import DataLoader

from logging import INFO
from flwr.common.logger import log

from driver import train, test
from models.base_model import Net
from utils.utils_dataset import load_dataset
from utils.utils_model import load_model
from common.parameter import parameters_to_weights, weights_to_parameters
from common.typing import (
    Status,
    Code,
    GetParametersIns,
    GetParametersRes,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Weights,
)
from .client import Client
from typing import Dict
from pathlib import Path

DATA_ROOT = Path("./data")
warnings.filterwarnings("ignore")

class FlowerRayClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # json configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # model configuration
        self.model = config["model_name"]
        if self.dataset == "CIFAR10":
            input_spec = (3,32,32)
            out_dims = 10
        elif self.dataset == "CelebA":
            input_spec = (3,64,64)
            out_dims = 2
        self.net: Net = load_model(name=self.model, input_spec=input_spec, out_dims=out_dims)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        raise Exception("Not implemented (server-side parameter initialization)")
    
    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: Weights = parameters_to_weights(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])

        # set parameters
        self.net.set_weights(weights)

        # ray configuration
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        # dataset configuration train /
        trainset = load_dataset(name=self.dataset, id=self.cid, train=True, target=self.target)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)

        results = train(self.net, trainloader=trainloader, epochs=epochs, lr=lr, device=self.device)
        parameters_prime: Parameters = weights_to_parameters(self.net.get_weights())
        log(INFO, "fit() on client cid=%s: train loss %s / train acc %s", self.cid, results["train_loss"], results["train_acc"])

        return FitRes(status=Status(Code.OK ,message="Success fit"), parameters=parameters_prime, num_examples=len(trainset), metrics=results)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: Weights = parameters_to_weights(ins.parameters)
        steps: int = int(ins.config["val_steps"])
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testset = load_dataset(name=self.dataset, id=self.cid, train=False, target=self.target)
        testloader = DataLoader(testset, batch_size=batch_size)
        loss, acc = test(self.net, testloader=testloader, steps=steps)
        log(INFO, "evaluate() on client cid=%s: test loss %s / test acc %s", self.cid, loss, acc)

        return EvaluateRes(status=Status(Code.OK, message="Success eval"), loss=float(loss), num_examples=len(testset), metrics={"accuracy": acc})


if __name__ == "__main__":
    client_config = {
        "dataset_name": "CIFAR10",
        "model_name": "tiny_CNN"
    }
    def fit_config()->Dict[str, int]:
        config = {
            "local_epochs": 5,
            "batch_size": 10,
        }
        return config
    client = FlowerRayClient(cid="0", config=client_config)
    config = fit_config()
    model = load_model(name="tiny_CNN", input_spec=(3,32,32))
    init_parameters = weights_to_parameters(model.get_weights())
    fit_ins = FitIns(parameters=init_parameters, config=config)
    eval_ins = EvaluateIns(parameters=init_parameters, config={"val_steps": 5, "batch_size": 10})
    client.fit(fit_ins)
    client.evaluate(eval_ins)
    print("Dry Run Successful")
