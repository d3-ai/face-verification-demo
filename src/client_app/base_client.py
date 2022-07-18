import warnings

import torch
from torch.utils.data import DataLoader

from logging import INFO
from flwr.common.logger import log

from driver import train, test
from models.base_model import Net
from utils.utils_dataset import (
    configure_dataset,
    load_dataset,
    split_validation
)
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

warnings.filterwarnings("ignore")

class FlowerClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]
        validation_ratio=0.8
        dataset = load_dataset(name=self.dataset, id=self.cid, train=True, target=self.target)
        self.trainset, self.valset = split_validation(dataset, split_ratio=validation_ratio)
        self.testset = load_dataset(name=self.dataset, id=self.cid, train=False, target=self.target)

        # model configuration
        self.model = config["model_name"]
        dataset_config = configure_dataset(self.dataset)
        self.net: Net = load_model(name=self.model, input_spec=dataset_config['input_spec'], out_dims=dataset_config['out_dims'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = weights_to_parameters(self.net.get_weights())
        return GetParametersRes(status=Code.OK, parameters=parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: Weights = parameters_to_weights(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(self.valset, batch_size=100,shuffle=False, drop_last=False)

        train(self.net, trainloader=trainloader, epochs=epochs, lr=lr, device=self.device)
        results = test(self.net, valloader, device=self.device)
        parameters_prime: Parameters = weights_to_parameters(self.net.get_weights())
        log(INFO, "fit() on client cid=%s: val loss %s / val acc %s", self.cid, results["loss"], results["acc"])

        return FitRes(status=Status(Code.OK ,message="Success fit"), parameters=parameters_prime, num_examples=len(self.trainset), metrics=results)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: Weights = parameters_to_weights(ins.parameters)
        steps: int = int(ins.config["val_steps"])
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size)
        results = test(self.net, testloader=testloader, steps=steps)
        log(INFO, "evaluate() on client cid=%s: test loss %s / test acc %s", self.cid, results['loss'], results['acc'])

        return EvaluateRes(status=Status(Code.OK, message="Success eval"), loss=float(results['loss']), num_examples=len(self.testset), metrics={"accuracy": results['acc']})


if __name__ == "__main__":
    client_config = {
        "dataset_name": "CIFAR10",
        "model_name": "tinyCNN"
    }
    def fit_config()->Dict[str, int]:
        config = {
            "local_epochs": 5,
            "batch_size": 10,
        }
        return config
    client = FlowerClient(cid="0", config=client_config)
    config = fit_config()
    model = load_model(name="tiny_CNN", input_spec=(3,32,32))
    init_parameters = weights_to_parameters(model.get_weights())
    fit_ins = FitIns(parameters=init_parameters, config=config)
    eval_ins = EvaluateIns(parameters=init_parameters, config={"val_steps": 5, "batch_size": 10})
    client.fit(fit_ins)
    client.evaluate(eval_ins)
    print("Dry Run Successful")
