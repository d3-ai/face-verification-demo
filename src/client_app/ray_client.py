import torch
from torch.utils.data import DataLoader
from flwr.client import Client

from logging import INFO
from flwr.common.logger import log

from models.base_model import Net
from utils.utils_model import load_model
from utils.utils_dataset import configure_dataset, load_federated_dataset
from common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    Status,
    Code,
    GetParametersIns,
    GetParametersRes,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    NDArrays,
)
from driver import train, test
from typing import Dict

class FlowerRayClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # model configuration
        self.model = config["model_name"]
        dataset_config = configure_dataset(self.dataset, target=self.target)
        self.net: Net = load_model(name=self.model, input_spec=dataset_config['input_spec'], out_dims=dataset_config['out_dims'])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.net.get_weights())
        return GetParametersRes(status=Code.OK, parameters=parameters)
    
    def fit(self, ins: FitIns) -> FitRes:
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainset = load_federated_dataset(dataset_name=self.dataset, id = self.cid, train=True, target= self.target)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True, drop_last=True)

        train(self.net, trainloader=trainloader, epochs=epochs, lr=lr, weight_decay=weight_decay, device=self.device)
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())
        
        return FitRes(status=Status(Code.OK ,message="Success fit"), parameters=parameters_prime, num_examples=len(trainset), metrics={})

    def evaluate(self, ins: EvaluateIns)-> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        
        testset = load_federated_dataset(dataset_name=self.dataset, id = self.cid, train=False, target= self.target)
        # testset = load_dataset(name=self.dataset, id=self.cid, train=False, target=self.target)
        testloader = DataLoader(testset, batch_size=batch_size)
        results = test(self.net, testloader=testloader,)
        log(INFO, "evaluate() on client cid=%s: test loss %s / test acc %s", self.cid, results['loss'], results['acc'])

        return EvaluateRes(status=Status(Code.OK, message="Success eval"), loss=float(results['loss']), num_examples=len(testset), metrics={"accuracy": results['acc']})