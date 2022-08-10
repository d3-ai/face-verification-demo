import ray
import torch
from torch.utils.data import DataLoader
from flwr.client import NumPyClient

from logging import INFO
from flwr.common.logger import log

from models.base_model import Net
from utils.utils_model import load_model
from utils.utils_dataset import load_dataset, split_validation, configure_dataset
from common import ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, Parameters
from driver import train, test
from typing import Dict

class FlowerClient(NumPyClient):
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
    
    def get_parameters(self, config):
        parameters = ndarrays_to_parameters(self.net.get_weights())
        return parameters
    
    def fit(self, weights, config):
        # unwrapping FitIns
        epochs: int = int(config["local_epochs"])
        batch_size: int = int(config["batch_size"])
        lr: float = float(config["lr"])

        # set parameters
        self.net.set_weights(weights)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])

        # dataset configuration train / validation
        trainloader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        valloader = DataLoader(self.valset, batch_size=100,shuffle=False, drop_last=False)

        train(self.net, trainloader=trainloader, epochs=epochs, lr=lr, device=self.device)
        results = test(self.net, valloader, device=self.device)
        parameters_prime: NDArrays = self.net.get_weights()
        log(INFO, "fit() on client cid=%s: val loss %s / val acc %s", self.cid, results["loss"], results["acc"])

        return parameters_prime, len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(parameters)
        steps: int = int(config["val_steps"])
        batch_size: int = int(config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size)
        results = test(self.net, testloader=testloader, steps=steps)
        log(INFO, "evaluate() on client cid=%s: test loss %s / test acc %s", self.cid, results['loss'], results['acc'])

        return float(results["loss"]), len(testloader.dataset), {"accuracy": float(results["acc"])}