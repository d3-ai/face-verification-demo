import warnings
from logging import INFO
from typing import Dict

import torch
from flwr.client import Client, NumPyClient
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    NDArrays,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from torch.utils.data import DataLoader

from facefl.model.base_model import Net
from facefl.model.driver import test, train
from facefl.utils.utils_dataset import (
    configure_dataset,
    load_federated_dataset,
    split_validation,
)
from facefl.utils.utils_model import load_model

warnings.filterwarnings("ignore")


class FlowerClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]
        validation_ratio = 0.8
        dataset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=True, target=self.target
        )
        self.trainset, self.valset = split_validation(
            dataset, split_ratio=validation_ratio
        )
        self.testset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=False, target=self.target
        )

        # model configuration
        self.model = config["model_name"]
        dataset_config = configure_dataset(self.dataset, target=self.target)
        self.net: Net = load_model(
            name=self.model,
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
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
        print(ins.config)
        weight_decay: float = float(ins.config["weight_decay"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        valloader = DataLoader(
            self.valset, batch_size=100, shuffle=False, drop_last=False
        )

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
            use_tqdm=True,
        )
        results = test(self.net, valloader, device=self.device)
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())
        log(
            INFO,
            "fit() on client cid=%s: val loss %s / val acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size)
        results = test(self.net, testloader=testloader)
        log(
            INFO,
            "evaluate() on client cid=%s: test loss %s / test acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return EvaluateRes(
            status=Status(Code.OK, message="Success eval"),
            loss=float(results["loss"]),
            num_examples=len(self.testset),
            metrics={"accuracy": results["acc"]},
        )


class FlowerRayClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # model configuration
        self.model = config["model_name"]
        dataset_config = configure_dataset(self.dataset, target=self.target)
        self.net: Net = load_model(
            name=self.model,
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
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
        trainset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=True, target=self.target
        )
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())

        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(trainset),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)

        testset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=False, target=self.target
        )
        # testset = load_dataset(name=self.dataset, id=self.cid, train=False, target=self.target)
        testloader = DataLoader(testset, batch_size=batch_size)
        results = test(
            self.net,
            testloader=testloader,
        )
        log(
            INFO,
            "evaluate() on client cid=%s: test loss %s / test acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return EvaluateRes(
            status=Status(Code.OK, message="Success eval"),
            loss=float(results["loss"]),
            num_examples=len(testset),
            metrics={"accuracy": results["acc"]},
        )


class FlowerNumPyClient(NumPyClient):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]
        validation_ratio = 0.8
        dataset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=True, target=self.target
        )
        self.trainset, self.valset = split_validation(
            dataset, split_ratio=validation_ratio
        )
        self.testset = load_federated_dataset(
            dataset_name=self.dataset, id=self.cid, train=False, target=self.target
        )

        # model configuration
        self.model = config["model_name"]
        dataset_config = configure_dataset(self.dataset)
        self.net: Net = load_model(
            name=self.model,
            input_spec=dataset_config["input_spec"],
            out_dims=dataset_config["out_dims"],
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config) -> NDArrays:
        return self.net.get_weights()

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> FitRes:
        # unwrapping FitIns
        epochs: int = int(config["local_epochs"])
        batch_size: int = int(config["batch_size"])
        lr: float = float(config["lr"])

        # set parameters
        self.net.set_weights(parameters)

        # dataset configuration train / validation
        trainloader = DataLoader(
            self.trainset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        # valloader = DataLoader(
        #     self.valset, batch_size=100, shuffle=False, drop_last=False
        # )

        train(
            self.net, trainloader=trainloader, epochs=epochs, lr=lr, device=self.device
        )
        parameters_prime: NDArrays = self.net.get_weights()
        # results: Dict[str, Scalar] = test(self.net, valloader, device=self.device)

        return parameters_prime, len(self.trainset), {}

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)
        testloader = DataLoader(self.testset, batch_size=batch_size)
        results = test(
            self.net,
            testloader=testloader,
        )
        log(
            INFO,
            "evaluate() on client cid=%s: test loss %s / test acc %s",
            self.cid,
            results["loss"],
            results["acc"],
        )

        return EvaluateRes(
            status=Status(Code.OK, message="Success eval"),
            loss=float(results["loss"]),
            num_examples=len(self.testset),
            metrics={"accuracy": results["acc"]},
        )


if __name__ == "__main__":
    client_config = {"dataset_name": "CIFAR10", "model_name": "tinyCNN"}

    def fit_config() -> Dict[str, int]:
        config = {
            "local_epochs": 5,
            "batch_size": 10,
        }
        return config

    client = FlowerClient(cid="0", config=client_config)
    config = fit_config()
    model = load_model(name="tiny_CNN", input_spec=(3, 32, 32))
    init_parameters = ndarrays_to_parameters(model.get_weights())
    fit_ins = FitIns(parameters=init_parameters, config=config)
    eval_ins = EvaluateIns(
        parameters=init_parameters, config={"val_steps": 5, "batch_size": 10}
    )
    client.fit(fit_ins)
    client.evaluate(eval_ins)
    print("Dry Run Successful")
