import timeit
from logging import INFO
from typing import Dict

import torch
from flwr.client import Client
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
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from models.base_model import Net
from models.driver import test, train
from models.metric_learning import ArcFaceLoss, CosineContrastiveLoss
from torch.utils.data import DataLoader
from utils.utils_dataset import load_federated_dataset
from utils.utils_model import load_arcface_model


class FlowerFaceClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        self.trainset = load_federated_dataset(dataset_name=self.dataset, id=self.cid, train=True, target=self.target)
        self.testset = load_federated_dataset(dataset_name=self.dataset, id=self.cid, train=False, target=self.target)

        # model configuration
        self.model = config["model_name"]
        self.out_dims = config["out_dims"]
        self.input_spec = config["input_spec"]

        self.net: Net = load_arcface_model(
            name=self.model,
            input_spec=self.input_spec,
            out_dims=self.out_dims,
        )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.net.get_weights())
        return GetParametersRes(status=Code.OK, parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        start_time = timeit.default_timer()
        # unwrapping FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        epochs: int = int(ins.config["local_epochs"])
        batch_size: int = int(ins.config["batch_size"])
        lr: float = float(ins.config["lr"])
        weight_decay: float = float(ins.config["weight_decay"])
        criterion_name: str = str(ins.config["criterion_name"])

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

        if criterion_name == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "ArcFace":
            assert "scale" in ins.config
            assert "margin" in ins.config
            criterion = ArcFaceLoss(s=float(ins.config["scale"]), m=float(ins.config["margin"]))
        elif criterion_name == "CCL":
            criterion = CosineContrastiveLoss()

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            criterion=criterion,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())
        comp_stamp = timeit.default_timer() - start_time
        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(self.trainset),
            metrics={"comp": comp_stamp},
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


class FlowerFaceRayClient(Client):
    def __init__(self, cid: str, config: Dict[str, str]):
        self.cid = cid

        # dataset configuration
        self.dataset = config["dataset_name"]
        self.target = config["target_name"]

        # model configuration
        self.model = config["model_name"]
        self.net: Net = load_arcface_model(
            name=self.model,
            input_spec=config["input_spec"],
            out_dims=config["out_dims"],
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
        criterion_name: str = str(ins.config["criterion_name"])

        # set parameters
        self.net.set_weights(weights)

        # dataset configuration train / validation
        trainset = load_federated_dataset(dataset_name=self.dataset, id=self.cid, train=True, target=self.target)
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

        if criterion_name == "CrossEntropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "ArcFace":
            assert ins.config["scale"] is not None
            assert ins.config["margin"] is not None
            criterion = ArcFaceLoss(s=float(ins.config["scale"]), m=float(ins.config["margin"]))
        elif criterion_name == "CCL":
            criterion = CosineContrastiveLoss()

        train(
            self.net,
            trainloader=trainloader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            criterion=criterion,
            device=self.device,
        )
        parameters_prime: Parameters = ndarrays_to_parameters(self.net.get_weights())

        return FitRes(
            status=Status(Code.OK, message="Success fit"),
            parameters=parameters_prime,
            num_examples=len(trainset),
            metrics={"cid": self.cid},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # unwrap FitIns
        weights: NDArrays = parameters_to_ndarrays(ins.parameters)
        batch_size: int = int(ins.config["batch_size"])

        self.net.set_weights(weights)

        testset = load_federated_dataset(dataset_name=self.dataset, id=self.cid, train=False, target=self.target)
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
