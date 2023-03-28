from functools import reduce
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArray,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from facefl.model.metric_learning import SpreadoutRegularizer


class FedAwS(FedAvg):
    """
    Federated Learning with Only Positive Labels [F. X. Yu et al. ICML 2020]
    Proposed strategy: FedAwS (Federated Averaging with Spreadout)
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        initial_embeddings: Optional[NDArray] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        nu: Optional[float] = 0.9,
        eta: Optional[float] = 0.1,
        lam: Optional[float] = 0.1,
    ) -> None:
        """Federated learning strategy using Spreadout on server-side.
        Implementation based on http://proceedings.mlr.press/v119/yu20f/yu20f.pdf
        Args:
            id_dict:
                simulation: key is index('0','1', ...), value is index('0','1', ...) cofiguring dataset partitioning
                exp: key is IPADDRESS:PORT, value is index('0','1', ...) cofiguring dataset partitioning
            nu (float, optional): Server-side margin parameter. Default to 0.9.
            eta (float, optional): Client-side learning rate. Defaults to 1e-1.
            lambda (float, optional): Server-side learning rate. Defaults to 1e-1.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )

        self.nu = nu
        self.eta = eta
        self.lam = lam
        self.embeddings = initial_embeddings
        self.client_dict = {}
        # self.embeddings_dict = {}

    def __repr__(self) -> str:
        rep = f"FedAwS(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        weights: NDArrays = parameters_to_ndarrays(parameters)
        parameters_dict: Dict[str, Parameters] = {}
        for i, client in enumerate(clients):
            if any(self.client_dict):
                idx = int(self.client_dict[client.cid])
            else:
                idx = i
            weights[-1] = self.embeddings[np.newaxis, idx, :]
            parameters_dict[client.cid] = ndarrays_to_parameters(weights)

        return [
            (client, FitIns(parameters=parameters_dict[client.cid], config=config))
            for client in clients
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit results as follows:
            Feature extracter: weighted
            Classifier Matrix:
        """
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        if not any(self.client_dict):
            for client, fit_res in results:
                self.client_dict[client.cid] = fit_res.metrics["cid"]
        # Convert results
        weights_results = [
            (
                parameters_to_ndarrays(fit_res.parameters),
                fit_res.num_examples,
                int(fit_res.metrics["cid"]),
            )
            for client, fit_res in results
        ]

        parameters_aggregated, self.embeddings = aggregate_and_spreadout(
            weights_results,
            num_clients=len(weights_results),
            num_features=512,
            nu=self.nu,
            lr=self.eta * self.lam,
        )

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = {
                client.cid: res.metrics for client, res in results
            }  # modified here
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated


def aggregate_and_spreadout(
    results: List[Tuple[NDArrays, int]],
    num_clients: int,
    num_features: int,
    nu: float,
    lr: float,
) -> Tuple[NDArrays, NDArray]:
    """Compute weighted average."""
    # Create a classification matrix from class embeddings
    embeddings: NDArray = np.zeros((num_clients, num_features))
    for weights, _, cid in results:
        embeddings[cid, :] = weights[-1]

    embeddings = torch.nn.Parameter(torch.tensor(embeddings))
    regularizer = SpreadoutRegularizer(nu=nu)
    optimizer = torch.optim.SGD([embeddings], lr=lr)
    optimizer.zero_grad()
    loss = regularizer(embeddings, out_dims=num_clients)
    print(loss)
    loss.backward()
    optimizer.step()
    embeddings = F.normalize(embeddings).detach().cpu().numpy()

    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples, _ in results])

    # Create a list of weights, each multiplied by the related number of examples
    feature_weights = [
        [layer * num_examples for layer in weights[:-1]]
        for weights, num_examples, _ in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*feature_weights)
    ]
    weights_prime.append(embeddings)

    return weights_prime, embeddings
