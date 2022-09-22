from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    NDArray,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from server_app.client_manager import ClientManager
from server_app.client_proxy import ClientProxy

from .aggregate import aggregate_and_spreadout, weighted_loss_avg
from .fedavg import FedAvg

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
        lam: Optional[float] = 0.1,) -> None:
        """Federated learning strategy using Spreadout on server-side.
        Implementation based on http://proceedings.mlr.press/v119/yu20f/yu20f.pdf
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_evaluate (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            evaluate_fn : Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]]
                ]
            ]: Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters): Initial set of parameters from the server.
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
                Metrics aggregation function, optional.
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
        """
        self.embeddings: C x d tensors comprising of client class embeddings.
        """
        self.initial_embeddings = initial_embeddings
        self.embeddings_dict = {}

    def __repr__(self) -> str:
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    )-> List[Tuple[ClientProxy, FitIns]]:
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
        if not any(self.embeddings_dict):
            for idx, c in enumerate(clients):
                self.embeddings_dict[c.cid] = self.initial_embeddings[np.newaxis, idx, :]
        for client in clients:
            weights[-1] = self.embeddings_dict[client.cid]
            parameters_dict[client.cid] = ndarrays_to_parameters(weights)

        return [(client, FitIns(parameters=parameters_dict[client.cid], config=config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
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

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, client.cid)
            for client, fit_res in results
        ]

        parameters_aggregated, self.embeddings_dict = aggregate_and_spreadout(weights_results, num_clients=len(weights_results), num_features=512, nu=self.nu, lr=self.eta*self.lam)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(parameters_aggregated), metrics_aggregated
