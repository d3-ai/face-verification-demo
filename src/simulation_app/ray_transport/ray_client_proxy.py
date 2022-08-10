"""Ray-based Flower ClientProxy implementation."""


from typing import Callable, Dict, Optional, cast

import ray
from common import (
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    ReconnectIns,
    DisconnectRes,
)
from client_app import Client, ClientLike, to_client
from server_app.client_proxy import ClientProxy

ClientFn = Callable[[str], ClientLike]


class RayClientProxy(ClientProxy):
    """Flower client proxy which delegates work using Ray."""

    def __init__(self, client_fn: ClientFn, cid: str, resources: Dict[str, int]):
        super().__init__(cid)
        self.client_fn = client_fn
        self.resources = resources

    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        """Returns client's properties."""
        future_get_properties_res = launch_and_get_properties.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        res = ray.worker.get(future_get_properties_res, timeout=timeout)
        return cast(
            GetPropertiesRes,
            res,
        )

    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        """Return the current local model parameters."""
        future_paramseters_res = launch_and_get_parameters.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        res = ray.worker.get(future_paramseters_res, timeout=timeout)
        return cast(
            GetParametersRes,
            res,
        )

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        """Train model parameters on the locally held dataset."""
        future_fit_res = launch_and_fit.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        res = ray.worker.get(future_fit_res, timeout=timeout)
        return cast(
            FitRes,
            res,
        )

    def evaluate(
        self, ins: EvaluateIns, timeout: Optional[float]
    ) -> EvaluateRes:
        """Evaluate model parameters on the locally held dataset."""
        future_evaluate_res = launch_and_evaluate.options(  # type: ignore
            **self.resources,
        ).remote(self.client_fn, self.cid, ins)
        res = ray.worker.get(future_evaluate_res, timeout=timeout)
        return cast(
            EvaluateRes,
            res,
        )

    def reconnect(
        self, ins: ReconnectIns, timeout: Optional[float]
    ) -> DisconnectRes:
        """Disconnect and (optionally) reconnect later."""
        return DisconnectRes(reason="")  # Nothing to do here (yet)


@ray.remote
def launch_and_get_properties(
    client_fn: ClientFn, cid: str, get_properties_ins: GetPropertiesIns
) -> GetPropertiesRes:
    """Exectue get_properties remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.get_properties(get_properties_ins)


@ray.remote
def launch_and_get_parameters(
    client_fn: ClientFn, cid: str, get_parameters_ins: GetParametersIns
) -> GetParametersRes:
    """Exectue get_parameters remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.get_parameters(get_parameters_ins)


@ray.remote(max_calls=1)
def launch_and_fit(
    client_fn: ClientFn, cid: str, fit_ins: FitIns
) -> FitRes:
    """Exectue fit remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.fit(fit_ins)


@ray.remote
def launch_and_evaluate(
    client_fn: ClientFn, cid: str, evaluate_ins: EvaluateIns
) -> EvaluateRes:
    """Exectue evaluate remotely."""
    client: Client = _create_client(client_fn, cid)
    return client.evaluate(evaluate_ins)


def _create_client(client_fn: ClientFn, cid: str) -> Client:
    """Create a client instance."""
    client_like: ClientLike = client_fn(cid)
    return to_client(client_like)