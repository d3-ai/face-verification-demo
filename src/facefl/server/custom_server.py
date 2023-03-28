import concurrent
import os
import timeit
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

import torch
from flwr.common import Code, FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy
from models.base_model import Net

from .custom_history import CustomHistory

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]


class CustomServer(Server):
    """
    Flower server implementation for system performance measurement.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        save_model: bool = False,
        save_dir: str = None,
        net: Net = None,
    ) -> None:
        super(CustomServer, self).__init__(
            client_manager=client_manager, strategy=strategy
        )
        self.save_model = save_model
        if self.save_model:
            assert net is not None
            assert save_dir is not None
            self.net = net
            self.save_dir = save_dir

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = CustomHistory()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1]
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(
                server_round=current_round, timeout=timeout, start_time=start_time
            )
            if res_fit:
                (
                    parameters_prime,
                    timestamps_cen,
                    timestamps_fed,
                    _,
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                timestamps_cen["fit_round"] = timeit.default_timer() - start_time

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                timestamps_cen["eval_round"] = timeit.default_timer() - start_time
            history.add_timestamps_centralized(
                server_round=current_round, timestamps=timestamps_cen
            )
            history.add_timestamps_distributed(
                server_round=current_round, timestamps=timestamps_fed
            )

        if self.save_model:
            weights = parameters_to_ndarrays(self.parameters)
            self.net.set_weights(weights)
            save_path = os.path.join(self.save_dir, "final_model.pth")
            torch.save(self.net.to("cpu").state_dict(), save_path)
        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def fit_round(
        self, server_round: int, timeout: Optional[float], start_time: Optional[float]
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        timestamps: Dict[str, Scalar] = {}

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        timestamps["client_sampling"] = timeit.default_timer() - start_time
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s) at %s",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
            timestamps["client_sampling"],
        )
        self.set_max_workers(max_workers=len(client_instructions))

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        timestamps["fitres_received"] = timeit.default_timer() - start_time
        log(
            DEBUG,
            "fit_round %s: received %s results and %s failures at %s",
            server_round,
            len(results),
            len(failures),
            timestamps["fitres_received"],
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)
        timestamps["model_aggregation"] = timeit.default_timer() - start_time
        log(
            DEBUG,
            "fit_round %s: strategy aggregate the received parameters at %s",
            server_round,
            timestamps["model_aggregation"],
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return (
            parameters_aggregated,
            timestamps,
            metrics_aggregated,
            (results, failures),
        )


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    start_time = timeit.default_timer()
    fit_res = client.fit(ins, timeout=timeout)
    total_time = timeit.default_timer() - start_time
    fit_res.metrics["total"] = total_time
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
