from flwr.common.logger import log

import timeit
from logging import INFO, DEBUG

# typing
from .client_manager import ClientManager
from .server import Server
from .history import History
from .strategy.strategy import Strategy
from .server import fit_clients, FitResultsAndFailures
from common.typing import Parameters, Scalar
from typing import Optional, Tuple, Dict


class CustomServer(Server):
    """
    Flower server implementation for system performance measurement.
    """
    def __init__(self, client_manager: ClientManager, strategy: Optional[Strategy] = None) -> None:
        super(CustomServer, self).__init__(client_manager=client_manager,strategy=strategy)

    def fit(self, num_rounds: int, timeout: Optional[float]):
        history = History()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO,"initial parameters (loss, other metrics): %s, %s",res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(server_round=current_round, timeout=timeout, start_time=start_time)
            if res_fit:
                parameters_prime, timestamps_cen, timestamps_fed, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                timestamps_cen["fit_round"] = timeit.default_timer()-start_time

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                log(INFO,"fit progress: (%s, %s, %s, %s)",current_round,loss_cen,metrics_cen,timeit.default_timer() - start_time,)
                timestamps_cen["eval_round"] = timeit.default_timer()-start_time
            history.add_timestamps_centralized(server_round=current_round, timestamps=timestamps_cen)
            history.add_timestamps_distributed(server_round=current_round, timestamps=timestamps_fed)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history, self.parameters
    
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
        start_time: Optional[float]
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        timestamps: Dict[str, Scalar] ={}

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
            timestamps["fitres_received"]
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
            timestamps["model_aggregation"]
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, timestamps, metrics_aggregated, (results, failures)