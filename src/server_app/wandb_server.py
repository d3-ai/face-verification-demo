import os
import timeit
from logging import INFO
from typing import Optional

import torch
import wandb
from flwr.common import parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import Server
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from models.base_model import Net


class WandbServer(Server):
    """
    Flower server implementation for parameter search using ray.tune and wandb.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        save_model: bool = False,
        net: Net = None,
    ) -> None:
        super(WandbServer, self).__init__(client_manager=client_manager, strategy=strategy)
        self.save_model = save_model
        if self.save_model:
            assert net is not None
            self.net = net

    def fit(self, num_rounds: int, timeout: Optional[float]):
        history = History()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(INFO, "initial parameters (loss, other metrics): %s, %s", res[0], res[1])
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
            wandb.log({"test_loss": res[0], "test_acc": res[1]["accuracy"], "Aggregation round": 0})

        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)
                wandb.log(
                    {"test_loss": loss_cen, "test_acc": metrics_cen["accuracy"], "Aggregation round": current_round}
                )
        if self.save_model:
            weights = parameters_to_ndarrays(self.parameters)
            self.net.set_weights(weights)
            save_path = os.path.join(wandb.run.dir, "final_model.pth")
            torch.save(self.net.to("cpu").state_dict(), save_path)
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
