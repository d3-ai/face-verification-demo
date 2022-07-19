from flwr.server import Server
from flwr.server.history import History
from flwr.common.logger import log

from ray.tune.integration.wandb import wandb_mixin

import timeit
from logging import DEBUG, INFO

# typing
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.strategy import Strategy
from typing import Optional

import wandb

class RayTuneServer(Server):
    def __init__(self, client_manager: ClientManager, strategy: Optional[Strategy] = None) -> None:
        super(RayTuneServer, self).__init__(client_manager, strategy)
    @wandb_mixin
    def fit(self, config):
        history = History()

        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(INFO,"initial parameters (loss, other metrics): %s, %s",res[0], res[1])
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])
        
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            res_fit = self.fit_round(rnd, current_round)