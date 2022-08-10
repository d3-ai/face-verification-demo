"""Flower server app."""


import time
from dataclasses import dataclass
from logging import INFO, WARN
from typing import Optional, Tuple

from flwr.common.logger import log
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from .client_manager import ClientManager, SimpleClientManager
from .server import Server

DEFAULT_SERVER_ADDRESS = "[::]:8080"


@dataclass
class ServerConfig:
    """Flower server config.
    All attributes have default values which allows users to configure
    just the ones they care about.
    """

    num_rounds: int = 1
    round_timeout: Optional[float] = None

def _init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> Tuple[Server, ServerConfig]:
    # Create server instance if none was given
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config


def _fl(
    server: Server,
    config: ServerConfig,
) -> History:
    # Fit model
    hist = server.fit(num_rounds=config.num_rounds, timeout=config.round_timeout)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist


def run_server() -> None:
    """Run Flower server."""
    print("Running Flower server...")
    time.sleep(3)