from typing import Callable, Dict, Optional, Tuple

from flwr.common import MetricsAggregationFn, NDArray, NDArrays, Parameters, Scalar
from flwr.server.strategy import Strategy
from server_app.strategy.fedavg import FedAvg
from server_app.strategy.fedaws import FedAwS


def load_strategy(
    strategy_name: str,
    params_config: Dict[str, Scalar],
    init_parameters: Parameters,
    init_embeddings: NDArray = None,
    evaluate_fn: Optional[
        Callable[
            [int, NDArrays, Dict[str, Scalar]],
            Optional[Tuple[float, Dict[str, Scalar]]],
        ]
    ] = None,
    fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
) -> Strategy:
    def eval_config(server_round: int) -> Dict[str, Scalar]:
        if params_config["batch_size"] > 10:
            config = {"batch_size": params_config["batch_size"]}
        else:
            config = {"batch_size": 10}
        return config

    if strategy_name == "FedAvg":
        assert params_config["criterion"] == "ArcFace"
        assert "scale" in params_config
        assert "margin" in params_config

        def fit_config(server_rnd: int) -> Dict[str, Scalar]:
            config = {
                "round": server_rnd,
                "local_epochs": params_config["local_epochs"],
                "batch_size": params_config["batch_size"],
                "lr": params_config["lr"],
                "weight_decay": params_config["weight_decay"],
                "criterion_name": params_config["criterion"],
                "scale": params_config["scale"],
                "margin": params_config["margin"],
            }
            return config

        strategy = FedAvg(
            fraction_fit=params_config["fraction_fit"],
            fraction_evaluate=1,
            min_fit_clients=int(params_config["num_clients"] * params_config["fraction_fit"]),
            min_evaluate_clients=params_config["num_clients"],
            min_available_clients=params_config["num_clients"],
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            initial_parameters=init_parameters,
        )
    elif strategy_name == "FedAwS":
        assert params_config["criterion"] == "CCL"
        assert "nu" in params_config
        assert "lam" in params_config

        def fit_config(server_rnd: int) -> Dict[str, Scalar]:
            config = {
                "round": server_rnd,
                "local_epochs": params_config["local_epochs"],
                "batch_size": params_config["batch_size"],
                "lr": params_config["lr"],
                "weight_decay": params_config["weight_decay"],
                "criterion_name": params_config["criterion"],
            }
            return config

        strategy = FedAwS(
            fraction_fit=params_config["fraction_fit"],
            fraction_evaluate=1,
            min_fit_clients=int(params_config["num_clients"] * params_config["fraction_fit"]),
            min_evaluate_clients=params_config["num_clients"],
            min_available_clients=params_config["num_clients"],
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=eval_config,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            initial_parameters=init_parameters,
            initial_embeddings=init_embeddings,
            nu=params_config["nu"],
            eta=params_config["lr"],
            lam=params_config["lam"],
        )

    return strategy
