import os
import yaml

import ray
from ray import tune
from ray.tune.trial import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.docker import DockerSyncer

from common import Scalar
from typing import Any, Callable, Dict, Optional

def trial_name_creator(trial: Trial):
    name = trial.trainable_name + "_" + trial.trial_id
    return name

def get_tune_sampler(config: Dict[str, Scalar]):
    if config['sampler'] == "choice":
        return tune.choice(config['categories'])
    elif config['sampler'] == "loguniform":
        return tune.loguniform(lower=config['lower'], upper=config['upper'], base=config['base'])
    else:
        raise NotImplementedError(f"{config['sampler']} is not implemented.")

def get_search_space_from_yaml(yaml_path: str)->Dict[str, Any]:
    with open(yaml_path, 'r') as f:
        yaml_dict: Dict[str, Dict[str, Scalar]] = yaml.safe_load(f)
    search_space = {}
    for key, config in yaml_dict.items():
        if config['sampler'] == "no":
            search_space[key] = config['value']
        else:
            search_space[key] = get_tune_sampler(config)
    return search_space

def run_tuning(
    tune_fn: Callable,
    metric: str,
    mode: str,
    name: str,
    yaml_path: str,
    group: str,  
    resources: Dict[str, Scalar],
    num_samples: int,
    local_dir: str = "./",
    search_alg_name: str = "Optuna", 
    schedular_name: str = "ASHA",
    trial_name_creator: Callable = trial_name_creator,
    sync_config: str =None,
    fail_fast: bool = True,
    )->Optional[Trial]:
    search_space = get_search_space_from_yaml(yaml_path)
    wandb_config = {
        "wandb": {
            "api_key_file": os.environ['WANDB_API_KEY_FILE'],
            "project": search_space["dataset_name"],
            "group": group,
        }
    }
    config = {}
    config.update(search_space)
    config.update(wandb_config)

    if search_alg_name == "Optuna":
        search_alg = OptunaSearch(metric=metric, mode=mode)
    else:
        raise NotImplementedError(f"{search_alg_name} is not implemented")
    
    if schedular_name == "ASHA":
        schedular = ASHAScheduler(metric=metric, mode=mode,grace_period=int(config["max_epochs"]/10), reduction_factor=2, max_t=config["max_epochs"])
    else:
        raise NotImplementedError(f"{schedular_name} is not implemented")

    if sync_config is None:
        sync_config = tune.SyncConfig(syncer=DockerSyncer)

    ray.init(address="auto")
    analysis = tune.run(
        tune_fn,
        name=name,
        config=config,
        resources_per_trial=resources,
        num_samples=num_samples,
        local_dir=local_dir,
        search_alg=search_alg,
        scheduler=schedular,
        checkpoint_at_end=True,
        log_to_file=False,
        trial_name_creator=trial_name_creator,
        sync_config=sync_config,
        fail_fast=fail_fast,
    )
    best_trial: Trial = analysis.get_best_trial('loss', 'min', 'last')
    print('Best trial config: {}'.format(best_trial.config))
    print('Best trial final validation loss: {}'.format(best_trial.last_result['loss']))
    print('Best trial final validation accuracy: {}'.format(best_trial.last_result['accuracy']))

    return best_trial


