import wandb
import uuid
import os

from typing import Dict, Any

def custom_wandb_init(config: Dict[str, Any], project: str, strategy: str):
    with open(config["api_key_file"], "r") as fp:
        api_key = fp.readline().strip()
    os.environ["WANDB_API_KEY"] = api_key
    config.pop("api_key_file")
    
    trial_id = uuid.uuid4().hex[:8]
    wandb.init(
        project=project,
        config=config,
        group=f"{strategy}_{config['model_name']}",
        name=f"{strategy}_{trial_id}",
    )