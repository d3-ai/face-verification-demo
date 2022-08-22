import argparse
import json
from pathlib import Path

import random
import numpy as np
import torch

from tune_app.centralized import centralized
from utils.utils_tune import run_tuning

parser=argparse.ArgumentParser("Centralized traing with ray tune.")
parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "CelebA"], help="Centralized ray tuning config: dataset (project) name")
parser.add_argument("--model", type=str, required=True, choices=["tinyCNN", "ResNet18", "GNResNet18"], help="Centralized ray tuning config: model name")
parser.add_argument("--num_samples", type=int, required=True, help="Centralized ray tuning config: num of trial")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    group = "Centralized_" + args.model
    yaml_path = Path("./conf") / args.dataset / group / "search_space.yaml"

    best_trial = run_tuning(
        tune_fn=centralized,
        metric='loss',
        mode='min',
        name="CIFAR10_Centralized",
        yaml_path=yaml_path,
        group=group,
        resources={"cpu": 2, "gpu":0.08},
        num_samples=args.num_samples,
    )
    
    json_path = Path("./conf") / args.dataset / group / "best_config.json"
    with open(json_path, "w") as outfile:
        best_config = {key: val for key, val in best_trial.config.items() if key != "wandb"}
        json.dump(best_config, outfile)

if __name__ == "__main__":
    main()