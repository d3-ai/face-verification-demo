import argparse
import json
import os
from pathlib import Path
import ray
from ray import tune
from ray.tune.trial import Trial
from ray.tune.integration.wandb import wandb_mixin
from torch.utils.data import random_split
import wandb

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

import driver
from models.base_model import Net
from utils.utils_model import load_model
from utils.utils_dataset import configure_dataset, load_dataset
from utils.utils_tune import get_search_space_from_yaml, run_tuning, save_model

parser=argparse.ArgumentParser("Centralized traing with ray tune.")
parser.add_argument("--dataset", type=str, required=True, choices=["CIFAR10", "CelebA"], help="Centralized ray tuning config: dataset (project) name")
parser.add_argument("--model", type=str, required=True, choices=["tiny_CNN", "ResNet18"], help="Centralized ray tuning config: model name")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

@wandb_mixin
def trainer(config):
    dataset_config = configure_dataset(dataset_name=config["dataset_name"])
    net: Net = load_model(
        name=config["model_name"], input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"], pretrained=False
    )

    dataset = load_dataset(name=config["dataset_name"], train=True, download=True)
    testset = load_dataset(name=config["dataset_name"], train=False, download=True)
    train_len = int(len(dataset)*0.9)
    val_len = int(len(dataset)*0.1)
    trainset, valset = random_split(dataset, [train_len, val_len])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True)
    valloader = DataLoader(valset, batch_size=100)
    testloader = DataLoader(testset, batch_size=100)
    val_losses = []
    val_acces = []
    for epoch in range(config["max_epochs"]):
        driver.train(net=net, trainloader=trainloader, epochs=1, lr=config["lr"], device=device)
        val_res = driver.test(net=net, testloader=valloader, device=device)
        test_res = driver.test(net=net, testloader=testloader, device=device)
        val_losses.append(val_res['loss'])
        val_acces.append(val_res['acc'])
        if val_res['loss'] == min(val_losses):
            save_model(wandb.run.dir, 'checkpoint_best.pt', net, val_losses, val_acces, epoch)
        wandb.log({"loss": val_res['loss'], "accuracy": val_res['acc'], "test_loss": test_res['loss'], "test_acc": test_res['acc']})
        tune.report(loss=val_res['loss'], accuracy=val_res['acc'],test_loss=test_res['loss'],test_acc=test_res['acc'],)

def main():
    args = parser.parse_args()
    print(args)
    group = "Centralized_" + args.model
    yaml_path = Path("./conf") / args.dataset / group / "search_space.yaml"
    params_dict = get_search_space_from_yaml(yaml_path)
    wandb_dict ={
        "wandb": {
            "api_key_file": os.environ['WANDB_API_KEY_FILE'],
            "project": args.dataset,
            "group": group,
        }
    }
    config = {}
    config.update(params_dict)
    config.update(wandb_dict)

    ray.init(address="auto")
    best_trial: Trial = run_tuning(
        tune_fn=trainer,
        metric='loss',
        mode='min',
        name="CIFAR10_Centralized",
        config=config,
        resources={"cpu": 1, "gpu":0.25},
        num_samples=4,
    )
    
    json_path = Path("./conf") / args.dataset / group / "best_config.json"
    with open(json_path, "w") as outfile:
        config = {key: val for key, val in best_trial.config.items()if key != "wandb"}
        json.dump(best_trial.config, outfile)

if __name__ == "__main__":
    main()