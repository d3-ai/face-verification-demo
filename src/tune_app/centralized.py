import os

from ray import tune
from ray.tune.integration.wandb import wandb_mixin
import wandb

import torch
from torch.utils.data import DataLoader


# original modules
from driver import train, test
from models.base_model import Net
from utils.utils_dataset import configure_dataset, load_dataset, split_validation
from utils.utils_model import load_model

# type notations
from typing import List
from torch.utils.data.dataset import Dataset

@wandb_mixin
def centralized(config):
    dataset_config = configure_dataset(dataset_name=config["dataset_name"])
    net: Net = load_model(name=config["model_name"], input_spec=dataset_config["input_spec"], out_dims=dataset_config["out_dims"], pretrained=False)

    # load dataset
    split_ratio = 0.8
    dataset: Dataset = load_dataset(name=config["dataset_name"], train=True, download=True)
    trainset, valset = split_validation(dataset=dataset, split_ratio=split_ratio)
    testset: Dataset = load_dataset(name=config["dataset_name"], train=False, download=True)

    # device setting
    device = "cpu"
    kwargs = {}
    if torch.cuda.is_available():
        device = "cuda:0"
        kwargs["num_workers"] = 2
        kwargs["pin_memory"] = True
        # if torch.cuda.device_count()>1:
            # net = torch.nn.DataParallel(net)
    net.to(device)
    
    trainloader: DataLoader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, **kwargs)
    valloader: DataLoader = DataLoader(valset, batch_size=100, shuffle=False, **kwargs)
    testloader: DataLoader = DataLoader(testset, batch_size=100, shuffle=False, **kwargs)

    val_losses: List[float] = []
    val_acces: List[float] = []

    for epoch in range(config["max_epochs"]):
        train(net=net, trainloader=trainloader, epochs=1, lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"], device=device)
        val_res = test(net=net, testloader=valloader, device=device)
        test_res = test(net=net, testloader=testloader, device=device)

        val_losses.append(val_res['loss'])
        val_acces.append(val_res['acc'])
        # if val_res['loss'] == min(val_losses):
            # save_model(wandb.run.dir, 'checkpoint_best.pt', net, val_losses, val_acces, epoch)
            # file_path = os.path.join(wandb.run.dir,"checkpoint_best.pt")
        wandb.log({"loss": val_res['loss'], "accuracy": val_res['acc'], "test_loss": test_res['loss'], "test_acc": test_res['acc']})
        tune.report(loss=val_res['loss'], accuracy=val_res['acc'],test_loss=test_res['loss'],test_acc=test_res['acc'],)

def save_model(
    save_dir: str,
    filename: str,
    net: Net,
    losses:List[float],
    accs:List[float],
    num_epochs: int):
    path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': num_epochs,'model_state_dict': net.to("cpu").state_dict(),
        'val_loss': losses[-1],'val_accs': accs[-1],
    }, path)