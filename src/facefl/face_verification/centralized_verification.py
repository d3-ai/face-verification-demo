import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils_wandb import custom_wandb_init

from facefl.dataset import configure_dataset, load_centralized_dataset
from facefl.model import load_arcface_model

# import wandb
from facefl.model.base_model import Net
from facefl.model.metric_learning import ArcFaceLoss

parser = argparse.ArgumentParser("Simulation: Centralized learning.")
parser.add_argument(
    "--dataset",
    type=str,
    required=False,
    choices=["CIFAR10", "CelebA"],
    default="CIFAR10",
    help="dataset name for Centralized training",
)
parser.add_argument(
    "--target",
    type=str,
    required=False,
    choices=["small", "medium", "large"],
    default=None,
    help="dataset size for Centralized training",
)
parser.add_argument(
    "--model",
    type=str,
    required=False,
    choices=["ResNet18", "GNResNet18"],
    default="ResNet18",
    help="model name for Centralized training",
)
parser.add_argument(
    "--pretrained",
    type=str,
    required=False,
    choices=["IMAGENET1K_V1", "None"],
    default="None",
    help="model name for Federated training",
)
parser.add_argument(
    "--max_epochs", type=int, required=False, default=100, help="Max epochs"
)
parser.add_argument(
    "--batch_size", type=int, required=False, default=10, help="batchsize for training"
)
parser.add_argument(
    "--criterion",
    type=str,
    required=False,
    default="CrossEntropy",
    choices=["CrossEntropy", "ArcFace"],
    help="Criterion of classification performance",
)
parser.add_argument(
    "--lr", type=float, required=False, default=0.01, help="learning rate"
)
parser.add_argument(
    "--momentum", type=float, required=False, default=0.0, help="momentum"
)
parser.add_argument(
    "--weight_decay", type=float, required=False, default=0.0, help="weigh_decay"
)
parser.add_argument(
    "--scale", type=float, required=False, default=0.0, help="scale for arcface loss"
)
parser.add_argument(
    "--margin", type=float, required=False, default=0.0, help="margin for arcface loss"
)
parser.add_argument(
    "--save_model", type=int, required=False, default=0, help="flag for model saving"
)
parser.add_argument(
    "--seed", type=int, required=False, default=1234, help="Random seed"
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    net: Net = load_arcface_model(
        name=args.model,
        input_spec=dataset_config["input_spec"],
        out_dims=dataset_config["out_dims"],
        pretrained=args.pretrained,
    )
    print(net)

    params_config = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "criterion": args.criterion,
        "pretrained": args.pretrained,
        "scheduler": "CosineAneealingWarmup",
        "margin": args.margin,
        "target": args.target,
        "scale": args.scale,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "model_name": args.model,
        "seed": args.seed,
        "api_key_file": os.environ["WANDB_API_KEY_FILE"],
    }
    custom_wandb_init(
        config=params_config,
        project=f"{args.dataset}_verifications",
        strategy="Centralized",
    )

    # dataset
    trainset = load_centralized_dataset(
        dataset_name=args.dataset, train=True, target=args.target
    )
    testset = load_centralized_dataset(
        dataset_name=args.dataset, train=False, target=args.target
    )

    trainloader = DataLoader(
        trainset,
        batch_size=params_config["batch_size"],
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    testloader = DataLoader(
        testset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True
    )

    # criterion
    if args.criterion == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "ArcFace":
        criterion = ArcFaceLoss(s=args.scale, m=args.margin)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    # train loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # optimizer = torch.optim.SGD(
    #     net.parameters(),
    #     lr=params_config["lr"],
    #     momentum=params_config["momentum"],
    #     weight_decay=params_config["weight_decay"],
    # )
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.005,
        momentum=params_config["momentum"],
        weight_decay=params_config["weight_decay"],
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.005)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=80,
        cycle_mult=1.0,
        max_lr=params_config["lr"],
        min_lr=0.005,
        warmup_steps=20,
        gamma=1.0,
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    for epoch in range(params_config["max_epochs"]):
        net.train()
        for _, data in tqdm(
            enumerate(trainloader),
            total=len(trainloader),
            file=sys.stdout,
            desc=f"Epoch: {epoch} / {params_config['max_epochs']}",
            leave=False,
        ):
            images, labels = data[0].to(device, non_blocking=True), data[1].to(
                device, non_blocking=True
            )
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        net.eval()
        correct, total, steps, loss = 0, 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device, non_blocking=True), labels.to(
                    device, non_blocking=True
                )
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                steps += 1
    #     wandb.log({"test_loss": loss / steps, "test_acc": correct / total})

    # if args.save_model:
    #     save_path = os.path.join(wandb.run.dir, "final_model.pth")
    #     torch.save(net.to("cpu").state_dict(), save_path)


if __name__ == "__main__":
    main()
