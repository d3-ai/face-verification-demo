import argparse
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.base_model import Net
from models.metric_learning import ArcFaceLoss
from utils.utils_dataset import configure_dataset, load_centralized_dataset 
from utils.utils_model import load_arcface_model
from utils.utils_plot import visualize_2d_embedded_space
from utils.utils_wandb import custom_wandb_init

parser = argparse.ArgumentParser("Simulation: Centralized learning.")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA"], default="CIFAR10", help="dataset name for Centralized training")
parser.add_argument("--target", type=str, required=False, choices=["small", "medium", "large"], default=None, help="dataset size for Centralized training")
parser.add_argument("--model", type=str, required=False, choices=["ResNet18", "GNResNet18"], default="ResNet18", help="model name for Centralized training")
parser.add_argument("--pretrained", type=str, required=False, choices=["IMAGENET1K_V1", None], default=None, help="model name for Federated training")
parser.add_argument("--max_epochs", type=int, required=False, default=100, help="Max epochs")
parser.add_argument("--batch_size", type=int, required=False, default=10, help="batchsize for training")
parser.add_argument("--criterion", type=str, required=False, default="CrossEntropy", choices=["CrossEntropy", "ArcFace"], help="Criterion of classification performance")
parser.add_argument("--lr", type=float, required=False, default=0.01, help="learning rate")
parser.add_argument("--momentum", type=float, required=False, default=0.0, help="momentum")
parser.add_argument("--weight_decay", type=float, required=False, default=0.0, help="weigh_decay")
parser.add_argument("--scale", type=float, required=False, default=0.0, help="scale for arcface loss")
parser.add_argument("--margin", type=float, required=False, default=0.0, help="margin for arcface loss")
parser.add_argument("--save_model", type=int, required=False, default=0, help="flag for model saving")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)

    dataset_config = configure_dataset(dataset_name=args.dataset, target=args.target)
    net: Net = load_arcface_model(name=args.model, input_spec=dataset_config["input_spec"], out_dims=dataset_config["out_dims"], pretrained=args.pretrained)
    print(net)
    
    params_config = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "criterion": args.criterion,
        "pretrained": args.pretrained,
        "margin": args.margin,
        "target": args.target,
        "scale": args.scale,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "model_name": args.model,
        "seed": args.seed,
        "api_key_file": os.environ["WANDB_API_KEY_FILE"]
    }
    custom_wandb_init(
        config=params_config,
        project=f"{args.dataset}_verifications",
        strategy="Centralized"
    )
    
    # dataset
    trainset = load_centralized_dataset(dataset_name=args.dataset, train=True, target=args.target)
    testset = load_centralized_dataset(dataset_name=args.dataset, train=False, target=args.target)

    trainloader = DataLoader(trainset, batch_size=params_config["batch_size"], num_workers=2, pin_memory=True, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

    # criterion
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "ArcFace":
        criterion = ArcFaceLoss(s = args.scale, m = args.margin)
    else:
        raise ValueError("Not support criterion {}".format(args.criterion))

    # train loop
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=params_config["lr"], momentum=params_config["momentum"], weight_decay=params_config["weight_decay"])
    for epoch in range(params_config["max_epochs"]):
        net.train()
        for _, data in tqdm(enumerate(trainloader), total=len(trainloader), file=sys.stdout,desc=f"Epoch: {epoch} / {params_config['max_epochs']}", leave=False):
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
        net.eval()
        correct, total, steps, loss = 0,0,0,0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                steps += 1
        wandb.log({"test_loss": loss / steps, "test_acc": correct / total})
    
    if args.save_model:
        save_path = os.path.join(wandb.run.dir, "final_model.pth")
        torch.save(net.to("cpu").state_dict(), save_path)
    
    # visualize_2d_embedded_space(net, num_classes=10, testloader=testloader, save_path=os.path.join(wandb.run.dir, f"{args.criterion}.pdf"))

if __name__ == "__main__":
    main()


