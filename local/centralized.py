import argparse
import random
import numpy as np
import torch
from utils.utils_dataset import load_dataset
from torch.utils.data import DataLoader
from driver import train, test
from utils.utils_model import load_model

parser = argparse.ArgumentParser("Simulation: Centralized learning.")
parser.add_argument("--dataset", type=str, required=False, choices=["CIFAR10", "CelebA"], default="CIFAR10", help="dataset name for FL training")
parser.add_argument("--model", type=str, required=False, choices=["tinyCNN", "ResNet18"], default="tinyCNN", help="model name for Centralized training")
parser.add_argument("--seed", type=int, required=False, default=1234, help="Random seed")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    
    model = load_model(name=args.model, input_spec=(3,32,32), out_dims=10, pretrained=True)

    batch_size = 256
    epochs = 10
    lr = 0.001

    trainset = load_dataset(name=args.dataset, train=True)
    testset = load_dataset(name=args.dataset, train=False)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=100)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = train(model, trainloader, epochs=epochs, lr=lr, device=device)
    print(res)
    test_res = test(model, testloader,)
    print(test_res)
    return test_res[1]

if __name__ == "__main__":
    acc = main()
    print(acc)


