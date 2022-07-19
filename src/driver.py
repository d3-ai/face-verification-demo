import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base_model import Net

# type annotation
from typing import Dict
from common.typing import Scalar


def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    use_tqdm: bool=False,)->None:
    loader = tqdm(trainloader, file=sys.stdout) if use_tqdm else trainloader
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")

def test(
    net: Net,
    testloader: DataLoader,
    steps: int = None,
    device: str = "cpu")->Dict[str, Scalar]:
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, steps, loss = 0,0,0,0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
    loss /= steps
    acc = correct / total
    net.to("cpu")
    return {"loss": loss, "acc": acc}