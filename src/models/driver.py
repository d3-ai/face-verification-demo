import sys
from typing import Dict

import torch
import torch.nn as nn
from flwr.common.typing import Scalar
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base_model import Net


def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    criterion: torch.nn.modules.Module = nn.CrossEntropyLoss(),
    device: str = "cpu",
    use_tqdm: bool = False,
) -> None:
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    net.train()
    if use_tqdm:
        for epoch in range(epochs):
            for _, data in tqdm(
                enumerate(trainloader),
                total=len(trainloader),
                file=sys.stdout,
                desc=f"[Epoch: {epoch}/ {epochs}]",
                leave=False,
            ):
                images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    else:
        for _ in range(epochs):
            for images, labels in trainloader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
    # net.to("cpu")


def test(net: Net, testloader: DataLoader, steps: int = None, device: str = "cpu") -> Dict[str, Scalar]:
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, steps, loss = 0, 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += float(criterion(outputs, labels).item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            steps += 1
    loss /= steps
    acc = correct / total
    net.to("cpu")
    return {"loss": loss, "acc": acc}
