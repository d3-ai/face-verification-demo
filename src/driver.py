import sys
from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.base_model import Net

def train(net: Net, trainloader: DataLoader, valloader: DataLoader, epochs: int, lr: float, device: str)->Dict[str, float]:
    print("Starting training...")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in tqdm(trainloader, file=sys.stdout):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")
    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)
    results = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc,
    }
    return results

def test(net: Net, testloader: DataLoader, steps: int = None, device: str = "cpu")->Tuple[float, float]:
    print("Starting evaluation...")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0,0,0.0
    net.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if steps is not None and batch_idx == steps:
                break
        
    if steps is None:
        loss /= len(testloader.dataset)
    else:
        loss /= total
    acc = correct / total
    net.to("cpu")
    return loss, acc