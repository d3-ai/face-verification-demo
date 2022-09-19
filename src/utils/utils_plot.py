import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union
from models.metric_learning import ArcFaceResNet, ArcFaceResNetLR

def fig_setup():    
    plt.rcParams['font.size'] = 15
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 5.0
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["legend.edgecolor"] = 'black'
    plt.rcParams["legend.markerscale"] = 10
    # plt.rcParams['font.family'] = ['Arial']

def visualize_2d_embedded_space(net: Union[ArcFaceResNet, ArcFaceResNetLR], num_classes: int, testloader: DataLoader, save_path: str = None):
    net.to("cpu")
    w = F.normalize(net.state_dict()["arcmarginprod.weight"]).numpy()
    net.arcmarginprod = nn.Identity()
    for params in net.parameters():
        params.requires_grad = False
    
    x = []
    y = []
    with torch.no_grad():
        for i, data in enumerate(testloader):
            images, labels = data[0], data[1]
            outputs = F.normalize(net(images))
            for j in range(num_classes):
                features = outputs[torch.where(labels == j)].numpy()
                if i == 0:
                    x.append(features[:,0])
                    y.append(features[:,1])
                else:
                    x[j] = np.concatenate([x[j], features[:,0]])
                    y[j] = np.concatenate([y[j], features[:,1]])
    
    fig_setup()
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(5,5))
    for i in range(num_classes):
        plt.scatter(x[i], y[i], label=f"Identity {i}", s=0.5, color=cmap(i))
        plt.plot([0, w[i,0]], [0, w[i,1]], color=cmap(i))
    plt.xticks(ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    plt.yticks(ticks=[-1.0, -0.5, 0, 0.5, 1.0])
    plt.xlim([-1.05, 1.05])
    plt.ylim([-1.05, 1.05])
    plt.legend(bbox_to_anchor=(1.05,0), loc="lower left", borderaxespad=0)
    if save_path is not None:
        plt.savefig(save_path, format="pdf", bbox_inches='tight')
    plt.close()