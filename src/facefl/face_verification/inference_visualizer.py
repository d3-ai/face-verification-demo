import os
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.base_model import Net
from torch.utils.data import DataLoader
from utils.utils_dataset import load_centralized_dataset, load_federated_dataset
from utils.utils_model import load_arcface_model
from utils.utils_plot import fig_setup


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    # client conf
    cid = 9
    threshold = 0.5
    device = "cpu"
    net: Net = load_arcface_model(
        name="GNResNet18", input_spec=(3, 112, 112), out_dims=1, pretrained="CelebA"
    )

    # selfset = load_federated_dataset(dataset_name="CelebA", id = str(cid), train=False, target="small")
    selfset = load_federated_dataset(
        dataset_name="usbcam", id=str(cid), train=False, target="small"
    )
    selfloader = DataLoader(selfset, batch_size=1, shuffle=False)
    testset = load_centralized_dataset(
        dataset_name="CelebA", train=False, target="small"
    )
    testloader = DataLoader(testset, batch_size=6, shuffle=False)

    # figure conf
    w = torch.zeros((10, 512))
    row = 2
    col = 6

    j = 1
    while True:
        load_path = f"./tmp/model_{j}.pth"
        if os.path.isfile(load_path):
            sleep(1)
            i = 0
            state_dict = torch.load(load_path)
            net.load_state_dict(state_dict=state_dict)
            net.to(device)
            net.eval()
            plt.close()
            plt.figure(figsize=(10, 10))
            fig_setup()
            for images, labels in selfloader:
                i += 1
                outputs = net(images)
                plt.subplot(row, col, i)
                plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
                if outputs >= threshold:
                    plt.title(str(float(outputs[0]))[:7], color="green")
                else:
                    plt.title(str(float(outputs[0]))[:7], color="red")

                plt.axis("off")
            for images, labels in testloader:
                i += 1
                outputs = net(images)
                plt.subplot(row, col, i)
                plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
                if outputs[0] < threshold:
                    plt.title(str(float(outputs[0]))[:7], color="green")
                else:
                    plt.title(str(float(outputs[0]))[:7], color="red")
                plt.axis("off")
                if i == 12:
                    break
            plt.suptitle(f"FL round {j}")
            plt.pause(10)
            j += 1
        else:
            sleep(10)
