import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from utils.utils_plot import fig_setup


def clientwise_historam(load_dir: str, num_clients: int = 10):
    load_path = Path(load_dir) / "metrics" / "timestamps_federated.json"
    with open(load_path, "r") as f:
        res = json.load(f)

    comm_res = []
    comp_res = []
    comm_err = []
    comp_err = []
    for cid, val in res.items():
        comm = np.array(val["comm"])
        comm_res.append(comm[1:, 1].mean())
        comm_err.append(comm[1:, 1].std())
        comp = np.array(val["comp"])
        comp_res.append(comp[1:, 1].mean())
        comp_err.append(comp[1:, 1].std())

    # plot histogram
    labels = [f"D{cid}" for cid in range(num_clients)]
    left = np.arange(len(labels))
    print(left)
    width = 0.3
    fig_setup()
    plt.bar(left, comm_res, yerr=comm_err, color="r", width=width, align="center")
    plt.bar(
        left + width, comp_res, yerr=comp_err, color="b", width=width, align="center"
    )
    plt.xticks(left + width / 2, labels)
    plt.legend(
        labels=["comm", "comp"], loc="lower center", bbox_to_anchor=(0.5, 0.8), ncol=2
    )
    plt.savefig("hoge.png")


if __name__ == "__main__":
    clientwise_historam(
        load_dir="./exp/CelebA/FedAvg_GNResNet18/small/run_202210061745"
    )
