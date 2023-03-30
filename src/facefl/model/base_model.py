from collections import OrderedDict

import torch
import torch.nn as nn
from flwr.common import NDArrays


class Net(nn.Module):
    """Base class of neural network for federated learning"""

    def get_weights(self) -> NDArrays:
        """
        Get model weights as a list of NumPy ndarrays.
        """
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: NDArrays) -> None:
        """
        Set model weights from a list of NumPy ndarrays.
        """
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )

        self.load_state_dict(state_dict, strict=True)
