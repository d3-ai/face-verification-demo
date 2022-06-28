from typing import Tuple

from models.base_model import Net
from models.tiny_CNN import tinyCNN

def load_model(name: str, input_spec: Tuple[int, int, int], out_dims: int = 10)->Net:
    if name == "tiny_CNN":
        return tinyCNN(input_spec=input_spec, out_dims=out_dims)
    else:
        raise NotImplementedError(f"model {name} is not implemented.")