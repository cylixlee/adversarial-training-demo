import pathlib
from abc import ABC, abstractmethod
from typing import Any, override

import torch
from torch import nn
from transformers import AutoModel

__all__ = ["TargetModel", "MNISTMultiLayerPerceptron"]

MODEL_HOME = pathlib.Path(__file__).parent.parent / "pretrained-models"


class TargetModel(ABC, nn.Module):
    """
    A model accepting a tensor and output another.

    For this project, the target models must be of image-classification, and only receives images.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def _load_remote(name: str) -> Any:
    try:
        print(f"Trying local cache for '{name}'")
        return AutoModel.from_pretrained(name, trust_remote_code=True, cache_dir=MODEL_HOME, local_files_only=True)
    except OSError:
        print(f"Local cache not found, fetching remote for '{name}'")
        return AutoModel.from_pretrained(name, trust_remote_code=True, cache_dir=MODEL_HOME)


class MNISTMultiLayerPerceptron(TargetModel):
    _model: AutoModel

    def __init__(self):
        super().__init__()
        self._model = _load_remote("dacorvo/mnist-mlp")

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
