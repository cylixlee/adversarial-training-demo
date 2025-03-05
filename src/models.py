from abc import ABC, abstractmethod
from typing import override

import torch
from torch import nn
from transformers import AutoModel

__all__ = ["TargetModel", "MNISTTargetedMLP"]


class TargetModel(ABC, nn.Module):
    """
    A model accepting a tensor and output another.

    For this project, the target models must be of image-classification, and only receives images.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class MNISTTargetedMLP(TargetModel):
    _model: AutoModel

    def __init__(self):
        super().__init__()
        self._model = AutoModel.from_pretrained("dacorvo/mnist-mlp", trust_remote_code=True)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
