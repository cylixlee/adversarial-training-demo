from torch import nn
import torch

__all__ = ["MNISTMultiLayerPerceptron"]


class MNISTMultiLayerPerceptron(nn.Module):
    """
    A Multi-layer perceptron specifically designed for MNIST.

    Note that native MNIST tensors are of shape `[1, 28, 28]` (C x H x W).
    """

    def __init__(self) -> None:
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.Linear(1024, 10),
            nn.Softmax(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
