import pathlib
from abc import ABC, abstractmethod
from typing_extensions import override

from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

__all__ = [
    "DATA_ROOT",
    "DatasetProvider",
    "MNISTDatasetProvider",
]

DATA_ROOT = pathlib.Path(__file__).parent.parent / "data"
"""
The filesystem root for storing downloaded datasets.

This is a ``pathlib.Path`` object pointing to the `data` directory at project root. However, this could be changed to
another ``pathlib.Path`` object (or a ``str`` if a ``DatasetProvider`` supports) at any time earlier than a
``DatasetProvider`` initializes.
"""


class DatasetProvider(ABC):
    """
    Abstract base class for dataset providers.

    User-defined datasets can be introduced by ``torch.utils.data.Dataset``. However, in practice, ``DataLoader`` is a
    better choice than raw ``Dataset``s, so a dataset provider in this project needs to provide two ``DataLoader``s for
    training and testing respectively.
    """

    @property
    @abstractmethod
    def train_set(self) -> DataLoader:
        """Returns a ``DataLoader`` for training."""
        pass

    @property
    @abstractmethod
    def test_set(self) -> DataLoader:
        """Returns a ``DataLoader`` for testing."""
        pass


class MNISTDatasetProvider(DatasetProvider):
    """A dataset provider that provides MNIST datasets."""

    transform = transforms.ToTensor()
    """
    After this transform, MNIST dataset contains tensors of shape ``[1, 28, 28]`` (C x H x W).
    """

    _train_set: DataLoader
    _test_set: DataLoader

    def __init__(self):
        train = MNIST(DATA_ROOT, train=True, download=True, transform=self.transform)
        test = MNIST(DATA_ROOT, train=False, download=True, transform=self.transform)
        self._train_set = DataLoader(train, batch_size=32, shuffle=True)
        self._test_set = DataLoader(test, batch_size=32, shuffle=True)

    @property
    @override
    def train_set(self) -> DataLoader:
        return self._train_set

    @property
    @override
    def test_set(self) -> DataLoader:
        return self._test_set
