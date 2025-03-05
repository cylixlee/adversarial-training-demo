from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import override

import torchattacks

__all__ = ["AdversarialAttack", "FGSMAttack"]


class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks.

    There's a lot of attack implementations on GitHub or other repositories. However, for adversarial training, an
    attack is like a black box -- ``(x, y)`` in, ``x_adv`` out.

    Gradiend-based adversarial attacks often needs the model or its parameters for computing gradients. Those **state
    variables** should be kept as instance members of a specific ``AdversarialAttack`` subclass, not the parameters of
    the instance method ``perform``.
    """

    @abstractmethod
    def perform(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform attack on ``x``."""
        pass


class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method.
    """

    _attack: torchattacks.FGSM

    def __init__(self, model: nn.Module) -> None:
        self._attack = torchattacks.FGSM(model)

    @override
    def perform(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self._attack(x, labels)
