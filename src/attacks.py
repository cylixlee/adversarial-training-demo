from abc import ABC, abstractmethod
from typing import Protocol

import torch

__all__ = ["AdversarialAttack", "AdversarialAttackWrapper"]


class AdversarialAttack(Protocol):
    """
    Duck-type interface (protocol) for adversarial attacks.

    There's a lot of attack implementations on GitHub or other repositories. However, for adversarial training, an
    attack is like a black box -- ``(x, y)`` in, ``x_adv`` out.

    Gradient-based adversarial attacks often needs the model or its parameters for computing gradients. Those **state
    variables** should be kept as instance members of a specific subclass, not the parameters of the actual attack
    method.
    """

    def __call__(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pass


class AdversarialAttackWrapper(ABC):
    """Wrapper class matching the ``AdversarialAttack`` protocol."""

    @abstractmethod
    def perform(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform attack on ``x``."""
        pass

    def __call__(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.perform(x, labels)
