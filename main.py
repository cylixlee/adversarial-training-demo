import torch
from tqdm import tqdm

from src.attacks import AdversarialAttack, PGDAttack
from src.datasets import DatasetProvider, MNISTDatasetProvider
from src.models import MNISTMultiLayerPerceptron, TargetModel


def main() -> None:
    dataset_provider = MNISTDatasetProvider()
    model = MNISTMultiLayerPerceptron()

    evaluate_model(model, dataset_provider)
    evaluate_model(model, dataset_provider, PGDAttack(model))


def evaluate_model(
    model: TargetModel,
    dataset_provider: DatasetProvider,
    attack: AdversarialAttack | None = None,
) -> None:
    model.eval()
    correct = 0
    total = 0
    if attack is None:
        desc = "Test"
        title = "Accuracy"
    else:
        desc = "Adversarial Test"
        title = "Adversarial Accuracy"
    for x, labels in tqdm(dataset_provider.test_set, desc=desc, leave=False):
        if attack is not None:
            x = attack(x, labels)
        prediction = model(x)
        correct += torch.argmax(prediction, dim=1).eq(labels).sum().item()
        total += len(labels)
    print(title, correct / total)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
