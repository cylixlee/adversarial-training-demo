import torch
import torchattacks
from tqdm import tqdm

from src.attacks import AdversarialAttack
from src.datasets import DatasetProvider, MNISTDatasetProvider
from src.models import MNISTMultiLayerPerceptron, TargetModel


def main() -> None:
    dataset_provider = MNISTDatasetProvider()
    model = MNISTMultiLayerPerceptron()

    evaluate_model(model, dataset_provider)
    evaluate_model(model, dataset_provider, torchattacks.BIM(model, eps=64 / 256, steps=40))


def evaluate_model(model: TargetModel, dataset_provider: DatasetProvider, attack: AdversarialAttack = None) -> None:
    model.eval()
    correct = 0
    total = 0
    if attack is None:
        description = "Test"
        title = "Accuracy"
    else:
        description = "Adversarial Test"
        title = "Adversarial Accuracy"

    for x, labels in tqdm(dataset_provider.test_set, desc=description, leave=False):
        if attack is not None:
            x = attack(x, labels)
        prediction = model(x)
        correct += torch.argmax(prediction, dim=1).eq(labels).sum().item()
        total += len(labels)
    print(title, correct / total)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
