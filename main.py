import torch
from tqdm import tqdm

from src.datasets import MNISTDatasetProvider
from src.models import MNISTMultiLayerPerceptron


def main() -> None:
    dataset_provider = MNISTDatasetProvider()
    model = MNISTMultiLayerPerceptron()

    model.eval()
    correct = 0
    total = 0
    for x, labels in tqdm(dataset_provider.test_set, desc="Test", leave=False):
        prediction = model(x)
        correct += torch.argmax(prediction, dim=1).eq(labels).sum().item()
        total += len(labels)
    print("accuracy:", correct / total)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
