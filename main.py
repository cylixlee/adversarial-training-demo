import torch
import torchattacks
from torch import nn, optim
from tqdm import tqdm

from src.attacks import AdversarialAttack
from src.datasets import DatasetProvider, MNISTDatasetProvider
from src.models import MNISTMultiLayerPerceptron, TargetModel


def main() -> None:
    dataset_provider = MNISTDatasetProvider()
    model = MNISTMultiLayerPerceptron()
    attack = torchattacks.CW(model)

    evaluate_model(model, dataset_provider)
    evaluate_model(model, dataset_provider, attack)
    adversarial_train(model, dataset_provider, attack)


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


def adversarial_train(model: TargetModel, dataset_provider: DatasetProvider, attack: AdversarialAttack) -> None:
    epoch = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    model.train()
    with tqdm(total=epoch, desc="Epoch", leave=False) as epoch_progress:
        for _ in range(epoch):
            correct_natural = 0
            correct_adversarial = 0
            total = 0
            for x, labels in tqdm(dataset_provider.train_set, desc="Progress", leave=False):
                # type annotations to make IDE happy :)
                x: torch.Tensor
                labels: torch.Tensor
                predictions: torch.Tensor
                loss: torch.Tensor

                # forward propagation (natural)
                x, labels = x.to(device), labels.to(device)
                predictions = model(x)
                correct_natural += torch.argmax(predictions, dim=1).eq(labels).sum().item()

                # forward propagation (adversarial)
                perturbed = attack(x, labels)
                predictions = model(perturbed)
                correct_adversarial += torch.argmax(predictions, dim=1).eq(labels).sum().item()

                # back propagation
                loss = criteria(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # collect statistics
                total += len(labels)
            # update progress
            epoch_progress.set_postfix(
                {
                    "acc_nat": correct_natural / total,
                    "acc_adv": correct_adversarial / total,
                }
            )
            epoch_progress.update()
    print("Train Natural Accuracy:", correct_natural / total)
    print("Train Adversarial Accuracy:", correct_adversarial / total)

    model.eval()
    correct_natural = 0
    correct_adversarial = 0
    total = 0
    for x, labels in tqdm(dataset_provider.test_set, desc="Test", leave=False):
        x, labels = x.to(device), labels.to(device)
        predictions = model(x)
        correct_natural += torch.argmax(predictions, dim=1).eq(labels).sum().item()

        perturbed = attack(x, labels)
        predictions = model(perturbed)
        correct_adversarial += torch.argmax(predictions, dim=1).eq(labels).sum().item()
        total += len(labels)
    print("Test Natural Accuracy:", correct_natural / total)
    print("Test Adversarial Accuracy:", correct_adversarial / total)


# Guideline recommended Main Guard
if __name__ == "__main__":
    main()
