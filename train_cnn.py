"""Train SmallCNN on Fashion-MNIST on the CPU and save the test metrics."""
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dl_lab.cnn import SmallCNN

EPOCHS, BATCH_SIZE, SEED = 5, 128, 42


@torch.no_grad()
def evaluate(model, loader, num_classes=10):
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, labels in loader:
        predictions = model(images).argmax(dim=1)
        for true, predicted in zip(labels, predictions):
            confusion[true, predicted] += 1
    return confusion


def main():
    torch.manual_seed(SEED)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_set = datasets.FashionMNIST("data", train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    model = SmallCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    started, history = time.time(), []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        confusion = evaluate(model, test_loader)
        accuracy = confusion.diag().sum().item() / confusion.sum().item()
        row = {"epoch": epoch, "train_loss": round(total_loss / len(train_set), 4),
               "test_accuracy": round(accuracy, 4)}
        history.append(row)
        print(row, flush=True)

    per_class = confusion.diag() / confusion.sum(dim=1)
    report = {
        "dataset": "Fashion-MNIST",
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": EPOCHS,
        "seed": SEED,
        "device": "cpu",
        "train_seconds": round(time.time() - started),
        "final_test_accuracy": history[-1]["test_accuracy"],
        "per_class_accuracy": {c: round(a.item(), 4) for c, a in zip(train_set.classes, per_class)},
        "history": history,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/cnn.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
