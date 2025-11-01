import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import csv

from src.dataset_loader import get_dataloaders
from src.models.custom_cnn import VolcanoCNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += labels.size(0)

    return total_loss / total_seen, (total_correct / total_seen if total_seen else 0.0)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += labels.size(0)

    return total_loss / total_seen, (total_correct / total_seen if total_seen else 0.0)


def main():
    results_dir = Path("results/custom_cnn_sgdm")
    figs_dir = Path("results/figs")
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, _, class_to_idx, class_counts = get_dataloaders(
        data_root="data_processed",
        img_size=32,
        batch_size=128,
        num_workers=2,
        use_weighted_sampler=True
    )

    print("class_to_idx:", class_to_idx)
    if class_counts is not None:
        print("Training class counts:", class_counts)

    model = VolcanoCNN(num_classes=2).to(device)

    if class_counts is not None:
        non_volcano_ct = float(class_counts[0])
        volcano_ct = float(class_counts[1])
        weight_non_volcano = 1.0
        weight_volcano = non_volcano_ct / volcano_ct if volcano_ct > 0 else 1.0
        class_weights = torch.tensor([weight_non_volcano, weight_volcano],
                                     dtype=torch.float32, device=device)
        print("class_weights used in loss:", class_weights)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # SGD with momentum for Task 11
    optimizer = optim.SGD(model.parameters(), lr=0.01,
                          momentum=0.9, nesterov=False)

    num_epochs = 20
    history = {"epoch": [], "train_loss": [],
               "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_model_path = results_dir / "best_model.pth"

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} | "
              f"train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Custom CNN training (SGD + Momentum)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "custom_loss_curve_sgdm.png", dpi=200)
    plt.close()

    with open(results_dir / "training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for e, tl, vl, ta, va in zip(history["epoch"], history["train_loss"],
                                     history["val_loss"], history["train_acc"], history["val_acc"]):
            w.writerow([e, tl, vl, ta, va])

    print("Training complete (SGD+Momentum).")
    print("Best model stored at:", best_model_path)
    print("Loss curve stored at: results/figs/custom_loss_curve_sgdm.png")
    print("Training log stored at:", results_dir / "training_log.csv")


if __name__ == "__main__":
    main()
