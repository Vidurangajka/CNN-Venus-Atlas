import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import csv

from src.dataset_loader import get_dataloaders
from torchvision import models


def replace_conv1_with_grayscale(resnet):
    """Replace conv1 (3→64) with 1→64 and initialize weights from RGB by averaging."""
    old = resnet.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        # old.weight: [64, 3, 7, 7] → average across channel dim
        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    resnet.conv1 = new
    return resnet


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_seen = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
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
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * imgs.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += int((preds == labels).sum().item())
        total_seen += labels.size(0)
    return total_loss / total_seen, (total_correct / total_seen if total_seen else 0.0)


def main():
    results_dir = Path("results/resnet18")
    figs_dir = Path("results/figs")
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Use 224×224 for pretrained backbones
    train_loader, val_loader, _, class_to_idx, class_counts = get_dataloaders(
        data_root="data_processed",
        img_size=224,
        batch_size=64,        # larger input; keep batch moderate
        num_workers=2,
        use_weighted_sampler=True
    )
    print("class_to_idx:", class_to_idx)
    if class_counts is not None:
        print("Training class counts:", class_counts)

    # Build model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = replace_conv1_with_grayscale(model)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # Optional partial freezing (uncomment if you want shallow FT)
    # for p in model.layer1.parameters(): p.requires_grad = False

    # Class-weighted loss
    if class_counts is not None:
        non_volcano_ct = float(class_counts[0])
        volcano_ct = float(class_counts[1])
        weights = torch.tensor(
            [1.0, non_volcano_ct / volcano_ct], dtype=torch.float32, device=device)
    else:
        weights = None
    print("class_weights used in loss:", weights)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Adam with small LR for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 12
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

    # Plot curve
    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet18 fine-tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "custom_loss_curve_resnet18.png", dpi=200)
    plt.close()

    # Save CSV
    with open(results_dir / "training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for e, tl, vl, ta, va in zip(history["epoch"], history["train_loss"],
                                     history["val_loss"], history["train_acc"], history["val_acc"]):
            w.writerow([e, tl, vl, ta, va])

    print("Training complete (ResNet18).")
    print("Best model stored at:", best_model_path)
    print("Loss curve stored at: results/figs/custom_loss_curve_resnet18.png")
    print("Training log stored at:", results_dir / "training_log.csv")


if __name__ == "__main__":
    main()