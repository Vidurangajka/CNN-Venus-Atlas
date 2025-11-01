import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import csv

from src.dataset_loader import get_dataloaders
from torchvision import models


def replace_first_conv_vgg16(vgg):
    """Replace first conv (3→64) with 1→64 and initialize from RGB average."""
    first = vgg.features[0]
    assert isinstance(first, nn.Conv2d)
    new = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                    stride=first.stride, padding=first.padding, bias=first.bias is not None)
    with torch.no_grad():
        # first.weight: [64, 3, 3, 3] → average across channel dim
        new.weight.copy_(first.weight.mean(dim=1, keepdim=True))
        if first.bias is not None:
            new.bias.copy_(first.bias)
    vgg.features[0] = new
    return vgg


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
    results_dir = Path("results/vgg16")
    figs_dir = Path("results/figs")
    results_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Use 224×224 for pretrained backbones
    train_loader, val_loader, _, class_to_idx, class_counts = get_dataloaders(
        data_root="data_processed",
        img_size=224,
        batch_size=32,        # VGG16 is heavier; keep batch smaller
        num_workers=2,
        use_weighted_sampler=True
    )
    print("class_to_idx:", class_to_idx)
    if class_counts is not None:
        print("Training class counts:", class_counts)

    # Build model
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = replace_first_conv_vgg16(model)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    model = model.to(device)

    # Optional partial freezing (uncomment if you want shallow FT)
    # for p in model.features[:10].parameters(): p.requires_grad = False

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
    plt.title("VGG16 fine-tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "custom_loss_curve_vgg16.png", dpi=200)
    plt.close()

    # Save CSV
    with open(results_dir / "training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        for e, tl, vl, ta, va in zip(history["epoch"], history["train_loss"],
                                     history["val_loss"], history["train_acc"], history["val_acc"]):
            w.writerow([e, tl, vl, ta, va])

    print("Training complete (VGG16).")
    print("Best model stored at:", best_model_path)
    print("Loss curve stored at: results/figs/custom_loss_curve_vgg16.png")
    print("Training log stored at:", results_dir / "training_log.csv")


if __name__ == "__main__":
    main()
