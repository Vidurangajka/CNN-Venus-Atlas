import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from src.dataset_loader import get_dataloaders
from src.models.custom_cnn import VolcanoCNN


@torch.no_grad()
def evaluate_on_test(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_preds, all_labels


def plot_confusion_matrix(cm, class_names, out_path):
    """
    cm: 2x2 numpy array
    class_names: list like ['non_volcano', 'volcano']
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test Set)",
    )

    # write numbers in each cell
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataloaders, but we only need test_loader here
    # We set use_weighted_sampler=False because at evaluation time
    # we want to iterate through the test set exactly once, not resample.
    _, _, test_loader, class_to_idx, _ = get_dataloaders(
        data_root="data_processed",
        img_size=32,
        batch_size=256,
        num_workers=2,
        use_weighted_sampler=False
    )

    # class_to_idx is {'non_volcano':0, 'volcano':1},
    # we can invert it for pretty printing
    idx_to_class = {v: k for (k, v) in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    # Recreate the same model architecture
    model = VolcanoCNN(num_classes=2).to(device)

    # Load best weights from training
    best_model_path = Path("results/vgg16/best_model.pth")
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Loss function (not strictly needed for accuracy/precision/recall,
    # but we keep it if you want to report test loss as well)
    criterion = nn.CrossEntropyLoss()

    # Run evaluation
    preds, labels = evaluate_on_test(model, test_loader, device)

    # Basic accuracy
    accuracy = (preds == labels).mean()

    # Precision and recall for volcano class (class index 1)
    precision_volcano = precision_score(
        labels, preds, pos_label=1, zero_division=0)
    recall_volcano = recall_score(labels, preds, pos_label=1, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # Save confusion matrix figure
    figs_dir = Path("results/figs")
    figs_dir.mkdir(parents=True, exist_ok=True)
    cm_path = figs_dir / "confusion_matrix_vgg16.png"
    plot_confusion_matrix(cm, class_names, cm_path)

    # Print results to console so you can copy to report
    print("Test set results (VGG16):")
    print("--------------------------------")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (volc):  {precision_volcano:.4f}")
    print(f"Recall    (volc):  {recall_volcano:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ] :")
    print(cm)
    print(f"Confusion matrix figure saved to: {cm_path}")


if __name__ == "__main__":
    main()
