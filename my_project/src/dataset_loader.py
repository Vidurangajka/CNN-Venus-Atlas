import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


def get_transforms(img_size=32):
    """
    Returns transform pipelines for train/val/test.

    We keep transforms simple and deterministic here.
    You can optionally add augmentation (random flip, small rotation)
    later if you want.
    """
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # grayscale -> shape (1,H,W), values [0,1]
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    return train_tf, eval_tf


def _make_weighted_sampler(image_folder_dataset):
    """
    Build a WeightedRandomSampler so the minority class (volcano)
    appears more often during training.

    image_folder_dataset.samples is a list of (filepath, class_index).
    """
    targets = [
        cls for (_, cls) in image_folder_dataset.samples]  # list of ints
    targets = np.array(targets)

    class_counts = np.bincount(targets)
    # class_counts[0] = number of non_volcano
    # class_counts[1] = number of volcano

    # inverse frequency
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, class_counts


def get_dataloaders(
    data_root="data_processed",
    img_size=32,
    batch_size=128,
    num_workers=2,
    use_weighted_sampler=True
):
    """
    Build PyTorch DataLoaders for train, val, and test splits.

    Returns:
        train_loader
        val_loader
        test_loader
        class_to_idx   (dict: {'non_volcano':0, 'volcano':1})
        class_counts   (np.array of length 2, counts in the TRAIN set)
    """

    train_tf, eval_tf = get_transforms(img_size)

    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{data_root}/val",   transform=eval_tf)
    test_ds = datasets.ImageFolder(f"{data_root}/test",  transform=eval_tf)

    # build sampler for train to address imbalance
    if use_weighted_sampler:
        sampler, class_counts = _make_weighted_sampler(train_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        class_counts = None
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_ds.class_to_idx, class_counts
