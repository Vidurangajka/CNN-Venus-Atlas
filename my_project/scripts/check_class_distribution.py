"""
check_class_distribution.py
-----------------------------------
Quick utility to print how many volcano vs non-volcano samples
are present in the intermediate dataset.
"""

import numpy as np


def main():
    labels_path = "data_intermediate/labels_all.npy"

    labels = np.load(labels_path)
    unique, counts = np.unique(labels, return_counts=True)

    print("\nClass distribution in data_intermediate/labels_all.npy:")
    print("--------------------------------------------------------")
    for u, c in zip(unique, counts):
        pct = 100 * c / len(labels)
        name = "non-volcano" if u == 0 else "volcano"
        print(f"  label {u:>1} ({name:>11}): {c:>7} samples ({pct:5.2f}%)")
    print("--------------------------------------------------------")
    print(f"  Total: {len(labels)} samples\n")


if __name__ == "__main__":
    main()
