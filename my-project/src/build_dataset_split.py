import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import csv


def save_chip_png(chip_array, out_path):
    img = Image.fromarray(chip_array.astype(np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    inter_dir = Path("data_intermediate")
    proc_dir = Path("data_processed")

    chips_all = np.load(inter_dir / "chips_all.npy")      # (N, 15, 15)
    labels_all = np.load(inter_dir / "labels_all.npy")    # (N,) 0 or 1

    N = chips_all.shape[0]
    idx_all = np.arange(N)

    # Split 70 / 30
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        idx_all,
        labels_all,
        test_size=0.30,
        stratify=labels_all,
        random_state=42
    )

    # Split the remaining 30 into 15 / 15 (so 50/50 of temp)
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=43
    )

    # helper function to export one split
    def export_split(split_name, indices, labels):
        rows = []
        for counter, (i, lab) in enumerate(zip(indices, labels)):
            chip = chips_all[i]  # (15, 15)
            class_dir = "volcano" if lab == 1 else "non_volcano"
            filename = f"chip_{counter:05d}.png"
            out_path = proc_dir / split_name / class_dir / filename
            save_chip_png(chip, out_path)
            rows.append([filename, str(out_path), int(lab)])
        return rows

    train_rows = export_split("train", idx_train, y_train)
    val_rows = export_split("val",   idx_val,   y_val)
    test_rows = export_split("test",  idx_test,  y_test)

    # Write CSVs for traceability
    with open(proc_dir / "train_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "path", "label"])
        w.writerows(train_rows)

    with open(proc_dir / "val_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "path", "label"])
        w.writerows(val_rows)

    with open(proc_dir / "test_split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "path", "label"])
        w.writerows(test_rows)

    print("Export complete.")
    print("Check data_processed/train, val, test folders.")


if __name__ == "__main__":
    main()
