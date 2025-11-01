import numpy as np
import csv
from pathlib import Path
from src.vread import vread


def is_volcano(label_value: int) -> int:
    """
    Map the original labels:
      1 = definitely a volcano
      2 = probably a volcano
      3 = possibly a volcano
      4 = only a pit is visible
      0 = not in ground truth (background)

    We'll merge {1,2,3,4} -> 1 (volcano-like)
    and 0 -> 0 (non-volcano).
    """
    return 1 if label_value >= 1 else 0


def load_one_split(base_dir: Path, prefix: str, split_tag: str):
    """
    base_dir: data_raw/package/Chips
    prefix: e.g. "exp_A1"
    split_tag: "trn" or "tst"

    Returns
    -------
    chips_arr : np.ndarray of shape (N, 15, 15), dtype uint8
    labels_arr : np.ndarray of shape (N,), dtype int32
    """

    chips_base = base_dir / f"{prefix}_C{split_tag}"
    labels_base = base_dir / f"{prefix}_L{split_tag}"

    chips_arr = vread(chips_base)      # shape (N, 225) for your data
    labels_arr = vread(labels_base)    # shape (N, 1)  uint8

    # ensure 2D chips (N, 225)
    if chips_arr.ndim != 2:
        raise ValueError(
            f"Expected 2D chips array, got {chips_arr.shape} for {chips_base}")

    N, flat_len = chips_arr.shape
    side = int(np.sqrt(flat_len))
    if side * side != flat_len:
        raise ValueError(f"Chip length {flat_len} is not a square number")

    chips_arr = chips_arr.reshape(
        N, side, side).astype(np.uint8)  # (N, 15, 15)

    # labels_arr is (N,1). Flatten it to (N,)
    labels_arr = np.array(labels_arr).reshape(-1).astype(np.int32)

    if labels_arr.shape[0] != chips_arr.shape[0]:
        raise ValueError(
            f"Label count {labels_arr.shape[0]} does not match chip count {chips_arr.shape[0]}"
        )

    return chips_arr, labels_arr


def main():
    base_dir = Path("data_raw/package/Chips")
    out_dir = Path("data_intermediate")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chips = []
    all_labels_orig = []
    all_meta = []

    # Find each experiment prefix by scanning for *_Ctrn.sdt
    for sdt_file in base_dir.glob("exp_*_Ctrn.sdt"):
        prefix = sdt_file.stem.replace("_Ctrn", "")  # "exp_A1"

        # Load train and test slices for this experiment
        chips_trn, labels_trn = load_one_split(base_dir, prefix, "trn")
        chips_tst, labels_tst = load_one_split(base_dir, prefix, "tst")

        # Append train entries
        for i in range(labels_trn.shape[0]):
            all_chips.append(chips_trn[i])
            all_labels_orig.append(int(labels_trn[i]))
            all_meta.append((f"{prefix}_trn_{i}", prefix, int(labels_trn[i])))

        # Append test entries
        for i in range(labels_tst.shape[0]):
            all_chips.append(chips_tst[i])
            all_labels_orig.append(int(labels_tst[i]))
            all_meta.append((f"{prefix}_tst_{i}", prefix, int(labels_tst[i])))

    all_chips = np.stack(all_chips, axis=0)           # shape (N, 15, 15)
    all_labels_orig = np.array(all_labels_orig)       # shape (N,)
    all_labels_bin = np.array(
        [is_volcano(x) for x in all_labels_orig],
        dtype=np.int64
    )

    # Save arrays
    np.save(out_dir / "chips_all.npy", all_chips)
    np.save(out_dir / "labels_all.npy", all_labels_bin)

    # meta_all.csv to help you document provenance in the report
    with open(out_dir / "meta_all.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chip_id", "source_experiment",
                        "original_label", "binary_label"])
        for (chip_id, prefix, orig), binlab in zip(all_meta, all_labels_bin):
            writer.writerow([chip_id, prefix, orig, int(binlab)])

    print("Wrote:")
    print(" data_intermediate/chips_all.npy      with shape", all_chips.shape)
    print(" data_intermediate/labels_all.npy     with shape", all_labels_bin.shape)
    print(" data_intermediate/meta_all.csv")


if __name__ == "__main__":
    main()
