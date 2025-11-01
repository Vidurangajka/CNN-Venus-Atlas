import numpy as np
from pathlib import Path
from src.vread import vread

# point this to one experiment in Chips
BASE = Path("data_raw/package/Chips")

chips_path_base = BASE / "exp_A1_Ctrn"
labels_path_base = BASE / "exp_A1_Ltrn"

chips_arr = vread(chips_path_base)
labels_arr = vread(labels_path_base)

print("chips_arr.shape:", chips_arr.shape, "dtype:", chips_arr.dtype)
print("labels_arr.shape:", labels_arr.shape, "dtype:", labels_arr.dtype)

# Peek at some values
print("labels_arr[0:10]:", labels_arr.flatten()[0:10])
