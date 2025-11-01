"""
vread.py

Python port of the original MATLAB vread.m provided in the
"Volcanoes on Venus - JARtool experiment" dataset.

The dataset uses a "VIEW" format:
    <name>.spr  : ASCII header
    <name>.sdt  : binary raster data

In MATLAB, usage is:
    A = vread('img1');

This reads img1.spr to determine array shape and dtype,
then reads img1.sdt and returns a 2-D matrix A.

This Python module provides the same behavior.

You will mainly use this to read:
 - raw Venus SAR images in package/Images/
 - chip stacks and label arrays in package/Chips/

Author (Python port): you
Original MATLAB author: M.C. Burl, JPL (1992–1999)
"""

import numpy as np
from pathlib import Path


class VReadFormatError(Exception):
    """Raised if the VIEW format header is invalid or unsupported."""
    pass


def _parse_spr(spr_path: Path):
    """
    Parse the .spr header file.

    The original MATLAB code (vread.m) does:

        ndim = fscanf(idp, '%d', 1);
        nc   = fscanf(idp, '%d', 1);
        junk = fscanf(idp, '%f', 1);
        junk = fscanf(idp, '%f', 1);
        nr   = fscanf(idp, '%d', 1);
        junk = fscanf(idp, '%f', 1);
        junk = fscanf(idp, '%f', 1);
        type = fscanf(idp, '%d', 1);

    Meaning of fields:
        ndim : number of dimensions (must be 2 for images)
        nc   : number of columns
        (skip two floats)
        nr   : number of rows
        (skip two floats)
        type : numeric code for data type

    The file is whitespace-separated, not line-structured, so we just
    read all tokens and consume them in order.

    Returns
    -------
    nr : int
        number of rows
    nc : int
        number of columns
    dtype : numpy dtype
        dtype to use when reading the .sdt file
    """

    if not spr_path.exists():
        raise FileNotFoundError(f"Header file not found: {spr_path}")

    # Read entire .spr and split on whitespace
    tokens = spr_path.read_text().split()

    # We expect at least 8 numeric tokens in the exact order.
    # ndim, nc, junk, junk, nr, junk, junk, type_code
    if len(tokens) < 8:
        raise VReadFormatError(
            f"Unexpected .spr format in {spr_path}. "
            f"Expected >= 8 numeric tokens, got {len(tokens)}."
        )

    idx = 0
    ndim = int(tokens[idx])
    idx += 1
    if ndim != 2:
        raise VReadFormatError(
            f"Only 2-D data supported, got ndim={ndim} in {spr_path}"
        )

    nc = int(tokens[idx])
    idx += 1

    # skip two floats
    idx += 1
    idx += 1

    nr = int(tokens[idx])
    idx += 1

    # skip two more floats
    idx += 1
    idx += 1

    type_code = int(tokens[idx])
    idx += 1

    # Map type codes from MATLAB vread.m to numpy dtypes
    #   0 -> 'unsigned char'  -> np.uint8
    #   2 -> 'int'            -> assume 32-bit signed int
    #   3 -> 'float'          -> np.float32
    #   5 -> 'double'         -> np.float64
    if type_code == 0:
        dtype = np.uint8
    elif type_code == 2:
        dtype = np.int32
    elif type_code == 3:
        dtype = np.float32
    elif type_code == 5:
        dtype = np.float64
    else:
        raise VReadFormatError(
            f"Unrecognized data type code {type_code} in {spr_path}"
        )

    return nr, nc, dtype


def vread(filename: str | Path) -> np.ndarray:
    """
    Read a VIEW-format pair (<filename>.spr, <filename>.sdt) into a numpy array.

    Parameters
    ----------
    filename : str or Path
        Base path without extension.
        Example:
            "data_raw/package/Images/img1"
        The function will open:
            "data_raw/package/Images/img1.spr"
            "data_raw/package/Images/img1.sdt"

    Returns
    -------
    A : np.ndarray
        2D numpy array with shape (nr, nc),
        where nr = rows and nc = cols.
        dtype is determined from the VIEW header.
    """

    base = Path(filename)
    spr_path = base.with_suffix(".spr")
    sdt_path = base.with_suffix(".sdt")

    # Parse header (.spr)
    nr, nc, dtype = _parse_spr(spr_path)

    # Read binary data (.sdt)
    if not sdt_path.exists():
        raise FileNotFoundError(f"Data file not found: {sdt_path}")

    # We expect nr * nc elements of that dtype
    count_expected = nr * nc
    with sdt_path.open("rb") as fbin:
        raw = np.fromfile(fbin, dtype=dtype, count=count_expected)

    if raw.size != count_expected:
        raise VReadFormatError(
            f"File size mismatch for {sdt_path}: "
            f"expected {count_expected} elements, got {raw.size}."
        )

    # MATLAB code:
    #   A = (fread(idd, [nc, nr], precision))';
    # Explanation:
    #   fread(...,[nc,nr]) returns (nc x nr) in column-major,
    #   then MATLAB transposes -> (nr x nc).
    #
    # In NumPy we read flat, then reshape row-major to (nr, nc),
    # which matches that final (nr x nc).
    A = raw.reshape((nr, nc))

    return A


# Optional quick convenience if you want to test interactively:
if __name__ == "__main__":
    # Example usage / sanity check:
    # Adjust this path to one of your actual files, e.g.
    #   python vread.py data_raw/package/Images/img1
    import sys
    if len(sys.argv) != 2:
        print("Usage: python vread.py <basepath_without_extension>")
        print("Example:")
        print("   python vread.py data_raw/package/Images/img1")
        raise SystemExit(1)

    arr = vread(sys.argv[1])
    print("Array shape :", arr.shape)
    print("Array dtype :", arr.dtype)
    print("Min / Max   :", arr.min(), "/", arr.max())
