#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert MATLAB .mat files to .npz for reconstruction_v4.py"""

import os
import numpy as np
import h5py
from scipy.io import loadmat


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "npz"))


def squeeze_arr(x):
    return np.asarray(x).squeeze()


def load_mat_var(path: str, key: str) -> np.ndarray:
    """Load variable from .mat, supporting v7.3 via h5py."""
    try:
        return loadmat(path)[key]
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            return np.array(f[key][()])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    hs = load_mat_var("HS.mat", "HS")
    A = load_mat_var("measurement_matrix.mat", "A")
    compressed_img = load_mat_var("compressed_img.mat", "compressed_img")

    try:
        cmf = loadmat("cie1931_2deg_400_700_10nm.mat")
        wavelength_nm = squeeze_arr(cmf["wavelength_nm"])
        xbar = squeeze_arr(cmf["xbar"])
        ybar = squeeze_arr(cmf["ybar"])
        zbar = squeeze_arr(cmf["zbar"])
    except NotImplementedError:
        with h5py.File("cie1931_2deg_400_700_10nm.mat", "r") as f:
            wavelength_nm = squeeze_arr(f["wavelength_nm"][()])
            xbar = squeeze_arr(f["xbar"][()])
            ybar = squeeze_arr(f["ybar"][()])
            zbar = squeeze_arr(f["zbar"][()])

    np.savez(os.path.join(OUT_DIR, "HS.npz"), HS=hs)
    np.savez(os.path.join(OUT_DIR, "measurement_matrix.npz"), A=A)
    np.savez(os.path.join(OUT_DIR, "compressed_img.npz"), compressed_img=compressed_img)
    np.savez(
        os.path.join(OUT_DIR, "cie1931_2deg_400_700_10nm.npz"),
        wavelength_nm=wavelength_nm,
        xbar=xbar,
        ybar=ybar,
        zbar=zbar,
    )

    print(f"Converted .mat files to .npz in: {OUT_DIR}")


if __name__ == "__main__":
    main()
