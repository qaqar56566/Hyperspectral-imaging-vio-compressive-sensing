#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated TwIST Reconstruction & Pseudo-Color Generation for HS CS
Optimized structure and I/O paths
"""

import os
import sys
import inspect
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_DIR = ROOT_DIR
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from hsi_reconstruction.core.twist import twist
from hsi_reconstruction.core.tv_denoise import tv_denoise_chambolle


DATA_DIR = os.path.join(ROOT_DIR, "npz")
OUT_DIR = os.path.join(ROOT_DIR, "outputs", "reconstruct_v4")


def load_inputs():
    hs = np.load(os.path.join(DATA_DIR, "HS.npz"))["HS"].astype(np.float32)
    A = np.load(os.path.join(DATA_DIR, "measurement_matrix.npz"))["A"].astype(np.float32)
    compressed_img = np.load(os.path.join(DATA_DIR, "compressed_img.npz"))["compressed_img"].astype(np.float32)
    cmf = np.load(os.path.join(DATA_DIR, "cie1931_2deg_400_700_10nm.npz"))

    wavelength_nm = cmf["wavelength_nm"].astype(np.float32).reshape(-1)
    xbar = cmf["xbar"].astype(np.float32).reshape(-1)
    ybar = cmf["ybar"].astype(np.float32).reshape(-1)
    zbar = cmf["zbar"].astype(np.float32).reshape(-1)

    return hs, A, compressed_img, (wavelength_nm, xbar, ybar, zbar)


def normalize_cube_orientation(hs: np.ndarray, compressed_img: np.ndarray) -> np.ndarray:
    h_img, w_img = compressed_img.shape
    if hs.shape[:2] == (h_img, w_img):
        return hs
    if hs.ndim == 3 and hs.shape[1:3] == (h_img, w_img):
        return np.transpose(hs, (1, 2, 0))
    raise ValueError(f"Unexpected HS cube shape: {hs.shape}")


def normalize_measurement_matrix(A: np.ndarray, cube_shape) -> np.ndarray:
    h, w, b = cube_shape
    if A.shape == (h, w, b):
        return A
    if A.shape == (b, h, w):
        return np.transpose(A, (1, 2, 0))
    raise ValueError(f"Unexpected measurement matrix shape: {A.shape}")


def tv_phi_replicate(cube_in: np.ndarray) -> float:
    """TV norm with replicate boundary (vectorized)."""
    cube_in = cube_in.astype(np.float32, copy=False)
    h, w, _ = cube_in.shape
    u_right = cube_in[:, np.r_[1:w, w - 1], :]
    u_down = cube_in[np.r_[1:h, h - 1], :, :]
    dx = u_right - cube_in
    dy = u_down - cube_in
    return float(np.sum(np.sqrt(dx * dx + dy * dy)))


def hs_to_srgb_cie1931(hypercube, wavelength_nm, xbar, ybar, zbar):
    hypercube = hypercube.astype(np.float32, copy=False)
    h, w, b = hypercube.shape

    wavelength_nm = np.asarray(wavelength_nm, dtype=np.float32).reshape(-1)
    xbar = np.asarray(xbar, dtype=np.float32).reshape(-1)
    ybar = np.asarray(ybar, dtype=np.float32).reshape(-1)
    zbar = np.asarray(zbar, dtype=np.float32).reshape(-1)

    if wavelength_nm.size != b:
        raise ValueError("wavelength_nm size mismatch")

    d_lambda = float(np.mean(np.diff(wavelength_nm)))

    spectral = hypercube.reshape(-1, b)
    X = (spectral @ xbar) * d_lambda
    Y = (spectral @ ybar) * d_lambda
    Z = (spectral @ zbar) * d_lambda
    xyz = np.stack([X, Y, Z], axis=1)

    yref = np.percentile(Y, 99)
    if yref > 0:
        xyz = xyz / yref

    M = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ], dtype=np.float32)

    rgb_lin = xyz @ M.T
    rgb_lin = np.maximum(rgb_lin, 0.0)

    a = 0.055
    thr = 0.0031308
    rgb = np.empty_like(rgb_lin)
    mask = rgb_lin <= thr
    rgb[mask] = 12.92 * rgb_lin[mask]
    rgb[~mask] = (1 + a) * np.power(rgb_lin[~mask], 1 / 2.4) - a
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.reshape(h, w, 3)


def main():
    # -------------------- 1) User Settings --------------------
    regularization_tau = 0.01
    max_main_iters = 100
    min_main_iters = 50
    tolerance_stop = 1e-6
    lambda_min_eig = 1e-4
    tv_inner_iters = 20
    enable_plots = True
    plot_after_run = True
    show_numpy_config = True

    rgb_gt_path = os.path.join(ROOT_DIR, "..", "feathers_ms", "feathers_RGB.bmp")

    os.makedirs(OUT_DIR, exist_ok=True)

    if show_numpy_config:
        print("NumPy configuration:")
        np.show_config()

    # -------------------- 2) Load Data --------------------
    hs, A, compressed_img, cmf = load_inputs()
    wavelength_nm, xbar, ybar, zbar = cmf

    hs = normalize_cube_orientation(hs, compressed_img)
    h, w, b = hs.shape
    A = normalize_measurement_matrix(A, hs.shape)

    # -------------------- 3) Operator Normalization C --------------------
    op_norm2_map = np.sum(A * A, axis=2)
    operator_normC = float(np.sqrt(np.max(op_norm2_map)))
    if operator_normC == 0:
        raise ValueError("Operator norm C is zero; check measurement matrix.")
    print(f"Operator Norm C: {operator_normC:.4f}")

    def forward_op(cube_est):
        return np.sum(A * cube_est, axis=2) / operator_normC

    def adjoint_op(meas_in):
        return (A * meas_in[:, :, None]) / operator_normC

    measurement_scaled = compressed_img / operator_normC

    # -------------------- 4) TV Psi / Phi --------------------
    def psi_tv(cube_in, tau_prox):
        tau_prox = float(max(tau_prox, 1e-12))
        lambd = 1.0 / tau_prox
        return tv_denoise_chambolle(cube_in, lambd, n_iter=tv_inner_iters, prefer_numba=True)

    phi_tv = tv_phi_replicate

    # -------------------- 5) Run TwIST --------------------
    init_cube = adjoint_op(measurement_scaled)
    init_cube = psi_tv(init_cube, 0.01 * regularization_tau)

    twist_kwargs = dict(
        y=measurement_scaled,
        A=forward_op,
        tau=regularization_tau,
        AT=adjoint_op,
        psi=psi_tv,
        phi=phi_tv,
        lam1=lambda_min_eig,
        tolA=tolerance_stop,
        maxiter=max_main_iters,
        miniter=min_main_iters,
        stop_criterion=1,
        enforce_monotone=True,
        sparse=False,
        initialization=2,
        verbose=True,
    )

    if "x0" in inspect.signature(twist).parameters:
        twist_kwargs["x0"] = init_cube

    result = twist(**twist_kwargs)

    recon_cube = result["x"].astype(np.float32)
    recon_cube = np.clip(recon_cube, 0.0, 1.0)

    if enable_plots and plot_after_run:
        plt.figure()
        plt.imshow(compressed_img, cmap="gray")
        plt.title("Measurement (Compressed Image)")

        plt.figure()
        plt.plot(result["objective"], "-")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("TwIST objective")
        plt.grid(True)

        band_show = min(15, b) - 1
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(hs[:, :, band_show], cmap="gray")
        plt.title(f"Ground Truth (Band {band_show + 1})")
        plt.subplot(1, 3, 2)
        plt.imshow(recon_cube[:, :, band_show], cmap="gray")
        plt.title(f"Reconstruction (Band {band_show + 1})")
        plt.subplot(1, 3, 3)
        plt.imshow(np.abs(recon_cube[:, :, band_show] - hs[:, :, band_show]), cmap="hot", vmin=0, vmax=0.2)
        plt.colorbar()
        plt.title("Absolute Error")

    # -------------------- 6) Quantitative Evaluation --------------------
    band_mse = np.zeros(b, dtype=np.float32)
    band_psnr = np.zeros(b, dtype=np.float32)
    for band_idx in range(b):
        diff_band = recon_cube[:, :, band_idx] - hs[:, :, band_idx]
        mse_val = float(np.mean(diff_band * diff_band))
        band_mse[band_idx] = mse_val
        band_psnr[band_idx] = 10.0 * np.log10(1.0 / max(mse_val, 1e-12))

    print(f"\nAverage MSE across all bands: {np.mean(band_mse):.6e}")
    print(f"Average PSNR across all bands: {np.mean(band_psnr):.2f} dB")

    if enable_plots and plot_after_run:
        plt.figure(figsize=(9, 3.5))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(1, b + 1), band_mse, "o-", linewidth=1.5)
        plt.xlabel("Band index")
        plt.ylabel("MSE")
        plt.title("Reconstruction MSE")
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(1, b + 1), band_psnr, "r^-", linewidth=1.5)
        plt.xlabel("Band index")
        plt.ylabel("PSNR (dB)")
        plt.title("Reconstruction PSNR")
        plt.grid(True)

    # -------------------- 7) Convert HS -> XYZ -> sRGB --------------------
    rgb_recon = hs_to_srgb_cie1931(recon_cube, wavelength_nm, xbar, ybar, zbar)
    rgb_hs_gt = hs_to_srgb_cie1931(hs, wavelength_nm, xbar, ybar, zbar)

    if os.path.isfile(rgb_gt_path):
        rgb_cam_gt = imageio.imread(rgb_gt_path)
    else:
        rgb_cam_gt = np.zeros((h, w, 3), dtype=np.uint8)
        print(f"Warning: Camera RGB Ground Truth not found at {rgb_gt_path}")

    if enable_plots and plot_after_run:
        plt.figure(figsize=(12, 4.5))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_cam_gt)
        plt.title("Camera RGB (True GT)")
        plt.subplot(1, 3, 2)
        plt.imshow(rgb_hs_gt)
        plt.title("Projected GT Cube (CIE1931 2°)")
        plt.subplot(1, 3, 3)
        plt.imshow(rgb_recon)
        plt.title("Reconstructed Cube (CIE1931 2°)")

    # -------------------- 8) Save Results --------------------
    rgb_out_png = os.path.join(OUT_DIR, f"pseudoColor_sRGB_tau{regularization_tau:.3f}.png")
    imageio.imwrite(rgb_out_png, (rgb_recon * 255).astype(np.uint8))
    print(f"Saved RGB image: {rgb_out_png}")

    out_npz = os.path.join(
        OUT_DIR,
        f"HS_recon_TwIST_tau{regularization_tau:.3f}_tv{tv_inner_iters}.npz",
    )
    np.savez(
        out_npz,
        recon_cube=recon_cube,
        objective=np.asarray(result["objective"], dtype=np.float32),
        times=np.asarray(result["times"], dtype=np.float32),
        band_mse=band_mse,
        band_psnr=band_psnr,
        regularization_tau=regularization_tau,
        tv_inner_iters=tv_inner_iters,
        lambda_min_eig=lambda_min_eig,
        operator_normC=operator_normC,
        rgb_recon=rgb_recon,
    )
    print(f"Saved reconstruction data: {out_npz}")

    if enable_plots and plot_after_run:
        plt.show()


if __name__ == "__main__":
    main()
