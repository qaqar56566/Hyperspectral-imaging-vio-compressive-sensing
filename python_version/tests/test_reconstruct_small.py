"""
Small-scale hyperspectral reconstruction validation with visualization.

This script mirrors the MATLAB reconstruct_v3.m flow on a cropped sub-cube,
and adds visualization so the reconstruction process and result are easy to inspect.

Pass criterion:
    mean reconstruction PSNR > 25 dB
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsi_reconstruction.core.twist import twist
from hsi_reconstruction.core.tv_denoise import tv_denoise_chambolle
from hsi_reconstruction.core.tv_norm import tv_norm_iso


DATA_ROOT = ROOT.parent
HS_PATH = DATA_ROOT / 'HS.mat'
A_PATH = DATA_ROOT / 'measurement_matrix.mat'
COMPRESSED_PATH = DATA_ROOT / 'compressed_img.mat'

VIS_DIR = ROOT / 'outputs'
VIS_PATH = VIS_DIR / 'test_reconstruct_small_visualization.png'


def load_hdf5_dataset(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, 'r') as handle:
        data = handle[key][()]
    return np.asarray(data)


def load_measurement_image(path: Path) -> np.ndarray:
    data = loadmat(path)
    if 'compressed_img' not in data:
        raise KeyError('compressed_img not found in compressed_img.mat')
    return np.asarray(data['compressed_img'], dtype=np.float32)


def calc_psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return 10.0 * np.log10((data_range * data_range) / (mse + 1e-12))


def print_stage(title: str) -> None:
    print('\n' + '-' * 72)
    print(title)
    print('-' * 72)


def save_visualization(
    measurement: np.ndarray,
    compressed_crop_scaled: np.ndarray,
    gt_cube: np.ndarray,
    recon_cube: np.ndarray,
    psnr_band: np.ndarray,
    objective: np.ndarray,
    mse_history: np.ndarray | None,
    mean_psnr: float,
) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    show_band = int(np.argmax(psnr_band))
    gt_band = gt_cube[show_band]
    recon_band = recon_cube[show_band]
    abs_err = np.abs(recon_band - gt_band)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f'Small-scale HSI Reconstruction Test | Mean PSNR = {mean_psnr:.2f} dB',
        fontsize=14,
        fontweight='bold',
    )

    im0 = axes[0, 0].imshow(measurement, cmap='gray')
    axes[0, 0].set_title('Synthesized Measurement (crop)')
    axes[0, 0].axis('off')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(compressed_crop_scaled, cmap='gray')
    axes[0, 1].set_title('Original Compressed (crop, scaled)')
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    diff_meas = np.abs(measurement - compressed_crop_scaled)
    im2 = axes[0, 2].imshow(diff_meas, cmap='hot')
    axes[0, 2].set_title('Measurement Difference |synth - real|')
    axes[0, 2].axis('off')
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    x = np.arange(1, len(psnr_band) + 1)
    axes[1, 0].bar(x, psnr_band, color='#1f77b4', alpha=0.85)
    axes[1, 0].axhline(25.0, color='red', linestyle='--', linewidth=1.5, label='threshold 25 dB')
    axes[1, 0].set_title('PSNR per band')
    axes[1, 0].set_xlabel('Band index')
    axes[1, 0].set_ylabel('PSNR (dB)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(objective, linewidth=2.0, color='#2ca02c')
    axes[1, 1].set_title('TwIST objective curve')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Objective')
    axes[1, 1].grid(alpha=0.2)

    if mse_history is not None and len(mse_history) > 0:
        axes[1, 2].plot(mse_history, linewidth=2.0, color='#ff7f0e')
        axes[1, 2].set_title('MSE history (vs GT)')
        axes[1, 2].set_xlabel('Iteration')
        axes[1, 2].set_ylabel('MSE')
        axes[1, 2].grid(alpha=0.2)
    else:
        axes[1, 2].text(0.5, 0.5, 'No MSE history', ha='center', va='center')
        axes[1, 2].set_title('MSE history (vs GT)')
        axes[1, 2].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(VIS_PATH, dpi=160)
    plt.close(fig)

    detail_path = VIS_DIR / 'test_reconstruct_small_band_detail.png'
    fig2, ax2 = plt.subplots(1, 3, figsize=(14, 4))
    fig2.suptitle(
        f'Band Detail (band {show_band + 1}) | PSNR {psnr_band[show_band]:.2f} dB',
        fontsize=13,
        fontweight='bold',
    )

    im_gt = ax2[0].imshow(gt_band, cmap='viridis', vmin=0.0, vmax=1.0)
    ax2[0].set_title('Ground truth')
    ax2[0].axis('off')
    fig2.colorbar(im_gt, ax=ax2[0], fraction=0.046, pad=0.04)

    im_recon = ax2[1].imshow(recon_band, cmap='viridis', vmin=0.0, vmax=1.0)
    ax2[1].set_title('Reconstruction')
    ax2[1].axis('off')
    fig2.colorbar(im_recon, ax=ax2[1], fraction=0.046, pad=0.04)

    im_err = ax2[2].imshow(abs_err, cmap='magma')
    ax2[2].set_title('Absolute error')
    ax2[2].axis('off')
    fig2.colorbar(im_err, ax=ax2[2], fraction=0.046, pad=0.04)

    fig2.tight_layout(rect=[0, 0, 1, 0.92])
    fig2.savefig(detail_path, dpi=160)
    plt.close(fig2)


def main() -> None:
    np.random.seed(7)

    band_count = 6
    crop_y = slice(120, 152)
    crop_x = slice(180, 212)

    print('=' * 72)
    print('Small-scale hyperspectral reconstruction test (visualized)')
    print('=' * 72)

    print_stage('Stage 1/5 | Loading and cropping data')
    hs = load_hdf5_dataset(HS_PATH, 'HS').astype(np.float32)
    a = load_hdf5_dataset(A_PATH, 'A').astype(np.float32)
    compressed_full = load_measurement_image(COMPRESSED_PATH)

    gt_cube = hs[:band_count, crop_y, crop_x]
    a_cube = a[:band_count, crop_y, crop_x]
    compressed_crop = compressed_full[crop_y, crop_x].astype(np.float32)

    print(f'Ground truth cube shape: {gt_cube.shape} (band, height, width)')
    print(f'Measurement cube shape:  {a_cube.shape} (band, height, width)')
    print(f'Compressed image crop:   {compressed_crop.shape}')
    print(f'GT value range:          [{gt_cube.min():.4f}, {gt_cube.max():.4f}]')

    print_stage('Stage 2/5 | Building forward/adjoint operators')
    op_norm_c = float(np.sqrt(np.max(np.sum(a_cube * a_cube, axis=0))))
    if op_norm_c <= 0.0:
        raise ValueError('Operator norm is zero; measurement cube is invalid')
    print(f'Operator norm C: {op_norm_c:.6f}')

    def forward_op(cube_est: np.ndarray) -> np.ndarray:
        return np.sum(a_cube * cube_est, axis=0) / op_norm_c

    def adjoint_op(meas: np.ndarray) -> np.ndarray:
        return (a_cube * meas[np.newaxis, :, :]) / op_norm_c

    measurement = forward_op(gt_cube)
    compressed_crop_scaled = compressed_crop / op_norm_c
    measurement_mse = float(np.mean((measurement - compressed_crop_scaled) ** 2))
    measurement_psnr = calc_psnr(compressed_crop_scaled, measurement)
    print(f'Measurement check MSE: {measurement_mse:.6e}')
    print(f'Measurement check PSNR (synth vs real crop): {measurement_psnr:.2f} dB')

    print_stage('Stage 3/5 | Configuring TwIST + TV proximal')
    tv_inner_iters = 10
    regularization_tau = 0.008
    max_main_iters = 40
    min_main_iters = 10
    tol_stop = 5e-4
    lambda_min_eig = 1e-4

    print(f'tv_inner_iters:    {tv_inner_iters}')
    print(f'regularization_tau:{regularization_tau}')
    print(f'max_main_iters:    {max_main_iters}')
    print(f'tol_stop:          {tol_stop}')

    def psi_tv(cube_in: np.ndarray, tau_prox: float) -> np.ndarray:
        tv_weight = float(1.0 / max(tau_prox, 1e-12))
        return tv_denoise_chambolle(cube_in, tv_weight, n_iter=tv_inner_iters)

    def phi_tv(cube_in: np.ndarray) -> float:
        return tv_norm_iso(cube_in)

    init_cube = adjoint_op(measurement)
    init_cube = psi_tv(init_cube, 0.01 * regularization_tau)

    print_stage('Stage 4/5 | Running TwIST optimization')
    result = twist(
        y=measurement,
        A=forward_op,
        tau=regularization_tau,
        AT=adjoint_op,
        psi=psi_tv,
        phi=phi_tv,
        lam1=lambda_min_eig,
        lamN=1.0,
        stop_criterion=1,
        tolA=tol_stop,
        debias=False,
        maxiter=max_main_iters,
        miniter=min_main_iters,
        initialization=2,
        enforce_monotone=True,
        sparse=False,
        true_x=gt_cube,
        verbose=True,
    )

    recon_cube = np.asarray(result['x'], dtype=np.float32)
    recon_cube = np.clip(recon_cube, 0.0, 1.0)

    mse_band = np.mean((recon_cube - gt_cube) ** 2, axis=(1, 2))
    psnr_band = 10.0 * np.log10(1.0 / (mse_band + 1e-12))
    mean_psnr = float(np.mean(psnr_band))

    print_stage('Stage 5/5 | Reporting and visualization')
    print('Band-wise PSNR:')
    for idx, value in enumerate(psnr_band, start=1):
        print(f'  band {idx:02d}: {value:6.2f} dB')

    print(f'\nMean PSNR:        {mean_psnr:.2f} dB')
    print(f'Final objective:   {result["objective"][-1]:.6e}')
    print(f'Iterations:        {result["iterations"]}')
    print(f'Final TV norm:     {tv_norm_iso(recon_cube):.6e}')

    mse_history = result['mses'] if result['mses'] is not None else None
    if mse_history is not None and len(mse_history) > 0:
        print(f'Initial MSE:       {mse_history[0]:.6e}')
        print(f'Final MSE:         {mse_history[-1]:.6e}')

    save_visualization(
        measurement=measurement,
        compressed_crop_scaled=compressed_crop_scaled,
        gt_cube=gt_cube,
        recon_cube=recon_cube,
        psnr_band=psnr_band,
        objective=np.asarray(result['objective']),
        mse_history=np.asarray(mse_history) if mse_history is not None else None,
        mean_psnr=mean_psnr,
    )

    print(f'Visualization saved to: {VIS_PATH}')
    print(f'Band detail saved to:   {VIS_DIR / "test_reconstruct_small_band_detail.png"}')

    assert mean_psnr > 25.0, f'Mean PSNR too low: {mean_psnr:.2f} dB'
    print('\n✅ Reconstruction is valid: mean PSNR > 25 dB')


if __name__ == '__main__':
    main()
