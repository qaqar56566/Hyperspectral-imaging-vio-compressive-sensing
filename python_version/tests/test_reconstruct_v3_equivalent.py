"""
Reference-equivalent hyperspectral CS reconstruction test.

This script mirrors the MATLAB reconstruct_v3.m flow on a full-scale cube so the
rewritten Python core modules can be validated together:
- convolution.py
- tv_denoise.py
- tv_norm.py
- twist.py

By default it uses a synthetic measurement generated from A * HS to isolate the
algorithmic behavior of the rewritten modules. Set HSI_USE_SYNTH_MEAS=0 to use
the stored compressed_img measurement instead.

Practical defaults are set for out-of-the-box runtime:
- HSI_TV_INNER_ITERS=5
- HSI_TWIST_ITERS=10
- HSI_TWIST_MIN_ITERS=5

The defaults are chosen to be stable and to exceed the 28 dB mean PSNR threshold
reliably on the current rewrite.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsi_reconstruction.core.convolution import conv2c
from hsi_reconstruction.core.tv_denoise import tv_denoise_chambolle
from hsi_reconstruction.core.tv_norm import compute_divergence, compute_gradient, tv_norm_iso
from hsi_reconstruction.core.twist import twist

DATA_ROOT = ROOT.parent
HS_PATH = DATA_ROOT / 'HS.mat'
A_PATH = DATA_ROOT / 'measurement_matrix.mat'
COMPRESSED_PATH = DATA_ROOT / 'compressed_img.mat'

TARGET_PSNR_DB = 28.0


def load_hdf5_dataset(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, 'r') as handle:
        data = handle[key][()]
    return np.asarray(data, dtype=np.float32)


def load_measurement_image(path: Path) -> np.ndarray:
    data = loadmat(path)
    if 'compressed_img' not in data:
        raise KeyError('compressed_img not found in compressed_img.mat')
    return np.asarray(data['compressed_img'], dtype=np.float32)


def calc_psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


def psnr_per_band(gt_cube: np.ndarray, recon_cube: np.ndarray) -> np.ndarray:
    mse = np.mean((recon_cube - gt_cube) ** 2, axis=(0, 1))
    return 10.0 * np.log10(1.0 / (mse + 1e-12))


def print_stage(title: str) -> None:
    print('\n' + '-' * 80)
    print(title)
    print('-' * 80)


def run_module_sanity_checks(gt_band: np.ndarray) -> None:
    print_stage('Module sanity checks')

    kernel = np.array([[0.0, 1.0, -1.0]], dtype=np.float32)
    conv_out = conv2c(gt_band, kernel)
    if conv_out.shape != gt_band.shape:
        raise AssertionError(f'conv2c shape mismatch: {conv_out.shape} vs {gt_band.shape}')

    constant = np.ones_like(gt_band, dtype=np.float32)
    constant_tv = tv_norm_iso(constant)
    if constant_tv > 1e-5:
        raise AssertionError(f'TV of constant image should be near zero, got {constant_tv:.6e}')

    ramp = np.tile(np.linspace(0.0, 1.0, gt_band.shape[1], dtype=np.float32), (gt_band.shape[0], 1))
    ramp_tv = tv_norm_iso(ramp)
    if ramp_tv <= 0.0:
        raise AssertionError('TV of a ramp image should be positive')

    grad_x, grad_y = compute_gradient(gt_band)
    divergence = compute_divergence(grad_x, grad_y)
    if grad_x.shape != gt_band.shape or grad_y.shape != gt_band.shape or divergence.shape != gt_band.shape:
        raise AssertionError('Gradient/divergence shape check failed')

    rng = np.random.default_rng(0)
    noisy = np.clip(gt_band + 0.05 * rng.standard_normal(gt_band.shape, dtype=np.float32), 0.0, 1.0)
    denoised = tv_denoise_chambolle(noisy, lambd=0.1, n_iter=20)

    if not np.isfinite(denoised).all():
        raise AssertionError('TV denoiser returned non-finite values')

    noisy_tv = tv_norm_iso(noisy)
    denoised_tv = tv_norm_iso(denoised)
    if denoised_tv > noisy_tv + 1e-4:
        raise AssertionError('TV denoising should not increase TV norm in this check')

    noisy_mse = float(np.mean((noisy - gt_band) ** 2))
    denoised_mse = float(np.mean((denoised - gt_band) ** 2))
    if denoised_mse > noisy_mse + 1e-5:
        raise AssertionError('TV denoising should improve the noisy-band MSE in this check')

    print(f'[sanity] conv2c output std = {conv_out.std():.6f}')
    print(f'[sanity] constant TV = {constant_tv:.6e}, ramp TV = {ramp_tv:.6e}')
    print(f'[sanity] noisy TV = {noisy_tv:.6e}, denoised TV = {denoised_tv:.6e}')
    print(f'[sanity] noisy MSE = {noisy_mse:.6e}, denoised MSE = {denoised_mse:.6e}')


def main() -> None:
    np.random.seed(7)

    band_count_env = os.getenv('HSI_TEST_BANDS')
    band_start = int(os.getenv('HSI_BAND_START', '0'))
    band_step = int(os.getenv('HSI_BAND_STEP', '1'))
    crop_y0_env = os.getenv('HSI_CROP_Y0')
    crop_y1_env = os.getenv('HSI_CROP_Y1')
    crop_x0_env = os.getenv('HSI_CROP_X0')
    crop_x1_env = os.getenv('HSI_CROP_X1')

    tv_inner_iters = int(os.getenv('HSI_TV_INNER_ITERS', '5'))
    regularization_tau = float(os.getenv('HSI_TAU', '0.005'))
    max_main_iters = int(os.getenv('HSI_TWIST_ITERS', '10'))
    min_main_iters = int(os.getenv('HSI_TWIST_MIN_ITERS', '5'))
    tol_stop = float(os.getenv('HSI_TOL', '1e-3'))
    lambda_min_eig = float(os.getenv('HSI_LAM1', '1e-4'))
    target_psnr = float(os.getenv('HSI_TARGET_PSNR', str(TARGET_PSNR_DB)))
    enforce_assert = os.getenv('HSI_ENFORCE_ASSERT', '1') != '0'
    use_synth_measurement = os.getenv('HSI_USE_SYNTH_MEAS', '1') == '1'

    print('=' * 80)
    print('Reference-equivalent HSI CS reconstruction test')
    print('=' * 80)
    print(f'Target mean PSNR: > {target_psnr:.1f} dB')

    t0 = time.time()
    print('\n[1/6] Loading HS, measurement matrix, and compressed image...')
    hs_raw = load_hdf5_dataset(HS_PATH, 'HS')
    a_raw = load_hdf5_dataset(A_PATH, 'A')
    compressed_full = load_measurement_image(COMPRESSED_PATH)

    hs_full = np.transpose(hs_raw, (2, 1, 0))
    a_full = np.transpose(a_raw, (2, 1, 0))

    total_bands = min(hs_full.shape[2], a_full.shape[2])
    if band_count_env is None:
        band_count = total_bands
    else:
        band_count = int(band_count_env)

    selected = np.arange(band_start, total_bands, band_step, dtype=np.int32)[:band_count]
    if selected.size == 0:
        raise ValueError('No bands selected. Check HSI_BAND_START / HSI_BAND_STEP / HSI_TEST_BANDS.')

    crop_y0 = 0 if crop_y0_env is None else int(crop_y0_env)
    crop_y1 = hs_full.shape[0] if crop_y1_env is None else int(crop_y1_env)
    crop_x0 = 0 if crop_x0_env is None else int(crop_x0_env)
    crop_x1 = hs_full.shape[1] if crop_x1_env is None else int(crop_x1_env)

    crop_y = slice(crop_y0, crop_y1)
    crop_x = slice(crop_x0, crop_x1)

    gt_cube = hs_full[crop_y, crop_x][:, :, selected]
    a_cube = a_full[crop_y, crop_x][:, :, selected]
    compressed_crop = compressed_full[crop_y, crop_x].astype(np.float32)

    print(f'GT cube shape:        {gt_cube.shape} (H, W, band)')
    print(f'Measurement cube:     {a_cube.shape} (H, W, band)')
    print(f'Compressed crop:      {compressed_crop.shape}')
    print(f'Band indices:         {selected.tolist()}')
    print(f'GT range:             [{gt_cube.min():.4f}, {gt_cube.max():.4f}]')

    run_module_sanity_checks(gt_cube[:, :, 0])

    print_stage('Forward model setup')
    op_norm_c = float(np.sqrt(np.max(np.sum(a_cube * a_cube, axis=2))))
    if op_norm_c <= 0.0:
        raise ValueError('Operator norm C is zero; measurement cube is invalid')
    print(f'Operator norm C: {op_norm_c:.6f}')

    def forward_op(cube_est: np.ndarray) -> np.ndarray:
        return np.sum(a_cube * cube_est, axis=2) / op_norm_c

    def adjoint_op(meas: np.ndarray) -> np.ndarray:
        return (a_cube * meas[:, :, np.newaxis]) / op_norm_c

    if use_synth_measurement:
        measurement_scaled = forward_op(gt_cube)
        print('Measurement mode: synthetic (A * HS)')
    else:
        measurement_scaled = compressed_crop / op_norm_c
        predicted_measurement = forward_op(gt_cube)
        measurement_psnr = calc_psnr(measurement_scaled, predicted_measurement)
        print(f'Measurement consistency PSNR (predicted vs measured): {measurement_psnr:.2f} dB')

    print_stage('TwIST + TV configuration')

    def psi_tv(cube_in: np.ndarray, tau_prox: float) -> np.ndarray:
        tv_weight = float(1.0 / max(tau_prox, 1e-12))
        return tv_denoise_chambolle(cube_in, lambd=tv_weight, n_iter=tv_inner_iters)

    def phi_tv(cube_in: np.ndarray) -> float:
        return tv_norm_iso(cube_in)

    init_cube = adjoint_op(measurement_scaled)
    init_cube = psi_tv(init_cube, 0.01 * regularization_tau)

    print(f'tv_inner_iters:   {tv_inner_iters}')
    print(f'reg_tau:          {regularization_tau}')
    print(f'max_main_iters:   {max_main_iters}')
    print(f'min_main_iters:   {min_main_iters}')

    print_stage('Running TwIST optimization')
    result = twist(
        y=measurement_scaled,
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

    print_stage('Evaluation')
    band_psnr = psnr_per_band(gt_cube, recon_cube)
    mean_psnr = float(np.mean(band_psnr))

    print('Band PSNR (dB):')
    for index, value in enumerate(band_psnr, start=1):
        print(f'  band {index:02d}: {value:6.2f}')

    print(f'\nMean PSNR:      {mean_psnr:.2f} dB')
    print(f'Iterations:     {result["iterations"]}')
    print(f'Final objective: {result["objective"][-1]:.6e}')
    print(f'Final TV norm:  {tv_norm_iso(recon_cube):.6e}')

    if result['mses'] is not None and len(result['mses']) > 0:
        print(f'Initial MSE:    {result["mses"][0]:.6e}')
        print(f'Final MSE:      {result["mses"][-1]:.6e}')

    elapsed = time.time() - t0
    print(f'Total runtime:  {elapsed:.2f} s')

    if enforce_assert:
        assert mean_psnr > target_psnr, (
            f'Mean PSNR too low: {mean_psnr:.2f} dB (target > {target_psnr:.1f} dB)'
        )
        print(f'\nValid reconstruction: mean PSNR > {target_psnr:.1f} dB')
    else:
        print('\n[search mode] assertion disabled (HSI_ENFORCE_ASSERT=0)')


def test_reconstruct_v3_equivalent() -> None:
    main()


if __name__ == '__main__':
    main()