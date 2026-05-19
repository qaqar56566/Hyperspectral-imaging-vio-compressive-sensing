"""
Runner using MATLAB-like TwIST/TV parameters to try to reach PSNR > 28 dB.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hsi_reconstruction.core.tv_denoise import tv_denoise_chambolle
from hsi_reconstruction.core.tv_norm import tv_norm_iso
from hsi_reconstruction.core.twist import twist


def load_hdf5_dataset(path: Path, key: str):
    with h5py.File(path, 'r') as f:
        return np.asarray(f[key][()], dtype=np.float32)


def main():
    root_parent = ROOT.parent
    hs = load_hdf5_dataset(root_parent / 'HS.mat', 'HS')
    a = load_hdf5_dataset(root_parent / 'measurement_matrix.mat', 'A')

    gt = hs[:6, 120:152, 180:212].astype(np.float32)
    a_cube = a[:6, 120:152, 180:212].astype(np.float32)
    c = float(np.sqrt(np.max(np.sum(a_cube * a_cube, axis=0))))

    def forward(x):
        return np.sum(a_cube * x, axis=0) / c

    def adj(y):
        return (a_cube * y[np.newaxis, :, :]) / c

    measurement = forward(gt)

    tv_inner_iters = 30
    reg_tau = 0.005
    max_main_iters = 200
    min_main_iters = 50

    def psi_tv(cube, tau_prox):
        # MATLAB mapping lambda_tv = 1 / tau_prox
        weight = float(1.0 / max(tau_prox, 1e-12))
        return tv_denoise_chambolle(cube, lambd=weight, n_iter=tv_inner_iters)

    def phi_tv(cube):
        return tv_norm_iso(cube)

    print('Running MATLAB-like parameters:')
    print(f'  reg_tau={reg_tau}, tv_inner_iters={tv_inner_iters}, max_main_iters={max_main_iters}')

    res = twist(
        y=measurement,
        A=forward,
        tau=reg_tau,
        AT=adj,
        psi=psi_tv,
        phi=phi_tv,
        lam1=1e-4,
        lamN=1.0,
        stop_criterion=1,
        tolA=1e-3,
        debias=False,
        maxiter=max_main_iters,
        miniter=min_main_iters,
        initialization=2,
        enforce_monotone=True,
        sparse=False,
        true_x=gt,
        verbose=True,
    )

    recon = np.clip(np.asarray(res['x'], dtype=np.float32), 0.0, 1.0)
    mse = np.mean((recon - gt) ** 2, axis=(1, 2))
    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
    mean_psnr = float(np.mean(psnr))
    print('Mean PSNR:', mean_psnr)


if __name__ == '__main__':
    main()
