"""
Debug runner for the v3-equivalent reconstruction test.

Wraps the TwIST call with try/except and prints full traceback so we can
see why the process was terminating early. Also saves result to a npz file.
"""

from __future__ import annotations

import traceback
import sys
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_reconstruct_v3_equivalent import (
    load_hdf5_dataset,
    load_measurement_image,
    tv_denoise_chambolle,
    tv_norm_iso,
)

from hsi_reconstruction.core.twist import twist


def main():
    try:
        hs = load_hdf5_dataset(ROOT.parent / 'HS.mat', 'HS')
        a = load_hdf5_dataset(ROOT.parent / 'measurement_matrix.mat', 'A')
        compressed = load_measurement_image(ROOT.parent / 'compressed_img.mat')

        # small crop
        gt = hs[:6, 120:152, 180:212].astype(np.float32)
        a_cube = a[:6, 120:152, 180:212].astype(np.float32)
        c = float(np.sqrt(np.max(np.sum(a_cube * a_cube, axis=0))))

        def forward(x):
            return np.sum(a_cube * x, axis=0) / c

        def adj(y):
            return (a_cube * y[np.newaxis, :, :]) / c

        measurement = forward(gt)

        def psi_tv(cube, tau_prox):
            return tv_denoise_chambolle(cube, lambd=float(1.0 / max(tau_prox, 1e-12)), n_iter=10)

        def phi_tv(cube):
            return tv_norm_iso(cube)

        print('Starting debug-run with verbose TwIST...')
        res = twist(
            measurement,
            forward,
            0.008,
            AT=adj,
            psi=psi_tv,
            phi=phi_tv,
            lam1=1e-4,
            lamN=1.0,
            stop_criterion=1,
            tolA=5e-4,
            debias=False,
            maxiter=100,
            miniter=10,
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
        print('Mean PSNR (debug-run):', mean_psnr)

        outp = Path.cwd() / 'debug_recon_result.npz'
        np.savez_compressed(outp, recon=recon, psnr=psnr, res=res)
        print('Saved debug result to', outp)

    except Exception:
        print('Exception during debug-run:')
        traceback.print_exc()


if __name__ == '__main__':
    main()
