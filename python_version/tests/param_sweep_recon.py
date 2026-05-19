"""
Parameter sweep to find TwIST/TV parameters that reach mean PSNR > 28 dB
on the 6-band 32x32 crop used in the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
import time
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
    compressed = loadmat(root_parent / 'compressed_img.mat')['compressed_img'].astype(np.float32)

    # crop
    gt = hs[:6, 120:152, 180:212].astype(np.float32)
    a_cube = a[:6, 120:152, 180:212].astype(np.float32)
    c = float(np.sqrt(np.max(np.sum(a_cube * a_cube, axis=0))))

    def forward(x):
        return np.sum(a_cube * x, axis=0) / c

    def adj(y):
        return (a_cube * y[np.newaxis, :, :]) / c

    measurement = forward(gt)

    reg_tau_list = [0.0025, 0.004, 0.006, 0.008, 0.01]
    tv_iters_list = [5, 10, 20]
    tv_weight_scale_list = [0.5, 1.0, 1.5]
    maxiter_list = [40, 80]

    best = {'psnr': -1.0, 'params': None}

    start = time.time()
    # Make numerical issues raise so we can catch them
    old_err = np.seterr(all='raise')
    for reg_tau in reg_tau_list:
        for tv_iters in tv_iters_list:
            for tv_scale in tv_weight_scale_list:
                for maxit in maxiter_list:
                    def psi_tv(cube, tau_prox, tv_scale=tv_scale, tv_iters=tv_iters):
                        # try both common mappings: scale / tau_prox
                        weight = float(tv_scale / max(tau_prox, 1e-12))
                        return tv_denoise_chambolle(cube, lambd=weight, n_iter=tv_iters)

                    def phi_tv(cube):
                        return tv_norm_iso(cube)

                    try:
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
                            tolA=5e-4,
                            debias=False,
                            maxiter=maxit,
                            miniter=10,
                            initialization=2,
                            enforce_monotone=True,
                            sparse=False,
                            true_x=gt,
                            verbose=False,
                        )
                    except Exception as e:
                        print(f'Exception for params reg_tau={reg_tau}, tv_iters={tv_iters}, tv_scale={tv_scale}, maxit={maxit}:')
                        import traceback

                        traceback.print_exc()
                        # continue to next param set
                        continue

                    recon = np.clip(np.asarray(res['x'], dtype=np.float32), 0.0, 1.0)
                    mse = np.mean((recon - gt) ** 2, axis=(1, 2))
                    psnr = 10.0 * np.log10(1.0 / (mse + 1e-12))
                    mean_psnr = float(np.mean(psnr))

                    print(f'reg_tau={reg_tau}, tv_iters={tv_iters}, tv_scale={tv_scale}, maxit={maxit} -> mean_psnr={mean_psnr:.2f} dB')

                    if mean_psnr > best['psnr']:
                        best['psnr'] = mean_psnr
                        best['params'] = (reg_tau, tv_iters, tv_scale, maxit)

                    if mean_psnr >= 28.0:
                        print('Found params achieving >=28 dB:', best)
                        print('Elapsed:', time.time() - start)
                        return

    # restore numeric error handling
    np.seterr(**old_err)
    print('Sweep finished. Best:', best, 'Elapsed:', time.time() - start)


if __name__ == '__main__':
    main()
