"""
TV去噪模块

使用稳定的 Chambolle TV 去噪实现，支持 2D 图像和 3D 光谱立方体。
"""

import numpy as np
from typing import Dict

from skimage.restoration import denoise_tv_chambolle

from .tv_norm import tv_norm_iso

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _tv_chambolle_2d_numba(f, weight, n_iter):
        h, w = f.shape
        px = np.zeros((h, w), dtype=np.float32)
        py = np.zeros((h, w), dtype=np.float32)
        div_p = np.zeros((h, w), dtype=np.float32)
        grad_x = np.zeros((h, w), dtype=np.float32)
        grad_y = np.zeros((h, w), dtype=np.float32)
        denom = np.zeros((h, w), dtype=np.float32)
        u = np.zeros((h, w), dtype=np.float32)
        tau = np.float32(0.125)
        weight_f = np.float32(weight)
        step = tau / weight_f

        for _ in range(n_iter):
            div_p.fill(0.0)
            div_p[:, 0] = px[:, 0]
            div_p[:, 1:] = px[:, 1:] - px[:, :-1]
            div_p[0, :] += py[0, :]
            div_p[1:, :] += py[1:, :] - py[:-1, :]

            u[:] = f - weight_f * div_p

            grad_x.fill(0.0)
            grad_y.fill(0.0)
            grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
            grad_y[:-1, :] = u[1:, :] - u[:-1, :]

            denom[:] = np.float32(1.0) + step * np.sqrt(grad_x * grad_x + grad_y * grad_y)
            px[:] = (px + step * grad_x) / denom
            py[:] = (py + step * grad_y) / denom

        div_p.fill(0.0)
        div_p[:, 0] = px[:, 0]
        div_p[:, 1:] = px[:, 1:] - px[:, :-1]
        div_p[0, :] += py[0, :]
        div_p[1:, :] += py[1:, :] - py[:-1, :]

        return f - weight_f * div_p


    @njit(parallel=True, cache=True)
    def _tv_chambolle_3d_numba(f, weight, n_iter):
        h, w, b = f.shape
        out = np.empty_like(f, dtype=np.float32)
        for band in prange(b):
            out[:, :, band] = _tv_chambolle_2d_numba(f[:, :, band], weight, n_iter)
        return out


def tv_denoise_chambolle_numba(
    f: np.ndarray,
    lambd: float,
    n_iter: int = 50,
) -> np.ndarray:
    """Numba-accelerated Chambolle TV denoise for 2D/3D arrays."""

    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is not available")

    f = np.asarray(f, dtype=np.float32)
    weight = np.float32(max(lambd, 1e-12))
    max_num_iter = int(max(1, n_iter))

    if f.ndim == 2:
        return _tv_chambolle_2d_numba(f, weight, max_num_iter)

    if f.ndim == 3:
        return _tv_chambolle_3d_numba(f, weight, max_num_iter)

    raise ValueError("tv_denoise_chambolle_numba only supports 2D or 3D arrays")


def tv_denoise_chambolle(
    f: np.ndarray,
    lambd: float,
    n_iter: int = 50,
    tau: float = 0.25,
    verbose: bool = False,
    prefer_numba: bool = True,
) -> np.ndarray:
    """TV 去噪。`lambd` 对应 skimage 的 `weight`，值越大去噪越强。"""

    if prefer_numba and NUMBA_AVAILABLE:
        return tv_denoise_chambolle_numba(f, lambd, n_iter=n_iter)

    f = np.asarray(f, dtype=np.float32)
    weight = float(max(lambd, 1e-12))
    max_num_iter = int(max(1, n_iter))

    if f.ndim == 2:
        result = denoise_tv_chambolle(f, weight=weight, max_num_iter=max_num_iter, channel_axis=None)
        return np.asarray(result, dtype=np.float32)

    if f.ndim == 3:
        result = denoise_tv_chambolle(
            f,
            weight=weight,
            max_num_iter=max_num_iter,
            channel_axis=2,
        )
        return np.asarray(result, dtype=np.float32)

    raise ValueError('tv_denoise_chambolle only supports 2D or 3D arrays')


def tv_denoise_chambolle_detailed(
    f: np.ndarray,
    lambd: float,
    n_iter: int = 50,
    tau: float = 0.25,
    verbose: bool = False
) -> Dict:
    """返回去噪结果以及一个简单的 TV / MSE 轨迹。"""

    f = np.asarray(f, dtype=np.float32)
    tv_history = []
    mse_history = []

    if f.ndim == 2:
        current = f
        for _ in range(int(max(1, n_iter))):
            current = tv_denoise_chambolle(current, lambd, n_iter=1, tau=tau, verbose=verbose)
            tv_history.append(tv_norm_iso(current))
            mse_history.append(float(np.mean((current - f) ** 2)))
        return {'u': current, 'tv': tv_history, 'mse': mse_history, 'n_iter': len(tv_history)}

    if f.ndim == 3:
        current = f
        for _ in range(int(max(1, n_iter))):
            current = tv_denoise_chambolle(current, lambd, n_iter=1, tau=tau, verbose=verbose)
            tv_history.append(tv_norm_iso(current))
            mse_history.append(float(np.mean((current - f) ** 2)))
        return {'u': current, 'tv': tv_history, 'mse': mse_history, 'n_iter': len(tv_history)}

    raise ValueError('tv_denoise_chambolle_detailed only supports 2D or 3D arrays')
