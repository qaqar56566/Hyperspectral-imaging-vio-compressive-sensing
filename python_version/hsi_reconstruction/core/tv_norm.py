"""
总变差 (Total Variation) 范数模块

使用周期边界的前向差分计算 TV 范数，支持 2D 图像和 3D 光谱立方体。
"""

import numpy as np


def _forward_diff_periodic(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """周期边界前向差分。"""
    dx = np.roll(u, -1, axis=1) - u
    dy = np.roll(u, -1, axis=0) - u
    return dx, dy


def tv_norm_iso(u: np.ndarray, eps: float = 0.0) -> float:
    """各向同性 TV 范数。"""
    u = np.asarray(u, dtype=np.float32)

    if u.ndim == 2:
        dx, dy = _forward_diff_periodic(u)
        return float(np.sum(np.sqrt(dx * dx + dy * dy + eps)))

    if u.ndim == 3:
        tv = 0.0
        for band in range(u.shape[2]):
            dx, dy = _forward_diff_periodic(u[:, :, band])
            tv += float(np.sum(np.sqrt(dx * dx + dy * dy + eps)))
        return tv

    raise ValueError('tv_norm_iso only supports 2D or 3D arrays')


def tv_norm_aniso(u: np.ndarray) -> float:
    """各向异性 TV 范数。"""
    u = np.asarray(u, dtype=np.float32)

    if u.ndim == 2:
        dx, dy = _forward_diff_periodic(u)
        return float(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))

    if u.ndim == 3:
        tv = 0.0
        for band in range(u.shape[2]):
            dx, dy = _forward_diff_periodic(u[:, :, band])
            tv += float(np.sum(np.abs(dx)) + np.sum(np.abs(dy)))
        return tv

    raise ValueError('tv_norm_aniso only supports 2D or 3D arrays')


def compute_gradient(u: np.ndarray) -> tuple:
    """计算周期边界下的梯度。"""
    u = np.asarray(u, dtype=np.float32)
    return _forward_diff_periodic(u)


def compute_divergence(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """计算周期边界下的散度。"""
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    div_x = p1 - np.roll(p1, 1, axis=1)
    div_y = p2 - np.roll(p2, 1, axis=0)
    return div_x + div_y
