"""
循环卷积模块 (Circular 2D Convolution)

用于TV去噪和其他图像处理操作的循环边界条件卷积。
"""

import numpy as np
from scipy import signal
from typing import Tuple


def wraparound(x: np.ndarray, m: int) -> np.ndarray:
    """
    为循环卷积实现wrap-around填充。
    
    参数
    ----
    x : np.ndarray
        输入数组 (2D)
    m : int
        核的大小 (假设核源在中心)
    
    返回值
    ------
    np.ndarray
        填充后的数组
    """
    h, w = x.shape
    
    # 计算填充量（基于核原点位置）
    p1 = (m - 1) // 2
    p2 = m - 1 - p1
    
    # 创建填充数组
    x_padded = np.zeros((h + p1 + p2, w + p1 + p2), dtype=x.dtype)
    
    # 放置原始数组
    x_padded[p1:p1+h, p1:p1+w] = x
    
    # 填充wrap-around区域
    # 四个角
    x_padded[:p1, :p1] = x[-p1:, -p1:]                    # 左上角
    x_padded[:p1, p1+w:] = x[-p1:, :p2]                   # 右上角
    x_padded[p1+h:, :p1] = x[:p2, -p1:]                   # 左下角
    x_padded[p1+h:, p1+w:] = x[:p2, :p2]                  # 右下角
    
    # 四条边
    x_padded[:p1, p1:p1+w] = x[-p1:, :]                   # 上边
    x_padded[p1+h:, p1:p1+w] = x[:p2, :]                  # 下边
    x_padded[p1:p1+h, :p1] = x[:, -p1:]                   # 左边
    x_padded[p1:p1+h, p1+w:] = x[:, :p2]                  # 右边
    
    return x_padded


def conv2c(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    循环2D卷积 (Circular 2D Convolution)
    
    计算带有wrap-around边界条件的2D卷积。
    输出大小与输入大小相同。
    
    参数
    ----
    x : np.ndarray
        输入图像 (2D, float32或float64)
    h : np.ndarray
        卷积核 (2D, float32或float64)
    
    返回值
    ------
    np.ndarray
        卷积结果，形状与x相同
    
    示例
    ----
    >>> x = np.random.randn(10, 10)
    >>> h = np.array([[1, 0, -1]])  # 水平差分
    >>> y = conv2c(x, h)
    >>> y.shape
    (10, 10)
    """
    x = np.asarray(x, dtype=np.float32)
    h = np.asarray(h, dtype=np.float32)
    
    # 获取核的大小
    kh, kw = h.shape
    
    # Wrap-around填充
    x_padded = wraparound(x, max(kh, kw))
    
    # 使用'valid'模式进行卷积（无额外填充）
    y_padded = signal.convolve2d(x_padded, h, mode='valid')
    
    # 裁剪回原始大小
    h_orig, w_orig = x.shape
    y = y_padded[:h_orig, :w_orig]
    
    return y


def conv2c_adjoint(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    循环卷积的伴随算子 (Adjoint)
    
    对应于旋转180度的核进行循环卷积。
    用于Chambolle算法中的divergence计算。
    
    参数
    ----
    x : np.ndarray
        输入数组 (2D)
    h : np.ndarray
        原始卷积核 (2D)
    
    返回值
    ------
    np.ndarray
        伴随卷积结果
    """
    # 核旋转180度（flip）
    h_flipped = np.flip(np.flip(h, axis=0), axis=1)
    
    # 执行循环卷积
    return conv2c(x, h_flipped)
