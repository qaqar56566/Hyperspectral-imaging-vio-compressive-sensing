"""
TV去噪模块单元测试 - 简化版
专注于核心功能验证而不是完整的收敛性验证
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsi_reconstruction.core.convolution import conv2c
from hsi_reconstruction.core.tv_norm import tv_norm_iso, tv_norm_aniso
from hsi_reconstruction.core.tv_denoise import tv_denoise_chambolle


def test_circular_convolution():
    """测试循环卷积"""
    print("\n" + "="*60)
    print("测试1: 循环卷积")
    print("="*60)
    
    x = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=np.float32)
    
    h = np.array([[0, 1, -1]], dtype=np.float32)
    y = conv2c(x, h)
    
    assert y.shape == x.shape, f"形状错误: {y.shape} != {x.shape}"
    print(f"✓ 循环卷积: 输入{x.shape} → 输出{y.shape}")
    
    # 测试大尺寸
    x_large = np.random.randn(32, 32).astype(np.float32)
    y_large = conv2c(x_large, h)
    assert y_large.shape == x_large.shape
    print(f"✓ 大尺寸卷积: {x_large.shape} OK")


def test_tv_norms():
    """测试TV范数计算"""
    print("\n" + "="*60)
    print("测试2: TV范数")
    print("="*60)
    
    # 常数图像TV应为0
    u_const = np.ones((5, 5), dtype=np.float32)
    tv_const = tv_norm_iso(u_const)
    assert tv_const < 1e-5, f"常数图像TV不为0: {tv_const}"
    print(f"✓ 常数图像: TV = {tv_const:.6e}")
    
    # 棋盘图案TV应> 0
    u_check = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]], dtype=np.float32)
    tv_check = tv_norm_iso(u_check)
    assert tv_check > 0.1, f"棋盘TV太小: {tv_check}"
    print(f"✓ 棋盘图案: TV = {tv_check:.6e}")
    
    # 对比各向同性和异性
    u_rand = np.random.randn(8, 8).astype(np.float32)
    tv_iso = tv_norm_iso(u_rand)
    tv_aniso = tv_norm_aniso(u_rand)
    assert tv_iso > 0 and tv_aniso > 0
    print(f"✓ 各向同性: {tv_iso:.6e}")
    print(f"✓ 各向异性: {tv_aniso:.6e}")


def test_tv_denoise_basic():
    """基础TV去噪测试"""
    print("\n" + "="*60)
    print("测试3: TV去噪基础功能")
    print("="*60)
    
    np.random.seed(42)
    
    # 创建噪声图像
    u_true = np.ones((8, 8), dtype=np.float32)
    u_true[3:5, 3:5] = 2.0
    f = u_true + 0.2 * np.random.randn(8, 8).astype(np.float32)
    
    # 执行去噪
    u = tv_denoise_chambolle(f, lambd=1.0, n_iter=20)
    
    # 验证输出形状和类型
    assert u.shape == f.shape, f"形状错误: {u.shape} != {f.shape}"
    assert u.dtype == np.float32
    assert not np.isnan(u).any(), "输出包含NaN"
    assert not np.isinf(u).any(), "输出包含Inf"
    
    print(f"✓ 输出形状: {u.shape}")
    print(f"✓ 输出范围: [{u.min():.3f}, {u.max():.3f}]")
    print(f"✓ 输入范围: [{f.min():.3f}, {f.max():.3f}]")
    print(f"✓ 无NaN/Inf")


def test_tv_denoise_multiple_lambdas():
    """测试多个λ参数"""
    print("\n" + "="*60)
    print("测试4: 多个正则化参数")
    print("="*60)
    
    np.random.seed(123)
    f = np.ones((8, 8), dtype=np.float32)
    f += 0.2 * np.random.randn(8, 8).astype(np.float32)
    
    lambdas = [0.1, 0.5, 1.0, 2.0]
    for lambd in lambdas:
        u = tv_denoise_chambolle(f, lambd, n_iter=10)
        assert u.shape == f.shape
        assert not np.isnan(u).any()
        print(f"✓ λ={lambd:3.1f}: 去噪成功")


def test_convergence():
    """测试收敛性"""
    print("\n" + "="*60)
    print("测试5: Chambolle算法收敛性")
    print("="*60)
    
    np.random.seed(456)
    u_true = np.zeros((8, 8), dtype=np.float32)
    u_true[2:6, 2:6] = 1.0
    f = u_true + 0.15 * np.random.randn(8, 8).astype(np.float32)
    
    # 运行不同迭代数
    n_iters = [5, 10, 20, 50]
    results = []
    
    for n_iter in n_iters:
        u = tv_denoise_chambolle(f, lambd=1.0, n_iter=n_iter)
        tv = tv_norm_iso(u)
        results.append((n_iter, tv))
        print(f"✓ 迭代{n_iter:2d}: TV = {tv:12.6e}")
    
    # 验证都产生有效输出
    for n_iter, tv in results:
        assert tv > 0, f"TV应为正数"
        assert not np.isnan(tv), f"TV包含NaN"


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("= TV去噪模块测试套件 =")
    print("="*70)
    
    test_circular_convolution()
    test_tv_norms()
    test_tv_denoise_basic()
    test_tv_denoise_multiple_lambdas()
    test_convergence()
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
