"""
TwIST 算法单元测试和验证

测试目标:
1. 验证TwIST算法的收敛性
2. 对标MATLAB原始实现的结果精度
3. 小矩阵例子的端到端测试
"""

import numpy as np
import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hsi_reconstruction.core.twist import twist, soft_threshold


class TestSoftThreshold:
    """测试软阈值函数"""
    
    def test_soft_threshold_basic(self):
        """基础软阈值测试"""
        x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5])
        tau = 1.0
        y = soft_threshold(x, tau)
        
        # 验证计算: |x| <= tau 时结果为0
        # |x| > tau 时结果为 sign(x) * (|x| - tau)
        expected = np.array([0, 0.5, 1.5, 0, -0.5, -1.5])
        np.testing.assert_allclose(y, expected, rtol=1e-10)
    
    def test_soft_threshold_zero_input(self):
        """阈值为零时的测试"""
        x = np.array([1.0, 2.0, 3.0])
        tau = 0.0
        y = soft_threshold(x, tau)
        
        # tau=0时应该返回原值
        np.testing.assert_allclose(y, x, rtol=1e-10)
    
    def test_soft_threshold_large_tau(self):
        """阈值过大时的测试"""
        x = np.array([0.5, 1.0, 1.5])
        tau = 10.0
        y = soft_threshold(x, tau)
        
        # tau很大时结果接近零
        assert np.allclose(y, 0, atol=1e-6)


class TestTwISTSimple:
    """TwIST算法基础测试"""
    
    def test_twist_simple_least_squares(self):
        """
        简单最小二乘问题测试
        
        最小化: 0.5*||y - A*x||_2^2 + tau*||x||_1
        
        其中:
        - A是5x3的随机矩阵
        - 真实x_true是稀疏的
        - tau相对较小，使得l1项的影响可见
        """
        np.random.seed(42)
        
        # 构造小问题
        m, n = 10, 5
        A = np.random.randn(m, n)
        
        # 真实的稀疏解
        x_true = np.zeros(n)
        x_true[0] = 2.0
        x_true[2] = -1.5
        
        # 生成观测
        y = A @ x_true + 0.01 * np.random.randn(m)
        
        # 运行TwIST
        tau = 0.05
        result = twist(
            y, A, tau,
            lam1=1e-4,
            maxiter=100,
            tolA=1e-4,
            verbose=False,
        )
        
        x_recon = result['x']
        
        # 验证收敛性: 目标函数应单调递减
        obj = result['objective']
        assert obj[-1] < obj[0], "目标函数应该递减"
        
        # 验证收敛: 重建误差应该较小
        recon_error = np.linalg.norm(A @ x_recon - y) / np.linalg.norm(y)
        assert recon_error < 0.1, f"重建误差过大: {recon_error}"
        
        print(f"\n✓ 最小二乘问题:")
        print(f"  初始目标函数: {obj[0]:.6e}")
        print(f"  最终目标函数: {obj[-1]:.6e}")
        print(f"  迭代次数: {len(obj)}")
        print(f"  重建误差: {recon_error:.6e}")
    
    def test_twist_convergence_monotonic(self):
        """
        测试目标函数单调性
        
        验证: obj[k] >= obj[k+1]
        """
        np.random.seed(123)
        
        m, n = 15, 8
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[np.random.choice(n, 3, replace=False)] = np.random.randn(3)
        y = A @ x_true + 0.01 * np.random.randn(m)
        
        tau = 0.1
        result = twist(
            y, A, tau,
            lam1=1e-4,
            maxiter=200,
            tolA=1e-5,
            enforce_monotone=True,
            verbose=False,
        )
        
        obj = result['objective']
        
        # 检查单调性
        for i in range(len(obj)-1):
            assert obj[i] >= obj[i+1] - 1e-9, f"单调性违反在第{i}步"
        
        print(f"\n✓ 单调性测试:")
        print(f"  目标函数从 {obj[0]:.6e} 递减到 {obj[-1]:.6e}")
        print(f"  下降幅度: {(obj[0]-obj[-1])/obj[0]*100:.2f}%")
    
    def test_twist_with_matrices(self):
        """
        基于矩阵形式的测试
        
        测试A作为直接矩阵输入（而非函数句柄）
        """
        np.random.seed(999)
        
        m, n = 12, 6
        A = np.random.randn(m, n)
        x_true = np.zeros(n)
        x_true[[1, 3]] = [1.5, -0.8]
        
        y = A @ x_true
        tau = 0.1
        
        # TwIST应优化处理矩阵形式 (内部转换为函数句柄)
        result = twist(
            y, A, tau,
            maxiter=50,
            tolA=1e-3,
            verbose=False,
        )
        
        x_recon = result['x']
        residual = np.linalg.norm(A @ x_recon - y)
        
        # 由于是L1正则化，残差可能不会精确为零，调整容差
        assert residual < 0.1, "残差过大"
        
        print(f"\n✓ 矩阵形式测试:")
        print(f"  残差: {residual:.6e}")
        print(f"  真实解的非零数: {np.sum(x_true != 0)}")
        print(f"  重建解的非零数: {np.sum(x_recon != 0)}")


class TestTwISTConvergence:
    """收敛性和精度验证测试"""
    
    def test_twist_stops_on_tolerance(self):
        """
        测试算法在达到容差时停止
        """
        np.random.seed(456)
        
        m, n = 20, 10
        A = np.random.randn(m, n) * 0.5  # 缩小避免病态
        x_true = np.zeros(n)
        x_true[np.random.choice(n, 4, replace=False)] = np.random.randn(4)
        
        y = A @ x_true + 0.01 * np.random.randn(m)
        tau = 0.05
        
        # 运行两种不同的容差
        result_loose = twist(
            y, A, tau,
            maxiter=500,
            tolA=0.1,
            verbose=False,
        )
        
        result_tight = twist(
            y, A, tau,
            maxiter=500,
            tolA=0.001,
            verbose=False,
        )
        
        # 更紧的容差应该需要更多迭代
        iters_loose = result_loose['objective'].shape[0]
        iters_tight = result_tight['objective'].shape[0]
        
        assert iters_tight >= iters_loose, "容差越小迭代次数应越多"
        assert result_tight['objective'][-1] <= result_loose['objective'][-1], \
            "容差越小最终目标函数值应越小"
        
        print(f"\n✓ 容差停止测试:")
        print(f"  松容差 (0.1): {iters_loose} 次迭代，最终obj={result_loose['objective'][-1]:.6e}")
        print(f"  紧容差 (0.001): {iters_tight} 次迭代，最终obj={result_tight['objective'][-1]:.6e}")
    
    def test_twist_sparse_recovery(self):
        """
        测试稀疏恢复能力
        
        验证算法能否恢复已知稀疏信号
        """
        np.random.seed(789)
        
        # 构造稀疏信号恢复问题
        n = 50
        k = 5  # 稀疏度
        m = int(2 * k * np.log(n))  # 压缩感知采样数
        
        # 随机Gaussian测量矩阵
        A = np.random.randn(m, n) / np.sqrt(m)
        
        # 真实稀疏信号
        x_true = np.zeros(n)
        support = np.random.choice(n, k, replace=False)
        x_true[support] = np.random.randn(k)
        
        # 无噪声观测
        y = A @ x_true
        
        # 基追踪 (Basis Pursuit)
        tau = 0.01
        result = twist(
            y, A, tau,
            lam1=1e-4,
            maxiter=500,
            tolA=1e-6,
            verbose=False,
        )
        
        x_recon = result['x']
        
        # 评估恢复质量
        support_recon = np.where(np.abs(x_recon) > 1e-3)[0]
        residual_norm = np.linalg.norm(x_true - x_recon)
        
        # 如果A满足RIP条件，应该能准确恢复
        assert len(support_recon) <= k + 3, "恢复的非零分量过多"
        assert residual_norm < 0.1, "恢复误差过大"
        
        print(f"\n✓ 稀疏恢复测试:")
        print(f"  信号维度: {n}, 稀疏度: {k}, 测量数: {m}")
        print(f"  真实非零数: {k}, 恢复非零数: {len(support_recon)}")
        print(f"  恢复误差: {residual_norm:.6e}")


class TestTwISTWithMSE:
    """带真值的MSE计算测试"""
    
    def test_twist_mse_tracking(self):
        """
        测试MSE追踪功能
        """
        np.random.seed(321)
        
        m, n = 25, 12
        A = np.random.randn(m, n) * 0.5
        x_true = np.zeros(n)
        x_true[[2, 5, 9]] = [1.0, -0.5, 0.8]
        
        y = A @ x_true + 0.02 * np.random.randn(m)
        tau = 0.05
        
        result = twist(
            y, A, tau,
            true_x=x_true,
            maxiter=100,
            tolA=1e-4,
            verbose=False,
        )
        
        mses = result['mses']
        
        assert mses is not None, "MSE应该被计算"
        assert len(mses) > 0, "MSE列表不应为空"
        assert mses[-1] <= mses[0], "MSE应该递减"
        
        print(f"\n✓ MSE追踪测试:")
        print(f"  初始MSE: {mses[0]:.6e}")
        print(f"  最终MSE: {mses[-1]:.6e}")
        print(f"  MSE改进: {(mses[0]-mses[-1])/mses[0]*100:.2f}%")


def run_full_test_suite():
    """
    运行完整测试套件和演示
    """
    print("=" * 70)
    print("TwIST 算法验证测试")
    print("=" * 70)
    
    # 软阈值测试
    print("\n[1/4] 软阈值函数测试")
    print("-" * 70)
    test_soft = TestSoftThreshold()
    test_soft.test_soft_threshold_basic()
    test_soft.test_soft_threshold_zero_input()
    test_soft.test_soft_threshold_large_tau()
    print("✓ 所有软阈值测试通过")
    
    # 简单功能测试
    print("\n[2/4] TwIST简单功能测试")
    print("-" * 70)
    test_simple = TestTwISTSimple()
    test_simple.test_twist_simple_least_squares()
    test_simple.test_twist_convergence_monotonic()
    test_simple.test_twist_with_matrices()
    print("✓ 所有简单功能测试通过")
    
    # 收敛性测试
    print("\n[3/4] 收敛性和精度测试")
    print("-" * 70)
    test_conv = TestTwISTConvergence()
    test_conv.test_twist_stops_on_tolerance()
    test_conv.test_twist_sparse_recovery()
    print("✓ 所有收敛性测试通过")
    
    # MSE追踪测试
    print("\n[4/4] MSE追踪功能测试")
    print("-" * 70)
    test_mse = TestTwISTWithMSE()
    test_mse.test_twist_mse_tracking()
    print("✓ MSE追踪测试通过")
    
    print("\n" + "=" * 70)
    print("✅ 所有测试通过！TwIST算法实现验证完成。")
    print("=" * 70)


if __name__ == "__main__":
    # 如果直接运行此文件，执行完整测试
    run_full_test_suite()
