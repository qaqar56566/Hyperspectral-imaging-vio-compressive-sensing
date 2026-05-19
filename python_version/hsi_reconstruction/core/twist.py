"""
TwIST (Two-Step Iterative Shrinkage/Thresholding) Algorithm

对应MATLAB文件: TwIST.m

这是一个通用的迭代收缩阈值算法框架，用于求解形如:
    min_x 0.5*||y - A*x||_2^2 + tau*phi(x)
的优化问题。

核心论文:
- Bioucas-Dias & Figueiredo, "A New TwIST: Two-Step Iterative 
  Shrinkage/Thresholding Algorithms for Image Restoration", 
  IEEE Transactions on Image Processing, 2007.

参考: www.lx.it.pt/~bioucas/TwIST
"""

import numpy as np
import time
from typing import Callable, Optional, Union, Tuple, Dict, Any


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """
    软阈值函数 (soft thresholding function)
    
    y = sign(x) * max(|x| - tau, 0)
    
    对于复数或实数都适用。
    
    Args:
        x: 输入数据
        tau: 阈值参数
        
    Returns:
        阈值处理后的输出
    """
    y = np.maximum(np.abs(x) - tau, 0)
    # 避免除以零
    y = y / (y + tau) * x
    return y


def twist(
    y: np.ndarray,
    A: Union[np.ndarray, Callable],
    tau: Union[float, np.ndarray],
    AT: Optional[Callable] = None,
    psi: Optional[Callable] = None,
    phi: Optional[Callable] = None,
    x0: Optional[np.ndarray] = None,
    lam1: float = 1e-4,
    lamN: float = 1.0,
    alpha: float = 0.0,
    beta: float = 0.0,
    max_svd_cap: float = 1e6,
    max_backtracks: int = 50,
    stop_criterion: int = 1,
    tolA: float = 0.01,
    debias: bool = False,
    tolD: float = 0.001,
    maxiter: int = 1000,
    maxiter_debias: int = 200,
    miniter: int = 5,
    miniter_debias: int = 5,
    initialization: int = 0,
    enforce_monotone: bool = True,
    sparse: bool = True,
    true_x: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    TwIST算法求解器
    
    解决形如下式的优化问题:
        arg min_x 0.5*||y - A*x||_2^2 + tau*phi(x)
    
    其中phi(.)是一个正则化函数，其去噪问题的解为:
        Psi_tau(y) = arg min_x 0.5*||x - y||_2^2 + tau*phi(x)
    
    Args:
        y: 观测数据 (1D向量或2D图像)
        
        A: 前向算子，可以是:
           - (m, n) 矩阵，用于y和x都是1D向量的情况
           - 函数句柄，计算A*v的乘积
           
        tau: 正则化参数 (标量或与x同形状的数组)
        
        AT: A的伴随算子函数句柄 (当A是函数时必需)
        
        psi: 去噪函数句柄，签名为 psi(x, tau)
             默认为软阈值函数
             
        phi: 正则泛函句柄，计算phi(x)的值
             默认为L1范数
             
        x0: 可选的显式初始化 (若提供，优先使用)

        lam1: TwIST参数，A'*A最小特征值的估计
              建议值:
              - 1e-4: 严重病态问题
              - 1e-2: 轻微病态问题  
              - 1.0:  A为标准正交
              
        lamN: A'*A最大特征值的估计 (通常设为1)
        
        alpha, beta: TwIST算法参数，默认自动计算
        
        stop_criterion: 停止准则选择
                       0: 非零分量变化数
                       1: 目标函数相对变化 (默认)
                       2: 相邻迭代估计的相对范数
                       3: 目标函数值
                       
        tolA: 停止阈值 (默认0.01)
        
        debias: 是否进行去偏差阶段 (布尔值)
        
        tolD: 去偏差阶段阈值 (默认0.001)
        
        maxiter: 主阶段最大迭代次数 (默认1000)
        
        maxiter_debias: 去偏差阶段最大迭代次数 (默认200)
        
        miniter: 主阶段最小迭代次数 (默认5)
        
        miniter_debias: 去偏差阶段最小迭代次数 (默认5)
        
        initialization: 初始化方式
                        0: 零初始化
                        1: 随机初始化
                        2: x0 = A'*y
                        
        enforce_monotone: 是否强制单调性 (默认True)
        
        sparse: 针对稀疏诱导正则化的加速 (默认True)
        
        true_x: 真实x (若给定，会计算MSE)
        
        verbose: 是否输出迭代信息 (默认True)
        
    Returns:
        字典包含:
        - 'x': 主算法解
        - 'x_debias': 去偏差后的解 (若debias=False则为None)
        - 'objective': 目标函数迭代历史
        - 'times': 每次迭代的计算时间
        - 'debias_start': 去偏差开始的迭代数
        - 'mses': MSE历史 (若给定true_x)
        - 'max_svd': 最大奇异值估计
    """
    
    # ==================== 参数检查与初始化 ====================
    
    # 检查必需参数数量 (Python中隐含)
    
    # 如果A是矩阵，转换为函数句柄
    if isinstance(A, np.ndarray):
        A_mat = A.copy()
        AT_func = lambda x: A_mat.T @ x
        A_func = lambda x: A_mat @ x
        A = A_func
        AT = AT_func
    elif AT is None:
        raise ValueError("When A is a function, AT must be provided")
    
    # 计算 A'*y (后面会频繁使用)
    Aty = AT(y)
    
    # 处理phi和psi函数
    if psi is None:
        psi = soft_threshold
    
    if phi is None:
        phi = lambda x: np.sum(np.abs(x))
    
    # ==================== TwIST参数计算 ====================
    
    rho0 = (1 - lam1 / lamN) / (1 + lam1 / lamN)
    
    if alpha == 0:
        alpha = 2 / (1 + np.sqrt(1 - rho0**2))
    
    if beta == 0:
        beta = alpha * 2 / (lam1 + lamN)
    
    # ==================== 初始化 ====================
    
    # 确定x的形状
    if x0 is not None:
        x = np.asarray(x0).copy()
    elif initialization == 0:
        # 零初始化
        x = np.zeros_like(Aty)
    elif initialization == 1:
        # 随机初始化
        x = np.random.randn(*Aty.shape)
    elif initialization == 2:
        # x = A'*y
        x = Aty.copy()
    else:
        raise ValueError("Unknown initialization option")
    
    # 最大奇异值估计
    max_svd = float(lamN) if lamN > 0 else 1.0
    
    # 初始化迭代状态
    IST_iters = 0
    TwIST_iters = 0
    xm2 = x.copy()  # x_{k-2}
    xm1 = x.copy()  # x_{k-1}
    
    # 计算初始目标函数值
    resid = y - A(x)
    prev_f = 0.5 * np.sum(resid**2) + tau * phi(x)
    
    # 初始化输出
    objective = [prev_f]
    times_list = [0.0]
    mses = [] if true_x is not None else None
    
    if true_x is not None:
        mse_val = np.sum((x - true_x)**2) / true_x.size
        mses.append(mse_val)
    
    # 非零分量
    nz_x = (x != 0.0)
    num_nz_x = np.sum(nz_x)
    
    # 开始计时
    t0 = time.time()
    
    if verbose:
        print(f"\nInitial objective = {prev_f:.6e}, nonzeros = {num_nz_x:7d}")
    
    # ==================== TwIST主循环 ====================
    
    iteration = 1
    debias_start = 0
    
    while iteration <= maxiter:
        
        # 计算梯度
        grad = AT(resid)
        
        # 内循环 (IST和TwIST)
        for_ever = True
        backtracks = 0
        while for_ever:

            if not np.isfinite(max_svd) or max_svd <= 0:
                max_svd = float(lamN) if lamN > 0 else 1.0
            
            # IST步
            x = psi(xm1 + grad / max_svd, tau / max_svd)
            
            # 检查是否可以进行TwIST步
            if (IST_iters >= 2) or (TwIST_iters != 0):
                
                # 处理稀疏情况
                if sparse:
                    mask = (x != 0)
                    xm1 = xm1 * mask
                    xm2 = xm2 * mask
                
                # 两步迭代
                xm2_new = (alpha - beta) * xm1 + (1 - alpha) * xm2 + beta * x
                
                # 计算残差和目标函数
                resid = y - A(xm2_new)
                f = 0.5 * np.sum(resid**2) + tau * phi(xm2_new)
                
                # 检查单调性
                if (f > prev_f) and enforce_monotone:
                    # 单调性违反，回到IST迭代
                    TwIST_iters = 0
                else:
                    # 接受TwIST步
                    TwIST_iters += 1
                    IST_iters = 0
                    x = xm2_new
                    
                    # 每10000步逐渐减小max_svd
                    if TwIST_iters % 10000 == 0:
                        max_svd = 0.9 * max_svd
                    
                    break  # 跳出内循环
            
            else:
                # 还在IST阶段
                resid = y - A(x)
                f = 0.5 * np.sum(resid**2) + tau * phi(x)
                
                if (not np.isfinite(f)) or (f > prev_f):
                    # 单调性失败 -> 增加max_svd
                    max_svd = min(2 * max_svd, max_svd_cap)
                    backtracks += 1
                    if verbose:
                        print(f"Incrementing max_svd = {max_svd:.2e}")
                    IST_iters = 0
                    TwIST_iters = 0
                else:
                    TwIST_iters += 1
                    break  # 跳出内循环

            if backtracks >= max_backtracks:
                if verbose:
                    print("Backtrack limit reached, accepting current IST step.")
                resid = y - A(x)
                f = 0.5 * np.sum(resid**2) + tau * phi(x)
                break
        
        # 更新迭代
        xm2 = xm1.copy()
        xm1 = x.copy()
        
        # 更新非零分量统计
        nz_x_prev = nz_x.copy()
        nz_x = (x != 0.0)
        num_nz_x = np.sum(nz_x)
        num_changes_active = np.sum(nz_x != nz_x_prev)
        
        # 计算停止准则
        if stop_criterion == 0:
            # 基于非零分量变化数
            criterion = num_changes_active
        elif stop_criterion == 1:
            # 基于目标函数相对变化
            criterion = abs(f - prev_f) / (abs(prev_f) + 1e-12)
        elif stop_criterion == 2:
            # 基于估计的相对范数变化
            criterion = np.linalg.norm(x - xm1) / (np.linalg.norm(x) + 1e-12)
        elif stop_criterion == 3:
            # 目标函数值本身
            criterion = f
        else:
            raise ValueError("Unknown stopping criterion")
        
        # 检查停止条件
        stop_main = (iteration > maxiter) or (criterion <= tolA and iteration > miniter)
        if stop_main and iteration > miniter:
            break
        
        # 更新记录
        iteration += 1
        prev_f = f
        objective.append(f)
        times_list.append(time.time() - t0)
        
        if true_x is not None:
            mse_val = np.sum((x - true_x)**2) / true_x.size
            mses.append(mse_val)
        
        # 输出进度
        if verbose:
            if true_x is not None:
                try:
                    isnr = 10 * np.log10(np.sum((A(x) - y)**2) / (np.sum(x - true_x)**2 + 1e-12))
                    print(f"Iter={iteration:4d}, ISNR={isnr:8.5e}, obj={f:9.5e}, "
                          f"nz={num_nz_x:7d}, crit={criterion/tolA:7.3e}")
                except:
                    print(f"Iter={iteration:4d}, obj={f:9.5e}, nz={num_nz_x:7d}, "
                          f"crit={criterion/tolA:7.3e}")
            else:
                print(f"Iter={iteration:4d}, obj={f:9.5e}, nz={num_nz_x:7d}, "
                      f"crit={criterion/tolA:7.3e}")
    
    # ==================== 去偏差阶段 ====================
    
    x_debias = None
    
    if debias:
        if verbose:
            print("\nStarting debiasing phase...\n")
        
        x_debias = x.copy()
        zero_ind = (x_debias != 0)
        debias_start = iteration
        
        # 初始化CG求解器
        resid = A(x_debias) - y
        rvec = AT(resid) * zero_ind
        rTr_cg = np.sum(rvec**2)
        
        # 收敛阈值
        tol_debias = tolD * (np.sum(rvec**2) + 1e-12)
        
        # 初始化CG方向
        pvec = -rvec.copy()
        
        # CG主循环
        for debias_iter in range(maxiter_debias):
            
            # A*p步
            RWpvec = A(pvec)
            Apvec = AT(RWpvec) * zero_ind
            
            # CG步长
            alpha_cg = rTr_cg / (np.sum(pvec * Apvec) + 1e-12)
            
            # 更新
            x_debias = x_debias + alpha_cg * pvec
            resid = resid + alpha_cg * RWpvec
            rvec = rvec + alpha_cg * Apvec
            
            rTr_cg_plus = np.sum(rvec**2)
            beta_cg = rTr_cg_plus / (rTr_cg + 1e-12)
            pvec = -rvec + beta_cg * pvec
            
            rTr_cg = rTr_cg_plus
            
            # 记录
            obj_debias = 0.5 * np.sum(resid**2) + tau * phi(x_debias)
            objective.append(obj_debias)
            times_list.append(time.time() - t0)
            
            if true_x is not None:
                mse_val = np.sum((x_debias - true_x)**2) / true_x.size
                mses.append(mse_val)
            
            # 输出进度
            if verbose:
                print(f"Debias iter={debias_iter+1:5d}, resid={np.sum(resid**2):13.8e}, "
                      f"conv={rTr_cg / (tol_debias + 1e-12):8.3e}")
            
            # 停止条件
            if (debias_iter + 1 >= miniter_debias and 
                rTr_cg <= tol_debias):
                break
        
        if verbose:
            print("\nFinished debiasing phase!")
    
    # ==================== 返回结果 ====================
    
    return {
        'x': x,
        'x_debias': x_debias,
        'objective': np.array(objective),
        'times': np.array(times_list),
        'debias_start': debias_start,
        'mses': np.array(mses) if mses is not None else None,
        'max_svd': max_svd,
        'iterations': iteration,
    }
