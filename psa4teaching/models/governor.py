"""
调速器模型 (Governor Models)
======================

包含电力系统中常用的调速器（Governor）模型实现，主要用于暂态稳定分析。

支持的模型：
    - TGOV1: 简化调速器模型（ENTSO-E标准）

数学模型
--------

TGOV1 调速器传递函数：

    G(s) = (1/R) * 1/(1+sT1) * (1+sT2)/(1+sT3)

其中：
    - R: 调差系数（%）
    - T1: 伺服系统时间常数（秒）
    - T2: 瞬态增益时间常数（秒）
    - T3: 调速器时间常数（秒）

状态空间实现（三阶）：
    - x1 = xL (伺服系统输出)
    - x2 = xT (调速器暂态输出)
    - x3 = PMECH (机械功率输出)

参考：
    - ENTSO-E "Model Exchange for Power System Stability Studies" Report
    - PSS/E 动态模型手册
    - IEEE Committee Report 1973
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
from scipy import signal


@dataclass
class TGOV1Params:
    """TGOV1 调速器参数

    Attributes:
        R: 调差系数（标幺值），通常 0.03-0.06
        T1: 伺服系统时间常数（秒），通常 0.1-1.0s
        T2: 瞬态增益时间常数（秒），通常 1-5s
        T3: 调速器时间常数（秒），通常 5-15s
        Dt: 调速器死区（标幺值），通常 0
        VMIN: 最小阀位限制（标幺值），通常 0
        VMAX: 最大阀位限制（标幺值），通常 1.0-1.2
        P_base_MW: 基准功率（MW），用于标幺值转换

    Note:
        - 调差系数 R = 1/δ，其中 δ 为调差率（%）
        - 死区 Dt 用于模拟调速器的不灵敏区
        - VMIN/VMAX 限制机械功率输出范围

    Example:
        >>> params = TGOV1Params(R=0.05, T1=0.5, T2=3.0, T3=10.0)
        >>> num, den = params.get_transfer_function()
    """

    R: float = 0.05                            # 调差系数（标幺值）
    T1: float = 0.5                            # 伺服系统时间常数（秒）
    T2: float = 3.0                            # 瞬态增益时间常数（秒）
    T3: float = 10.0                           # 调速器时间常数（秒）
    Dt: float = 0.0                            # 调速器死区（标幺值）
    VMIN: float = 0.0                          # 最小阀位限制
    VMAX: float = 1.0                          # 最大阀位限制
    P_base_MW: float = 475.0                   # 基准功率（MW）

    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 TGOV1 调速器的传递函数

        Returns:
            (num, den): 传递函数的分子和分母系数数组

        Note:
            G(s) = (1/R) * 1/(1+sT1) * (1+sT2)/(1+sT3)

            展开后：
            num = (1/R) * [T2, 1]
            den = [T1*T3, T1+T3, 1] * 与 (1+sT2) 卷积...

            实际实现使用串联形式：
            G1(s) = (1/R) / (1+sT1)
            G2(s) = (1+sT2) / (1+sT3)
            G(s) = G1(s) * G2(s)
        """
        # G1(s) = (1/R) / (1 + s*T1)
        num1 = np.array([1.0 / self.R])
        den1 = np.array([self.T1, 1.0])

        # G2(s) = (1 + s*T2) / (1 + s*T3)
        num2 = np.array([self.T2, 1.0])
        den2 = np.array([self.T3, 1.0])

        # 串联：G(s) = G1(s) * G2(s)
        num = np.convolve(num1, num2)
        den = np.convolve(den1, den2)

        return num, den

    def get_state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取 TGOV1 调速器的状态空间表示

        状态变量：
            - x1: 伺服系统状态（对应 1/(1+sT1)）
            - x2: 调速器暂态状态（对应 (1+sT2)/(1+sT3) 的分子）
            - x3: 机械功率输出 PMECH

        Returns:
            (A, B, C, D): 状态空间矩阵

        Note:
            状态空间实现采用可控标准型
            系统为三阶（3个状态变量）
        """
        # 状态空间矩阵（三阶系统）
        # x' = A*x + B*u,  y = C*x + D*u
        # u = Δω (转速偏差)

        A = np.array([
            [-1.0/self.T1, 0.0, 0.0],
            [0.0, -1.0/self.T3, 0.0],
            [0.0, 1.0/self.T2, 0.0]  # x3 跟踪 x2/T2
        ])

        B = np.array([
            [1.0/self.R],  # 输入到伺服系统
            [1.0/self.R],  # 输入到调速器暂态
            [0.0]
        ])

        C = np.array([[0.0, 0.0, 1.0]])  # 输出为 x3 (PMECH)

        D = np.array([[0.0]])

        return A, B, C, D

    def compute(self, delta_omega: float, dt: float,
                state: Optional[np.ndarray] = None,
                P_ref: float = 0.0) -> Tuple[float, np.ndarray]:
        """时域计算 TGOV1 调速器输出（欧拉法）

        ENTSO-E Fig 2-2: TGOV1 = 1/R → limiter → 1/(1+sT1) → Dt + (1+sT2)/(1+sT3)

        Args:
            delta_omega: 转速偏差 Δω（标幺值）
            dt: 时间步长（秒）
            state: 当前状态向量 [x1, x2, x3]，若为 None 则初始化为零
                x1: 1/(1+sT1) 输出，x2: 1/(1+sT3) 输出，x3: PMECH（输出跟踪）
            P_ref: 功率参考值（标幺值）

        Returns:
            (P_mech, new_state): 机械功率输出（标幺值）和新状态
        """
        if state is None:
            state = np.zeros(3)

        # 输入信号：u = P_ref - Δω/R
        u = P_ref - delta_omega / self.R
        # 阀位限幅（ENTSO-E Fig 2-2: 在 1/(1+sT1) 之前）
        u_limited = np.clip(u, self.VMIN, self.VMAX)

        x1, x2, x3 = state

        # 调速器/伺服: 1/(1+sT1), 输入 = u_limited
        dx1 = (u_limited - x1) / self.T1

        # 汽轮机: (1+sT2)/(1+sT3) = T2/T3 + (1-T2/T3)/(1+sT3)
        # x2 是 1/(1+sT3) 的输出, 输入 = x1
        dx2 = (x1 - x2) / self.T3
        turbine_out = (self.T2 / self.T3) * x1 + (1.0 - self.T2 / self.T3) * x2

        # PMECH = Dt * x1 + turbine_out
        P_mech = self.Dt * x1 + turbine_out

        new_state = state + dt * np.array([dx1, dx2, 0.0])
        new_state[2] = P_mech

        return P_mech, new_state

    def compute_rk4(self, delta_omega: float, dt: float,
                     state: Optional[np.ndarray] = None,
                     P_ref: float = 0.0) -> Tuple[float, np.ndarray]:
        """使用 RK4 方法计算 TGOV1 调速器输出

        ENTSO-E Fig 2-2 标准结构:
            u = P_ref - Δω/R → [VMIN,VMAX] → 1/(1+sT1) → x1
            → 分支A: Dt·x1 → PMECH
            → 分支B: (1+sT2)/(1+sT3)·x1 → PMECH

        Args:
            delta_omega: 转速偏差 Δω（标幺值）
            dt: 时间步长（秒）
            state: 当前状态向量 [x1, x2, x3]
                x1: 1/(1+sT1) 输出, x2: 1/(1+sT3) 输出, x3: PMECH（输出跟踪）
            P_ref: 功率参考值（标幺值，以汽轮机额定功率为基准）

        Returns:
            (P_mech, new_state): 机械功率输出和新状态
        """
        if state is None:
            state = np.zeros(3)

        # 输入信号与阀位限幅
        u = P_ref - delta_omega / self.R
        u_limited = np.clip(u, self.VMIN, self.VMAX)

        def f(x):
            """TGOV1 状态导数（按 ENTSO-E Fig 2-2）"""
            x1, x2, _x3 = x

            # 调速器/伺服: 1/(1+sT1)
            dx1 = (u_limited - x1) / self.T1

            # 汽轮机: (1+sT2)/(1+sT3) = T2/T3 + (1-T2/T3)/(1+sT3)
            dx2 = (x1 - x2) / self.T3

            return np.array([dx1, dx2, 0.0])

        # RK4 积分
        k1 = f(state)
        k2 = f(state + dt/2 * k1)
        k3 = f(state + dt/2 * k2)
        k4 = f(state + dt * k3)

        new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # 从新状态计算 PMECH
        x1_new, x2_new = new_state[0], new_state[1]
        turbine_out = (self.T2 / self.T3) * x1_new + (1.0 - self.T2 / self.T3) * x2_new
        P_mech = self.Dt * x1_new + turbine_out
        new_state[2] = P_mech

        return P_mech, new_state
