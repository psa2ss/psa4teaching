"""
励磁系统模型 (Exciter Models)
========================

包含电力系统中常用的励磁系统（Exciter）模型实现，主要用于暂态稳定分析。

支持的模型：
    - SEXS: 简化励磁系统模型（ENTSO-E标准）

数学模型
--------

SEXS 简化励磁系统传递函数：

    G(s) = (1+sTA)/(1+sTB) * K/(1+sTE)

其中：
    - K: 励磁系统增益
    - TA: 导前时间常数（秒）
    - TB: 滞后时间常数（秒）
    - TE: 励磁机时间常数（秒）

两级串联：
    第一级：G1(s) = (1+sTA)/(1+sTB) — 电压调节器相位补偿
    第二级：G2(s) = K/(1+sTE) — 励磁机响应

状态空间实现（三阶）：
    - x1: 电压调节器状态（第一级输出）
    - x2: 励磁机状态（第二级状态）
    - x3: 励磁电压 EFD（第二级输出）

参考：
    - ENTSO-E "Model Exchange for Power System Stability Studies" Report
    - IEEE Standard 421.5-2016
    - PSS/E Manual: SEXS Model
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
from scipy import signal


@dataclass
class SEXSParams:
    """SEXS 简化励磁系统参数

    Attributes:
        K: 励磁系统增益
        TA: 导前时间常数（秒）
        TB: 滞后时间常数（秒）
        TE: 励磁机时间常数（秒）
        EMIN: 最小励磁电压（标幺值），通常 0
        EMAX: 最大励磁电压（标幺值），通常 3-5

    Note:
        - K 通常为 50-400
        - TA 通常为 1-5s
        - TB 通常为 5-20s
        - TE 通常为 0.01-0.1s
        - EMAX 通常为 3-5 倍额定励磁电压
        - EMIN 通常为 0

    Example:
        >>> params = SEXSParams(K=200, TA=3.0, TB=10.0, TE=0.05)
        >>> A, B, C, D = params.get_state_space()
    """

    K: float = 200.0                              # 励磁系统增益
    TA: float = 3.0                               # 导前时间常数（秒）
    TB: float = 10.0                              # 滞后时间常数（秒）
    TE: float = 0.05                              # 励磁机时间常数（秒）
    EMIN: float = 0.0                             # 最小励磁电压
    EMAX: float = 4.0                             # 最大励磁电压

    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 SEXS 励磁系统的传递函数

        Returns:
            (num, den): 传递函数的分子和分母系数数组

        Note:
            G(s) = (1+sTA)/(1+sTB) * K/(1+sTE)

            展开为二阶系统：
            num = K * [TA, 1]
            den = [TB*TE, TB+TE, 1]
        """
        # G1(s) = (1 + s*TA) / (1 + s*TB)
        num1 = np.array([self.TA, 1.0])
        den1 = np.array([self.TB, 1.0])

        # G2(s) = K / (1 + s*TE)
        num2 = np.array([self.K])
        den2 = np.array([self.TE, 1.0])

        # 串联：G(s) = G1(s) * G2(s)
        num = np.convolve(num1, num2)
        den = np.convolve(den1, den2)

        return num, den

    def get_state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取 SEXS 励磁系统的状态空间表示

        状态变量：
            - x1: 电压调节器状态（相位补偿输出）
            - x2: 励磁机状态（内部状态）
            - x3: 励磁电压 EFD（输出）

        Returns:
            (A, B, C, D): 状态空间矩阵

        Note:
            状态空间实现：
            第一级：x'1 = (u - x1) / TB，输出 y1 = (TA/TB)*u + (1-TA/TB)*x1
            第二级：x'2 = (y1*K - x2) / TE，输出 y2 = x2
        """
        # 状态空间矩阵（三阶系统）
        # 使用可控标准型
        num, den = self.get_transfer_function()

        # 直接使用 scipy.signal.tf2ss 转换
        A, B, C, D = signal.tf2ss(num, den)

        return A, B, C, D

    def compute(self, V_ref: float, V_measured: float, V_S: float,
                dt: float, state: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """时域计算 SEXS 励磁系统输出（欧拉法）

        Args:
            V_ref: 参考电压（标幺值）
            V_measured: 实测端电压（标幺值）
            V_S: 附加稳定信号（标幺值，来自PSS）
            dt: 时间步长（秒）
            state: 当前状态向量 [x1, x2, x3]

        Returns:
            (efd, new_state): 励磁电压 EFD（标幺值）和新状态

        Note:
            输入信号：u = V_ref - V_measured + V_S
            两级传递函数串联计算
            输出限制在 [EMIN, EMAX] 范围内
        """
        if state is None:
            state = np.zeros(3)

        x1, x2, x3 = state

        # 输入信号：电压偏差 + PSS附加信号
        u = V_ref - V_measured + V_S

        # 第一级：G1(s) = (1+sTA)/(1+sTB)
        # dx1/dt = (u - x1) / TB
        # y1 = (TA/TB)*u + (1 - TA/TB)*x1
        dx1 = (u - x1) / self.TB
        y1 = (self.TA / self.TB) * u + (1.0 - self.TA / self.TB) * x1

        # 第二级：G2(s) = K/(1+sTE)
        # dx2/dt = (K*y1 - x2) / TE
        dx2 = (self.K * y1 - x2) / self.TE

        # 欧拉积分
        new_x1 = x1 + dt * dx1
        new_x2 = x2 + dt * dx2

        # 输出为 x3 = x2，限制励磁电压在 [EMIN, EMAX]
        efd = np.clip(new_x2, self.EMIN, self.EMAX)

        new_state = np.array([new_x1, new_x2, efd])

        # 如果输出被限幅，调整状态
        if efd >= self.EMAX:
            new_state[2] = self.EMAX
        elif efd <= self.EMIN:
            new_state[2] = self.EMIN

        return efd, new_state

    def compute_rk4(self, V_ref: float, V_measured: float, V_S: float,
                    dt: float, state: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """使用 RK4 方法计算 SEXS 励磁系统输出

        Args:
            V_ref: 参考电压（标幺值）
            V_measured: 实测端电压（标幺值）
            V_S: 附加稳定信号（标幺值）
            dt: 时间步长（秒）
            state: 当前状态向量 [x1, x2, x3]

        Returns:
            (efd, new_state): 励磁电压 EFD 和新状态
        """
        if state is None:
            state = np.zeros(3)

        def f_VR(x, u):
            """第一级电压调节器状态导数"""
            x1 = x[0]
            dx1 = (u - x1) / self.TB
            y1 = (self.TA / self.TB) * u + (1.0 - self.TA / self.TB) * x1
            return dx1, y1

        def f_EX(x, y1):
            """第二级励磁机状态导数"""
            x2 = x[0]
            dx2 = (self.K * y1 - x2) / self.TE
            return dx2

        u = V_ref - V_measured + V_S

        # RK4 for first stage (voltage regulator)
        state_VR = np.array([state[0]])

        def f_vr(x_vr):
            return np.array([(u - x_vr[0]) / self.TB])

        k1_vr = f_vr(state_VR)
        k2_vr = f_vr(state_VR + dt/2 * k1_vr)
        k3_vr = f_vr(state_VR + dt/2 * k2_vr)
        k4_vr = f_vr(state_VR + dt * k3_vr)

        new_x1 = state_VR[0] + dt/6 * (k1_vr[0] + 2*k2_vr[0] + 2*k3_vr[0] + k4_vr[0])

        # Output of first stage (use new state)
        y1 = (self.TA / self.TB) * u + (1.0 - self.TA / self.TB) * new_x1

        # RK4 for second stage (exciter)
        state_EX = np.array([state[1]])

        def f_ex(x_ex):
            return np.array([(self.K * y1 - x_ex[0]) / self.TE])

        k1_ex = f_ex(state_EX)
        k2_ex = f_ex(state_EX + dt/2 * k1_ex)
        k3_ex = f_ex(state_EX + dt/2 * k2_ex)
        k4_ex = f_ex(state_EX + dt * k3_ex)

        new_x2 = state_EX[0] + dt/6 * (k1_ex[0] + 2*k2_ex[0] + 2*k3_ex[0] + k4_ex[0])

        # 输出限幅
        efd = np.clip(new_x2, self.EMIN, self.EMAX)
        new_state = np.array([new_x1, new_x2, efd])

        return efd, new_state
