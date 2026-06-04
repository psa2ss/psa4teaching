"""
电力系统稳定器模型 (Power System Stabilizer Models)
==============================================

包含电力系统中常用的电力系统稳定器（PSS）模型实现，主要用于暂态稳定分析。

支持的模型：
    - PSS2A: 双输入电力系统稳定器（ENTSO-E标准）

数学模型
--------

PSS2A 传递函数结构：

    V_S = K_S * (1+sT8)^N/(1+sT9)^M * (1+sT7)/(1+sT6) * W(s) * W(s) * ...

其中 W(s) = (1+sT1)/(1+sT2) * (1+sT3)/(1+sT4) 为超前-滞后环节

简化版（N=0, M=0, T6=0, TW4=0）：
    - (1+sT8)^N/(1+sT9)^M = 1
    - T6=0: 直通
    - TW4=0: 第二通道无效

状态空间实现（约6阶）：
    - 清洗环节（Washout）：2阶（TW1, TW2, TW3）
    - 超前滞后环节：4阶（T1/T2, T3/T4各两对）

参考：
    - ENTSO-E "Model Exchange for Power System Stability Studies" Report
    - IEEE Standard 421.5-2016
    - PSS/E Manual: PSS2A Model
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
from scipy import signal


@dataclass
class PSS2AParams:
    """PSS2A 电力系统稳定器参数

    Attributes:
        KS1: 第一通道增益
        KS2: 第二通道增益
        KS3: 信号选择器增益（1=选ω，2=选P）
        TW1: 第一通道清洗时间常数（秒）
        TW2: 第二通道清洗时间常数（秒）
        TW3: 第三清洗时间常数（秒）
        TW4: 第四清洗时间常数（秒），0表示无效
        T1-T4: 第一对超前滞后时间常数
        T6: 第二对超前滞后输出时间常数（0=直通）
        T7: 第三对超前滞后时间常数
        T8: 第四对超前滞后时间常数
        T9: 第五对超前滞后时间常数
        VSTMIN: 最小稳定信号限制
        VSTMAX: 最大稳定信号限制
        N: 分子阶数
        M: 分母阶数
        IC1: 输入代码1（1=转速，2=功率）
        IC2: 输入代码2

    Note:
        - 简化版：N=0, M=0, T6=0, TW4=0
        - KS1 通常为 5-20
        - 清洗时间常数 TW 通常为 1-10s
        - 超前滞后时间常数 T1-T4 通常为 0.01-1s

    Example:
        >>> params = PSS2AParams(KS1=10, KS2=0.1564, TW1=2.0)
        >>> signal_out = params.compute_stabilizing_signal(delta_omega, P_gen, dt, state)
    """

    # 增益参数
    KS1: float = 10.0                             # 第一通道增益
    KS2: float = 0.1564                           # 第二通道增益
    KS3: float = 1.0                              # 信号选择器（1=ω，2=P）

    # 清洗环节时间常数
    TW1: float = 2.0                              # 第一清洗时间常数
    TW2: float = 2.0                              # 第二清洗时间常数
    TW3: float = 2.0                              # 第三清洗时间常数
    TW4: float = 0.0                              # 第四清洗时间常数（0=无效）

    # 超前滞后环节时间常数
    T1: float = 0.25                              # 第一超前时间常数
    T2: float = 0.03                              # 第一滞后时间常数
    T3: float = 0.15                              # 第二超前时间常数
    T4: float = 0.015                             # 第二滞后时间常数
    T6: float = 0.0                               # 输出时间常数（0=直通）
    T7: float = 2.0                               # 第三超前时间常数
    T8: float = 0.5                               # 第四超前时间常数
    T9: float = 0.1                               # 第五超前时间常数

    # 输出限制
    VSTMIN: float = -0.1                          # 最小稳定信号
    VSTMAX: float = 0.1                           # 最大稳定信号

    # 阶数
    N: int = 0                                    # 分子阶数
    M: int = 0                                    # 分母阶数

    # 输入代码
    IC1: int = 1                                  # 输入1（1=转速，2=功率）
    IC2: int = 3                                  # 输入2

    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 PSS2A 的传递函数（简化版）

        Returns:
            (num, den): 传递函数的分子和分母系数数组

        Note:
            简化版 N=0, M=0, T6=0, TW4=0
            传递函数结构：
            W(s) = (1+sT7)/(1+sT6) * (1+sT1)/(1+sT2) * (1+sT3)/(1+sT4)
            由于 T6=0，简化为：
            W(s) = (1+sT7) * (1+sT1)/(1+sT2) * (1+sT3)/(1+sT4)
        """
        # 清洗环节：Gw(s) = s*TW/(1+s*TW)
        # 简化处理：使用一阶高通滤波器
        num_w = np.array([self.TW1, 0.0])
        den_w = np.array([self.TW1, 1.0])

        # 超前滞后对1：G1(s) = (1+sT1)/(1+sT2)
        num1 = np.array([self.T1, 1.0])
        den1 = np.array([self.T2, 1.0])

        # 超前滞后对2：G2(s) = (1+sT3)/(1+sT4)
        num2 = np.array([self.T3, 1.0])
        den2 = np.array([self.T4, 1.0])

        # 超前环节：G3(s) = (1+sT7)
        num3 = np.array([self.T7, 1.0])
        den3 = np.array([1.0])

        # 串联所有环节
        num = np.convolve(num_w, num1)
        num = np.convolve(num, num2)
        num = np.convolve(num, num3)
        num = num * self.KS1  # 乘以增益

        den = np.convolve(den_w, den1)
        den = np.convolve(den, den2)
        den = np.convolve(den, den3)

        return num, den

    def get_state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取 PSS2A 的状态空间表示（简化版）

        Returns:
            (A, B, C, D): 状态空间矩阵

        Note:
            状态变量（约6阶）：
            - x1, x2: 清洗环节状态（TW1, TW2/TW3）
            - x3, x4: 第一对超前滞后状态
            - x5, x6: 第二对超前滞后状态
        """
        num, den = self.get_transfer_function()

        # 使用 scipy.signal.tf2ss 转换
        # 注意：对于高阶系统，可能需要手动实现状态空间
        A, B, C, D = signal.tf2ss(num, den)

        return A, B, C, D

    def _pss2a_derivatives(self, delta_omega: float, P_gen: float,
                           x: np.ndarray) -> np.ndarray:
        """PSS2A 状态导数（按 ENTSO-E Fig 2-4）

        N=0, M=0, T6=0, TW4=0 参数下:
            Path 1 (speed):  ω → washout(TW1) → washout(TW2) → ×KS1
            Path 2 (power):  Pe → washout(TW3) → [TW4=0 bypass] → 1/(1+sT7) → ×KS2
            Sum → ×KS3 → lead-lag(T1/T2) → lead-lag(T3/T4) → limiter

        状态向量 (6 维):
            x1: washout TW1,  x2: washout TW2
            x3: washout TW3,  x4: 1/(1+sT7)
            x5: lead-lag T1/T2,  x6: lead-lag T3/T4
        """
        x1, x2, x3, x4, x5, x6 = x

        # === Path 1: 转速通道 ===
        # Washout 1: sTW1/(1+sTW1), 输入 = delta_omega
        dx1 = (delta_omega - x1) / self.TW1
        y1 = delta_omega - x1

        # Washout 2: sTW2/(1+sTW2), 输入 = y1
        dx2 = (y1 - x2) / self.TW2
        y2 = y1 - x2 if self.TW2 > 0 else y1

        path1 = self.KS1 * y2

        # === Path 2: 功率通道 ===
        # Washout 3: sTW3/(1+sTW3), 输入 = P_gen
        dx3 = (P_gen - x3) / self.TW3
        y3 = P_gen - x3

        # TW4=0: 第二个 washout 旁路
        y4 = y3

        # 1/(1+sT7): 输入 = y4
        dx4 = (y4 - x4) / self.T7
        y5 = x4

        path2 = self.KS2 * y5

        # === 求和与公共通路 ===
        V_S_raw = (path1 + path2) * self.KS3

        # Lead-lag 1: (1+sT1)/(1+sT2)
        dx5 = (V_S_raw - x5) / self.T2
        y6 = (self.T1 / self.T2) * V_S_raw + (1.0 - self.T1 / self.T2) * x5

        # Lead-lag 2: (1+sT3)/(1+sT4)
        dx6 = (y6 - x6) / self.T4
        # y7 不需要作为中间变量在这里返回

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6])

    def _pss2a_output(self, delta_omega: float, P_gen: float,
                      x: np.ndarray) -> float:
        """从状态计算 PSS2A 输出"""
        x1, x2, x3, x4, x5, x6 = x

        # Path 1
        y1 = delta_omega - x1
        y2 = y1 - x2
        path1 = self.KS1 * y2

        # Path 2
        y3 = P_gen - x3
        y5 = x4
        path2 = self.KS2 * y5

        V_S_raw = (path1 + path2) * self.KS3

        # Lead-lag chain
        y6 = (self.T1 / self.T2) * V_S_raw + (1.0 - self.T1 / self.T2) * x5
        y7 = (self.T3 / self.T4) * y6 + (1.0 - self.T3 / self.T4) * x6

        return np.clip(y7, self.VSTMIN, self.VSTMAX)

    def compute_stabilizing_signal(self, delta_omega: float, P_gen: float,
                                    dt: float,
                                    state: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """时域计算 PSS2A 稳定信号输出（欧拉法）

        按 ENTSO-E Fig 2-4 实现双输入 PSS2A 结构。

        Args:
            delta_omega: 转速偏差 Δω（标幺值）, IC1=1
            P_gen: 发电机电磁功率（标幺值）, IC2=3
            dt: 时间步长（秒）
            state: 当前状态向量 [x1..x6]，若为 None 则初始化为零

        Returns:
            (V_S, new_state): 稳定信号（标幺值）和新状态
        """
        if state is None:
            state = np.zeros(6)

        dx = self._pss2a_derivatives(delta_omega, P_gen, state)
        new_state = state + dt * dx
        V_S = self._pss2a_output(delta_omega, P_gen, new_state)
        return V_S, new_state

    def compute_stabilizing_signal_rk4(self, delta_omega: float, P_gen: float,
                                        dt: float,
                                        state: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
        """使用 RK4 方法计算 PSS2A 稳定信号输出

        按 ENTSO-E Fig 2-4 实现双输入 PSS2A 结构（IC1=1 转速, IC2=3 功率）。

        Args:
            delta_omega: 转速偏差 Δω（标幺值）
            P_gen: 发电机电磁功率（标幺值）
            dt: 时间步长（秒）
            state: 当前状态向量 [x1..x6]

        Returns:
            (V_S, new_state): 稳定信号和新状态
        """
        if state is None:
            state = np.zeros(6)

        def f(x):
            return self._pss2a_derivatives(delta_omega, P_gen, x)

        k1 = f(state)
        k2 = f(state + dt/2 * k1)
        k3 = f(state + dt/2 * k2)
        k4 = f(state + dt * k3)

        new_state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        V_S = self._pss2a_output(delta_omega, P_gen, new_state)
        return V_S, new_state
