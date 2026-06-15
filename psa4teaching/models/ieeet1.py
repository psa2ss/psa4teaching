"""
IEEE Type 1 励磁系统模型 (IEEET1 Exciter Model)
=============================================

实现 IEEE Standard 421.5 Type 1 励磁系统，包含电压调节器、励磁机和
稳定反馈回路。IEEET1 是 Kundur 教材中最常用的励磁系统模型。

数学模型
--------

IEEET1 传递函数框图：

                        +---+    +--------+    +------+
    Vref ─→[+]─→[ Σ ]──→|KA |──→| 1+sTF  |──→|  1   |──→ Efd
           -↑           |1+sTA|  |  sKF   |   |KE+sTE|
            |           +---+    +--------+   +------+
            |              |                    |
            |   +----+     |                    |
            └───|V_S |─────┘                    |
                +----+                          |
            (来自PSS)                           |
                                                |
            Vt ─────────────────────────────────┘

    其中：
    - KA, TA: 电压调节器增益和时间常数
    - KE, TE: 励磁机常数和时间常数
    - KF, TF: 稳定反馈回路增益和时间常数
    - SE: 饱和函数 SE(Efd) = A_sat * exp(B_sat * Efd)

状态变量（四阶）:
    - x1: 电压调节器输出 VR
    - x2: 励磁机内部状态
    - x3: 励磁电压 Efd（励磁机输出）
    - x4: 稳定反馈信号 Rf

参考：
    - Kundur P. Power System Stability and Control, Ch.8
    - IEEE Standard 421.5-2016
    - PSS/E Model: IEEET1
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class IEEET1Params:
    """IEEE Type 1 励磁系统参数

    标准 IEEET1 模型包含一个电压调节器（超前-滞后）、励磁机
    和稳定反馈回路。

    Attributes:
        KA: 电压调节器增益（标幺值），通常 50-400
        TA: 电压调节器时间常数（秒），通常 0.01-0.1
        KE: 励磁机常数（标幺值），通常 -0.05 到 1.0
        TE: 励磁机时间常数（秒），通常 0.5-1.0
        KF: 稳定反馈增益（标幺值），通常 0.01-0.1
        TF: 稳定反馈时间常数（秒），通常 0.5-1.5
        SE1: 饱和系数（Efd_max 处），通常 0.05-0.15
        SE2: 饱和系数（0.75*Efd_max 处），通常 0.01-0.05
        VR_MIN: 调节器最小输出（标幺值），通常 -1.0
        VR_MAX: 调节器最大输出（标幺值），通常 5.0-7.0
        Efd_MIN: 最小励磁电压（标幺值），通常 0.0
        Efd_MAX: 最大励磁电压（标幺值），通常 4.0-7.0

    Note:
        - KA 越大，响应越快，但可能降低阻尼
        - KF 用于抑制励磁系统的负阻尼效应
        - KE 与励磁机负荷特性有关，可能为负值
        - 饱和系数 SE1, SE2 用于计算指数饱和函数

    Example:
        >>> # Kundur Ch.12 SMIB 典型参数
        >>> params = IEEET1Params(KA=200, TA=0.02, TE=0.5, KF=0.05, TF=1.0)
        >>> efd, state = params.compute(V_ref=1.05, V_measured=1.0, V_S=0.0, dt=0.01)
    """

    # 电压调节器参数
    KA: float = 200.0                             # 调节器增益
    TA: float = 0.02                              # 调节器时间常数（秒）

    # 励磁机参数
    KE: float = 1.0                               # 励磁机常数
    TE: float = 0.5                               # 励磁机时间常数（秒）

    # 稳定反馈回路
    KF: float = 0.05                              # 稳定反馈增益
    TF: float = 1.0                               # 稳定反馈时间常数（秒）

    # 饱和参数
    SE1: float = 0.10                             # 饱和系数（Efd_max 处）
    SE2: float = 0.03                             # 饱和系数（0.75*Efd_max 处）
    Efd_MAX: float = 5.0                          # 最大励磁电压

    # 限幅参数
    VR_MIN: float = -1.0                          # 调节器最小输出
    VR_MAX: float = 5.0                           # 调节器最大输出
    Efd_MIN: float = 0.0                          # 最小励磁电压

    # 内部计算饱和系数 A_sat, B_sat
    _A_sat: float = field(init=False, default=0.0, repr=False)
    _B_sat: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self):
        """计算指数饱和系数 A_sat 和 B_sat

        SE(Efd) = A_sat * exp(B_sat * Efd)

        利用两个已知点:
            SE(Efd_max) = SE1
            SE(0.75 * Efd_max) = SE2
        """
        E1 = self.Efd_MAX
        E2 = 0.75 * self.Efd_MAX

        if self.SE1 > 0 and self.SE2 > 0 and E1 > 0:
            # B_sat = ln(SE1/SE2) / (E1 - E2)
            self._B_sat = np.log(self.SE1 / self.SE2) / (E1 - E2)
            # A_sat = SE1 / exp(B_sat * E1)
            self._A_sat = self.SE1 / np.exp(self._B_sat * E1)
        else:
            self._A_sat = 0.0
            self._B_sat = 0.0

    def saturation(self, Efd: float) -> float:
        """计算饱和函数 SE(Efd)

        Args:
            Efd: 励磁电压（标幺值）

        Returns:
            饱和系数 SE(Efd)
        """
        if self._A_sat <= 0:
            return 0.0
        return self._A_sat * np.exp(self._B_sat * Efd)

    def compute(
        self,
        V_ref: float,
        V_measured: float,
        V_S: float,
        dt: float,
        state: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """时域计算 IEEET1 励磁系统输出（欧拉法）

        Args:
            V_ref: 参考电压（标幺值）
            V_measured: 实测端电压（标幺值）
            V_S: 附加稳定信号（标幺值，来自 PSS）
            dt: 时间步长（秒）
            state: 当前状态向量 [VR, x_ex, Efd, Rf]

        Returns:
            (Efd, new_state): 励磁电压（标幺值）和新状态

        Note:
            内部流程:
            1. 误差信号: Ve = V_ref - V_measured + V_S
            2. 调节器输出: VR（经 KA/(1+sTA)，带限幅）
            3. 励磁机输出: Efd（经 1/(KE+sTE) 和饱和反馈）
            4. 稳定反馈: Rf（经 sKF/(1+sTF)）
        """
        if state is None:
            # 初始化状态: [VR, x_ex, Efd, Rf]
            state = np.array([0.0, 0.0, 0.0, 0.0])

        VR, x_ex, Efd, Rf = state

        # 1. 电压误差
        Ve = V_ref - V_measured + V_S

        # 2. 电压调节器: dVR/dt = (KA*Ve - VR) / TA
        if self.TA > 0:
            dVR = (self.KA * Ve - VR) / self.TA
        else:
            dVR = 0.0
            VR = self.KA * Ve  # TA=0 时退化为纯比例

        # 3. 励磁机: dEfd/dt = (VR - KE*Efd - SE*Efd) / TE
        SE_fb = self.saturation(Efd)
        if self.TE > 0:
            dEfd = (VR - self.KE * Efd - SE_fb) / self.TE
        else:
            dEfd = 0.0

        # 4. 稳定反馈: dRf/dt = (KF*dEfd - Rf) / TF
        if self.TF > 0:
            dRf = (self.KF * dEfd / self.TE - Rf) / self.TF
        else:
            dRf = 0.0

        # 欧拉积分
        new_VR = VR + dt * dVR
        new_Efd = Efd + dt * dEfd
        new_Rf = Rf + dt * dRf

        # 限幅
        new_VR = np.clip(new_VR, self.VR_MIN, self.VR_MAX)
        new_Efd = np.clip(new_Efd, self.Efd_MIN, self.Efd_MAX)

        new_state = np.array([new_VR, new_Efd, new_Efd, new_Rf])

        return new_Efd, new_state

    def compute_rk4(
        self,
        V_ref: float,
        V_measured: float,
        V_S: float,
        dt: float,
        state: Optional[np.ndarray] = None
    ) -> Tuple[float, np.ndarray]:
        """使用 RK4 方法计算 IEEET1 励磁系统输出

        Args:
            V_ref: 参考电压（标幺值）
            V_measured: 实测端电压（标幺值）
            V_S: 附加稳定信号（标幺值）
            dt: 时间步长（秒）
            state: 当前状态向量 [VR, x_ex, Efd, Rf]

        Returns:
            (Efd, new_state): 励磁电压（标幺值）和新状态
        """
        if state is None:
            state = np.array([0.0, 0.0, 0.0, 0.0])

        Ve = V_ref - V_measured + V_S

        def derivatives(s):
            """状态导数函数 f(x, u)"""
            VR_s, Efd_s, _, Rf_s = s

            # 调节器导数
            if self.TA > 0:
                dVR_s = (self.KA * Ve - VR_s) / self.TA
            else:
                dVR_s = 0.0

            # 励磁机导数（含饱和）
            SE_b = self.saturation(Efd_s)
            if self.TE > 0:
                dEfd_s = (VR_s - self.KE * Efd_s - SE_b) / self.TE
            else:
                dEfd_s = 0.0

            # 稳定反馈导数
            if self.TF > 0:
                dRf_s = (self.KF * dEfd_s / self.TE - Rf_s) / self.TF if self.TE > 0 else 0.0
            else:
                dRf_s = 0.0

            return np.array([dVR_s, dEfd_s, dEfd_s, dRf_s])

        # RK4 积分
        k1 = derivatives(state)
        k2 = derivatives(state + dt / 2 * k1)
        k3 = derivatives(state + dt / 2 * k2)
        k4 = derivatives(state + dt * k3)

        new_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # 限幅
        new_state[0] = np.clip(new_state[0], self.VR_MIN, self.VR_MAX)
        new_state[1] = np.clip(new_state[1], self.Efd_MIN, self.Efd_MAX)
        new_state[2] = new_state[1]

        return new_state[1], new_state

    def get_transfer_function(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 IEEET1 简化传递函数（忽略饱和和非线性）

        Returns:
            (num, den): 传递函数分子和分母系数

        Note:
            简化传递函数（KE=0, 忽略饱和和限幅）:
            G(s) = KA * (1+sTF) / [(1+sTA) * (1+sTE) * (1+sTF) + KA*KF*s]
        """
        # 前向通路: KA * (1+sTF)
        num_forward = np.array([self.KA * self.TF, self.KA])

        # 反馈通路
        # 分母 = (1+sTA)(1+sTE)(1+sTF) + KA*KF*s
        den1 = np.array([self.TA, 1.0])
        den2 = np.array([self.TE, 1.0])
        den3 = np.array([self.TF, 1.0])
        den_fb = np.polymul(np.polymul(den1, den2), den3)
        # 加上 KA*KF*s 项
        den_fb[-2] += self.KA * self.KF  # s 项系数增加 KA*KF

        return num_forward, den_fb

    def get_state_space(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取 IEEET1 的状态空间表示（忽略饱和和限幅）

        Returns:
            (A, B, C, D): 状态空间矩阵

        Note:
            采用可控标准型实现。实际使用时建议直接使用
            compute() 或 compute_rk4() 方法获得非线性响应。
        """
        num, den = self.get_transfer_function()
        from scipy import signal as scipy_signal
        A, B, C, D = scipy_signal.tf2ss(num, den)
        return A, B, C, D

    def get_linearized_matrices(
        self,
        Efd0: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """获取 IEEET1 在工作点 Efd0 处的线性化状态矩阵

        用于小干扰稳定分析，构造 Heffron-Phillips 状态矩阵时使用。

        Args:
            Efd0: 稳态励磁电压（标幺值）

        Returns:
            (A_exc, B_exc, C_exc, D_exc): 线性化状态空间矩阵
                状态: [ΔVR, ΔEfd, ΔRf]^T
                输入: ΔVe = ΔV_ref - ΔV_measured + ΔV_S
                输出: ΔEfd

        Note:
            线性化考虑了饱和函数 SE(Efd) 在 Efd0 处的一阶导数:
            SE'(Efd) = A_sat * B_sat * exp(B_sat * Efd)
        """
        # 饱和函数在工作点处的斜率
        dSE_dEfd = self._A_sat * self._B_sat * np.exp(self._B_sat * Efd0)

        # 等效反馈: KE_eff = KE + dSE_dEfd
        KE_eff = self.KE + dSE_dEfd

        # 状态矩阵: 3×3
        A_exc = np.zeros((3, 3))

        if self.TA > 0:
            A_exc[0, 0] = -1.0 / self.TA
        if self.TE > 0:
            A_exc[1, 0] = 1.0 / self.TE
            A_exc[1, 1] = -KE_eff / self.TE
        if self.TF > 0 and self.TE > 0:
            A_exc[2, 0] = self.KF / (self.TF * self.TE)
            A_exc[2, 1] = -self.KF * KE_eff / (self.TF * self.TE)
            A_exc[2, 2] = -1.0 / self.TF

        # 输入矩阵: 3×1
        B_exc = np.zeros((3, 1))
        if self.TA > 0:
            B_exc[0, 0] = self.KA / self.TA

        # 输出矩阵: 1×3, 输出 Efd = 状态2
        C_exc = np.array([[0.0, 1.0, 0.0]])

        # 直馈矩阵
        D_exc = np.zeros((1, 1))

        return A_exc, B_exc, C_exc, D_exc


__all__ = ["IEEET1Params"]
