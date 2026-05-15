"""
负荷模型 (Load)
===============

电力系统负荷的数学模型。

数学模型
--------
1. 恒功率模型（静态负荷）：
    P = P0, Q = Q0
    负荷功率不随电压变化

2. 恒阻抗模型：
    P = P0 × (V/V0)², Q = Q0 × (V/V0)²
    负荷阻抗恒定，功率随电压平方变化
    Z = V0² / (P0 - jQ0)

3. 恒电流模型：
    P = P0 × (V/V0), Q = Q0 × (V/V0)
    负荷电流恒定，功率随电压线性变化

4. ZIP模型（综合模型）：
    P = P0 × [αp(V/V0)² + βp(V/V0) + γp]
    Q = Q0 × [αq(V/V0)² + βq(V/V0) + γq]
    其中：αp + βp + γp = 1，αq + βq + γq = 1

参考教材：陈珩《电力系统稳态分析》第一章 1.6节
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum
import numpy as np


class LoadModel(Enum):
    """负荷模型类型枚举

    Attributes:
        CONSTANT_POWER: 恒功率模型（PQ模型）
        CONSTANT_IMPEDANCE: 恒阻抗模型（Z模型）
        CONSTANT_CURRENT: 恒电流模型（I模型）
        ZIP: ZIP综合模型
    """
    CONSTANT_POWER = "PQ"           # 恒功率
    CONSTANT_IMPEDANCE = "Z"        # 恒阻抗
    CONSTANT_CURRENT = "I"          # 恒电流
    ZIP = "ZIP"                     # ZIP综合模型


@dataclass
class Load:
    """电力系统负荷模型

    Attributes:
        bus: 负荷所在节点编号
        name: 负荷名称（可选）
        P0: 额定有功功率（标幺值），正值表示消耗
        Q0: 额定无功功率（标幺值），正值表示消耗
        V0: 额定电压（标幺值），默认1.0
        model_type: 负荷模型类型

        ZIP模型系数（仅当model_type=ZIP时有效）：
        alpha_p, beta_p, gamma_p: 有功功率的恒阻抗、恒电流、恒功率比例
        alpha_q, beta_q, gamma_q: 无功功率的恒阻抗、恒电流、恒功率比例

    Note:
        潮流计算中通常使用恒功率模型
        短路计算和稳定计算中可能需要使用恒阻抗或ZIP模型

    Example:
        >>> # 恒功率负荷
        >>> load1 = Load(bus=3, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_POWER)
        >>> # ZIP模型负荷
        >>> load2 = Load(bus=4, P0=1.0, Q0=0.4, model_type=LoadModel.ZIP,
        ...              alpha_p=0.3, beta_p=0.3, gamma_p=0.4)
    """

    bus: int                                      # 负荷所在节点编号
    name: Optional[str] = None                   # 负荷名称

    # 基本参数
    P0: float = 0.0                              # 额定有功功率（标幺值）
    Q0: float = 0.0                              # 额定无功功率（标幺值）
    V0: float = 1.0                              # 额定电压（标幺值）

    # 模型类型
    model_type: LoadModel = LoadModel.CONSTANT_POWER

    # ZIP模型系数
    alpha_p: float = 0.0                         # 有功恒阻抗比例
    beta_p: float = 0.0                          # 有功恒电流比例
    gamma_p: float = 1.0                         # 有功恒功率比例
    alpha_q: float = 0.0                         # 无功恒阻抗比例
    beta_q: float = 0.0                          # 无功恒电流比例
    gamma_q: float = 1.0                         # 无功恒功率比例

    def __post_init__(self):
        """参数校验"""
        # 校验ZIP模型系数
        if self.model_type == LoadModel.ZIP:
            total_p = self.alpha_p + self.beta_p + self.gamma_p
            total_q = self.alpha_q + self.beta_q + self.gamma_q
            if abs(total_p - 1.0) > 1e-6:
                raise ValueError(f"ZIP模型有功系数之和应为1，当前为{total_p}")
            if abs(total_q - 1.0) > 1e-6:
                raise ValueError(f"ZIP模型无功系数之和应为1，当前为{total_q}")

    def get_power_at_voltage(self, V: float) -> tuple:
        """计算给定电压下的负荷功率

        Args:
            V: 节点电压幅值（标幺值）

        Returns:
            (P, Q): 有功和无功功率（标幺值）

        Note:
            不同模型类型：
                恒功率：P = P0, Q = Q0
                恒阻抗：P = P0×(V/V0)², Q = Q0×(V/V0)²
                恒电流：P = P0×(V/V0), Q = Q0×(V/V0)
                ZIP：P = P0×[αp(V/V0)² + βp(V/V0) + γp]
        """
        v_ratio = V / self.V0
        v_ratio_sq = v_ratio ** 2

        if self.model_type == LoadModel.CONSTANT_POWER:
            P = self.P0
            Q = self.Q0

        elif self.model_type == LoadModel.CONSTANT_IMPEDANCE:
            P = self.P0 * v_ratio_sq
            Q = self.Q0 * v_ratio_sq

        elif self.model_type == LoadModel.CONSTANT_CURRENT:
            P = self.P0 * v_ratio
            Q = self.Q0 * v_ratio

        elif self.model_type == LoadModel.ZIP:
            P = self.P0 * (self.alpha_p * v_ratio_sq +
                           self.beta_p * v_ratio +
                           self.gamma_p)
            Q = self.Q0 * (self.alpha_q * v_ratio_sq +
                           self.beta_q * v_ratio +
                           self.gamma_q)
        else:
            raise ValueError(f"未知的负荷模型类型: {self.model_type}")

        return P, Q

    def get_admittance(self, V0: float = 1.0) -> complex:
        """计算恒阻抗模型的等效导纳

        Args:
            V0: 参考电压（标幺值），默认1.0

        Returns:
            Y_load = P/V² - jQ/V²（负荷等效导纳，消耗功率为正）

        Note:
            仅对恒阻抗模型有效
            等效导纳计算：
                S = P + jQ = V × conj(I) = V × conj(V × Y) = V² × conj(Y)
                Y = (P - jQ) / V²
        """
        V_sq = V0 ** 2
        if V_sq == 0:
            return complex(0, 0)

        G = self.P0 / V_sq  # 电导
        B = -self.Q0 / V_sq  # 电纳（消耗无功为正，故取负）

        return complex(G, B)

    def get_current(self, V: complex) -> complex:
        """计算给定电压下的负荷电流

        Args:
            V: 节点电压（复数，标幺值）

        Returns:
            I: 负荷电流（复数，标幺值），流入负荷为正

        Note:
            I = conj(S/V) = conj((P+jQ)/V)
        """
        P, Q = self.get_power_at_voltage(abs(V))
        S = complex(P, Q)
        if abs(V) < 1e-10:
            return complex(0, 0)
        I = np.conj(S / V)
        return I

    def __repr__(self) -> str:
        return (f"Load({self.name or 'Load'}, bus={self.bus}, "
                f"P0={self.P0:.3f}, Q0={self.Q0:.3f}, model={self.model_type.value})")