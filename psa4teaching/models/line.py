"""
输电线路模型 (Line)
===================

输电线路的π型等值电路模型。

数学模型
--------
输电线路采用π型等值电路：

    I₁    ┌── Z = R + jX ──┐    I₂
   ○──────┤                 ├──────○
           └── Y/2    Y/2 ──┘
          │     │           │
         ═══   ═══         ═══
          ═     ═           ═
         ─┴─   ─┴─         ─┴─
          └     └           └

其中：
    Z = R + jX  —— 线路串联阻抗
    Y/2 = jωC/2 = jB/2  —— 线路对地电容的导纳

节点导纳矩阵贡献：
    Ybus[i,i] += Y/2 + 1/Z
    Ybus[j,j] += Y/2 + 1/Z
    Ybus[i,j] -= 1/Z
    Ybus[j,i] -= 1/Z

参考教材：陈珩《电力系统稳态分析》第一章 1.4节
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Line:
    """输电线路π型等值电路模型

    Attributes:
        from_bus: 送端节点编号
        to_bus: 受端节点编号
        R: 线路电阻（标幺值）
        X: 线路电抗（标幺值）
        B: 线路电纳（标幺值），对应全线充电电容的导纳 B = ωC
        name: 线路名称（可选）

    Note:
        线路参数均为标幺值，基准值需在外部统一确定。
        线路充电功率 Qc = V² × B

    Example:
        >>> line = Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02)
        >>> Y_series = line.get_series_admittance()  # 串联支路导纳
        >>> Y_shunt = line.get_shunt_admittance()    # 对地支路导纳
    """

    from_bus: int                      # 送端节点编号
    to_bus: int                        # 受端节点编号
    R: float                           # 电阻（标幺值）
    X: float                           # 电抗（标幺值）
    B: float                           # 电纳（标幺值），全线总充电电容对应的电纳
    name: Optional[str] = None         # 线路名称

    def __post_init__(self):
        """参数校验"""
        if self.R < 0:
            raise ValueError(f"电阻R必须非负，当前R={self.R}")
        if self.X <= 0:
            raise ValueError(f"电抗X必须为正，当前X={self.X}")

    @property
    def Z(self) -> complex:
        """线路串联阻抗 Z = R + jX

        Returns:
            复数形式的串联阻抗
        """
        return complex(self.R, self.X)

    def get_series_admittance(self) -> complex:
        """计算串联支路导纳

        Returns:
            Y_series = 1/Z = 1/(R + jX) = G + jB

        Note:
            数学公式：
                Y = 1/Z = (R - jX)/(R² + X²)
                G = R/(R² + X²)
                B = -X/(R² + X²)
        """
        Z = self.Z
        return 1.0 / Z

    def get_shunt_admittance(self) -> complex:
        """计算单侧对地支路导纳

        Returns:
            Y_shunt = jB/2（单侧对地电容的导纳）

        Note:
            π型等值电路中，每侧对地电容为全线电容的一半，
            故单侧对地导纳 Y_shunt = jB/2
        """
        return complex(0, self.B / 2)

    def get_ybus_contribution(self) -> Tuple[Tuple[int, int, complex], ...]:
        """计算该支路对节点导纳矩阵的贡献

        Returns:
            元组列表，每个元素为 (i, j, Y_ij)，表示Ybus[i,j]应累加Y_ij

        Note:
            贡献规则：
                Ybus[i,i] += Y_shunt + Y_series
                Ybus[j,j] += Y_shunt + Y_series
                Ybus[i,j] -= Y_series
                Ybus[j,i] -= Y_series

        Example:
            >>> line = Line(1, 2, 0.02, 0.1, 0.02)
            >>> contrib = line.get_ybus_contribution()
            >>> # contrib = [(1,1, Y11), (2,2, Y22), (1,2, Y12), (2,1, Y21)]
        """
        Y_series = self.get_series_admittance()
        Y_shunt = self.get_shunt_admittance()

        i, j = self.from_bus, self.to_bus

        return (
            (i, i, Y_shunt + Y_series),   # 对角元 i
            (j, j, Y_shunt + Y_series),   # 对角元 j
            (i, j, -Y_series),            # 非对角元 ij
            (j, i, -Y_series),            # 非对角元 ji
        )

    def compute_power_flow(self, Vi: complex, Vj: complex) -> Tuple[complex, complex]:
        """计算线路功率分布

        Args:
            Vi: 送端节点电压（复数形式）
            Vj: 受端节点电压（复数形式）

        Returns:
            (Sij, Sji): 送端和受端的复功率，Sij为从i端流入线路的功率

        Note:
            功率计算公式：
                Iij = (Vi - Vj)/Z + Vi × (jB/2)
                Sij = Vi × conj(Iij)

            线路损耗：
                ΔS = Sij + Sji
        """
        Y_series = self.get_series_admittance()
        Y_shunt = self.get_shunt_admittance()

        # 电流计算
        I_series = (Vi - Vj) * Y_series   # 串联支路电流
        I_shunt_i = Vi * Y_shunt          # i端对地支路电流
        Iij = I_series + I_shunt_i         # i端总流出电流

        I_shunt_j = Vj * Y_shunt          # j端对地支路电流
        Iji = -I_series + I_shunt_j        # j端总流出电流（注意方向）

        # 功率计算
        Sij = Vi * np.conj(Iij)
        Sji = Vj * np.conj(Iji)

        return Sij, Sji

    def compute_losses(self, Vi: complex, Vj: complex) -> complex:
        """计算线路功率损耗

        Args:
            Vi: 送端节点电压
            Vj: 受端节点电压

        Returns:
            ΔS = Sij + Sji，线路总有功和无功损耗
        """
        Sij, Sji = self.compute_power_flow(Vi, Vj)
        return Sij + Sji

    def __repr__(self) -> str:
        return (f"Line({self.from_bus}-{self.to_bus}, "
                f"Z={self.R:.4f}+j{self.X:.4f}, B={self.B:.4f})")