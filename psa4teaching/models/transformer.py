"""
变压器模型 (Transformer)
========================

双绕组变压器等值电路模型，支持非标准变比。

数学模型
--------
变压器采用π型等值电路模型，支持非标准变比k（非标准变比在高压侧）：

    非标准变比k的定义：
        k = U₁N/U₂N × U₂/U₁

    当k≠1时，变压器π型等值电路：

         I₁    k        I₂
        ○─────┬─────────○
              │
             ZT
              │
        ═══════════════════
              │
             ─┴─
              ╧
              └

    等值导纳：
        YT = 1/ZT = GT + jBT（短路导纳）
        YT' = k(k-1)YT（高压侧对地支路）
        YT'' = (1-k)YT（低压侧对地支路）

节点导纳矩阵贡献：
    Ybus[i,i] += YT/k²
    Ybus[j,j] += YT + (k-1)²YT/k²
    Ybus[i,j] -= YT/k
    Ybus[j,i] -= YT/k

参考教材：陈珩《电力系统稳态分析》第一章 1.5节
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class Transformer:
    """双绕组变压器模型

    Attributes:
        from_bus: 高压侧节点编号
        to_bus: 低压侧节点编号
        RT: 变压器电阻（标幺值，归算到低压侧）
        XT: 变压器电抗（标幺值，归算到低压侧）
        k: 非标准变比（高压侧实际电压/额定电压），默认为1.0
        GT: 变压器空载电导（标幺值，表示铁损），通常很小或忽略
        BT: 变压器空载电纳（标幺值，表示励磁电流），通常为负（感性）
        name: 变压器名称（可选）

    Note:
        - RT和XT通常由短路试验得到
        - GT和BT由空载试验得到，一般可忽略
        - 非标准变比k调整高压侧电压

    Example:
        >>> # 标准变比变压器
        >>> tx1 = Transformer(from_bus=1, to_bus=2, RT=0.002, XT=0.1, k=1.0)
        >>> # 非标准变比变压器（高压侧电压偏高5%）
        >>> tx2 = Transformer(from_bus=1, to_bus=2, RT=0.002, XT=0.1, k=1.05)
    """

    from_bus: int                          # 高压侧节点编号
    to_bus: int                            # 低压侧节点编号
    RT: float                              # 电阻（标幺值）
    XT: float                              # 电抗（标幺值）
    k: float = 1.0                         # 非标准变比
    GT: float = 0.0                        # 空载电导
    BT: float = 0.0                        # 空载电纳（通常为负）
    name: Optional[str] = None             # 变压器名称

    def __post_init__(self):
        """参数校验"""
        if self.RT < 0:
            raise ValueError(f"电阻RT必须非负，当前RT={self.RT}")
        if self.XT <= 0:
            raise ValueError(f"电抗XT必须为正，当前XT={self.XT}")
        if self.k <= 0:
            raise ValueError(f"非标准变比k必须为正，当前k={self.k}")

    @property
    def ZT(self) -> complex:
        """变压器短路阻抗 ZT = RT + jXT

        Returns:
            复数形式的短路阻抗
        """
        return complex(self.RT, self.XT)

    @property
    def YT(self) -> complex:
        """变压器短路导纳 YT = 1/ZT

        Returns:
            复数形式的短路导纳
        """
        return 1.0 / self.ZT if self.ZT != 0 else complex(0, 0)

    @property
    def YM(self) -> complex:
        """变压器励磁导纳 YM = GT + jBT

        Returns:
            复数形式的励磁导纳
        """
        return complex(self.GT, self.BT)

    def get_ybus_contribution(self) -> Tuple[Tuple[int, int, complex], ...]:
        """计算变压器支路对节点导纳矩阵的贡献

        Returns:
            元组列表，每个元素为 (i, j, Y_ij)

        Note:
            变压器π型等值电路的导纳矩阵贡献：
                k≠1时（非标准变比在高压侧i）：
                    Ybus[i,i] += (YT/k² + (k-1)YT/k²)
                    Ybus[j,j] += YT + (1-k)YT
                    Ybus[i,j] -= YT/k
                    Ybus[j,i] -= YT/k

                k=1时：
                    Ybus[i,i] += YT
                    Ybus[j,j] += YT
                    Ybus[i,j] -= YT
                    Ybus[j,i] -= YT
        """
        YT = self.YT
        k = self.k
        i, j = self.from_bus, self.to_bus

        if abs(k - 1.0) < 1e-10:
            # 标准变比，简化计算
            return (
                (i, i, YT),
                (j, j, YT),
                (i, j, -YT),
                (j, i, -YT),
            )
        else:
            # 非标准变比
            # π型等值电路参数
            Yii = YT / k            # 高压侧串联支路
            Yjj = YT                # 低压侧串联支路
            Yij = -YT / k           # 互导纳

            # 对地支路（由于非标准变比引入）
            # 高压侧对地：k(k-1)YT/k² = (k-1)YT/k
            # 低压侧对地：(1-k)YT
            Y_shunt_i = (k - 1) * YT / k
            Y_shunt_j = (1 - k) * YT

            return (
                (i, i, Yii + Y_shunt_i),
                (j, j, Yjj + Y_shunt_j),
                (i, j, Yij),
                (j, i, Yij),
            )

    def compute_power_flow(self, Vi: complex, Vj: complex) -> Tuple[complex, complex]:
        """计算变压器功率分布

        Args:
            Vi: 高压侧节点电压
            Vj: 低压侧节点电压

        Returns:
            (Sij, Sji): 高压侧和低压侧的复功率
        """
        k = self.k
        YT = self.YT

        # 理想变压器变比
        Vi_internal = Vi / k  # 归算到低压侧的电压

        # 电流计算（归算到低压侧）
        Iij = (Vi_internal - Vj) * YT

        # 功率计算
        Sij = Vi * np.conj(Iij / k)  # 高压侧功率
        Sji = Vj * np.conj(-Iij)     # 低压侧功率

        return Sij, Sji

    def __repr__(self) -> str:
        return (f"Transformer({self.from_bus}-{self.to_bus}, "
                f"ZT={self.RT:.4f}+j{self.XT:.4f}, k={self.k:.3f})")