"""
发电机模型 (Generator)
=====================

发电机暂态和次暂态模型，用于潮流计算、短路计算和稳定计算。

数学模型
--------
1. 潮流计算中的发电机模型：
    - PV节点：给定P和V，计算Q和δ
    - 平衡节点：给定V和δ，计算P和Q

2. 短路计算中的发电机模型：
    - 次暂态模型：E" = V + jX"d × I"（次暂态电势）
    - 暂态模型：E' = V + jX'd × I'（暂态电势）

3. 稳定计算中的发电机模型：
    - 经典模型：E'恒定，二阶模型
        dδ/dt = (ω - ωs) × ωs
        dω/dt = (Pm - Pe - D(ω - ωs)) / (2H)

    - 详细模型：包含励磁系统、调速系统、PSS等
        状态变量：δ, ω, Eq', Ed', Efd, Vm, ...

参考教材：
    - 陈珩《电力系统稳态分析》第六章
    - 李光琦《电力系统暂态分析》第二章
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class GeneratorModelType(Enum):
    """发电机模型类型枚举

    Attributes:
        CLASSIC: 经典模型（E'恒定，二阶模型）
        CLASSICAL: 同CLASSIC
        DETAIL: 详细模型（含励磁、调速）
        DETAILED: 同DETAIL
    """
    CLASSIC = "classic"        # 经典模型
    CLASSICAL = "classic"      # 别名
    DETAIL = "detail"          # 详细模型
    DETAILED = "detail"        # 别名


@dataclass
class Generator:
    """发电机模型

    Attributes:
        bus: 发电机所在节点编号
        name: 发电机名称（可选）
        Sb: 发电机额定容量（MVA），用于计算标幺值
        Vb: 发电机额定电压（kV），用于计算标幺值
        Xd: 同步电抗（标幺值），用于稳态分析
        Xd_prime: 暂态电抗Xd'（标幺值），用于暂态稳定分析
        Xd_doubleprime: 次暂态电抗Xd"（标幺值），用于短路计算
        Xq: q轴同步电抗（标幺值）
        Xq_prime: q轴暂态电抗Xq'（标幺值）
        Xq_doubleprime: q轴次暂态电抗Xq"（标幺值）
        Td0_prime: d轴暂态时间常数Td0'（秒）
        Td0_doubleprime: d轴次暂态时间常数Td0"（秒）
        Tq0_prime: q轴暂态时间常数Tq0'（秒）
        Tq0_doubleprime: q轴次暂态时间常数Tq0"（秒）
        H: 惯性时间常数（秒），通常2-10秒
        D: 阻尼系数（标幺值），通常0-2
        model_type: 发电机模型类型

    Note:
        - Xd' 约为 Xd 的 20%~30%
        - Xd" 约为 Xd' 的 60%~80%
        - H 通常为 2-10 秒
        - D 通常为 0-2

    Example:
        >>> # 创建经典模型发电机（用于暂态稳定）
        >>> gen = Generator(bus=1, name="G1", Xd_prime=0.3, H=6.0, D=1.0)
        >>> # 创建详细模型发电机
        >>> gen_detail = Generator(
        ...     bus=2, name="G2", Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.2,
        ...     Td0_prime=8.0, H=6.0, model_type=GeneratorModelType.DETAIL
        ... )
    """

    bus: int                                      # 发电机所在节点编号
    name: Optional[str] = None                   # 发电机名称

    # 额定参数
    Sb: float = 100.0                            # 额定容量（MVA）
    Vb: float = 10.5                             # 额定电压（kV）

    # 电抗参数（标幺值）
    Xd: float = 1.0                              # d轴同步电抗
    Xd_prime: float = 0.3                        # d轴暂态电抗Xd'
    Xd_doubleprime: float = 0.2                  # d轴次暂态电抗Xd"
    Xq: Optional[float] = None                   # q轴同步电抗（凸极机）
    Xq_prime: Optional[float] = None             # q轴暂态电抗Xq'
    Xq_doubleprime: Optional[float] = None       # q轴次暂态电抗Xq"

    # 时间常数（秒）
    Td0_prime: float = 8.0                       # d轴暂态时间常数Td0'
    Td0_doubleprime: float = 0.03                 # d轴次暂态时间常数Td0"
    Tq0_prime: Optional[float] = None            # q轴暂态时间常数Tq0'
    Tq0_doubleprime: Optional[float] = None      # q轴次暂态时间常数Tq0"

    # 机械参数
    H: float = 6.0                               # 惯性时间常数（秒）
    D: float = 0.0                               # 阻尼系数
    TA: float = 8.0                               # 惯性时间常数含原动机（秒）
    GD2: float = 162.114                         # 飞轮力矩（Mp·m²）
    inertia: float = 40.528                      # 转动惯量（t·m²）

    # 模型类型
    model_type: GeneratorModelType = GeneratorModelType.CLASSIC

    # 运行状态（潮流计算后更新）
    P: float = 0.0                               # 有功出力（标幺值）
    Q: float = 0.0                               # 无功出力（标幺值）
    V: complex = 1.0 + 0j                        # 端电压（复数，标幺值）

    def __post_init__(self):
        """初始化后处理"""
        # 如果Xq未指定，假设为隐极机（Xq = Xd）
        if self.Xq is None:
            self.Xq = self.Xd
        if self.Xq_prime is None:
            self.Xq_prime = self.Xq
        if self.Xq_doubleprime is None:
            self.Xq_doubleprime = self.Xq

    def get_transient_reactance(self) -> float:
        """获取暂态电抗

        Returns:
            Xd'（暂态电抗）

        Note:
            用于暂态稳定计算中的经典模型
        """
        return self.Xd_prime

    def get_subtransient_reactance(self) -> float:
        """获取次暂态电抗

        Returns:
            Xd"（次暂态电抗）

        Note:
            用于短路电流计算
        """
        return self.Xd_doubleprime

    def compute_transient_emf(self, V: complex, I: complex) -> complex:
        """计算暂态电势E'

        Args:
            V: 发电机端电压（复数，标幺值）
            I: 发电机电流（复数，标幺值，流出为正）

        Returns:
            E' = V + jXd' × I

        Note:
            数学公式：
                E' = V + jXd' × I
                其中电流方向：流出发电机为正
        """
        Xd_prime = self.Xd_prime
        return V + 1j * Xd_prime * I

    def compute_subtransient_emf(self, V: complex, I: complex) -> complex:
        """计算次暂态电势E"

        Args:
            V: 发电机端电压（复数，标幺值）
            I: 发电机电流（复数，标幺值，流出为正）

        Returns:
            E" = V + jXd" × I

        Note:
            用于短路电流计算，假设短路瞬间E"不变
        """
        Xd_doubleprime = self.Xd_doubleprime
        return V + 1j * Xd_doubleprime * I

    def get_inertia_constant(self) -> float:
        """获取惯性时间常数H

        Returns:
            H（秒）

        Note:
            物理意义：发电机在额定转速下，动能与额定容量之比
            典型值：2-10秒
        """
        return self.H

    def get_TA(self) -> float:
        """获取含原动机的惯性时间常数

        Returns:
            TA（秒）

        Note:
            包含原动机在内的时间常数，用于详细动态仿真
        """
        return self.TA

    def get_damping_coefficient(self) -> float:
        """获取阻尼系数D

        Returns:
            D（标幺值）

        Note:
            表示机械阻尼和电气阻尼的综合效应
            典型值：0-2
        """
        return self.D

    def get_mechanical_time_constants(self) -> dict:
        """获取所有机械时间常数

        Returns:
            包含 H, TA, GD2, inertia 的字典
        """
        return {
            'H': self.H,
            'TA': self.TA,
            'GD2': self.GD2,
            'inertia': self.inertia
        }

    def __repr__(self) -> str:
        return (f"Generator({self.name or 'G'}, bus={self.bus}, "
                f"Xd'={self.Xd_prime:.3f}, H={self.H:.1f}s)")