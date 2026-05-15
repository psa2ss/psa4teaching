"""
节点模型 (Bus)
==============

定义电力系统节点的类型和属性。

数学模型
--------
节点按给定量的不同分为三类：
1. PQ节点：给定有功功率P和无功功率Q，求节点电压幅值V和相角δ
2. PV节点：给定有功功率P和电压幅值V，求无功功率Q和电压相角δ
3. 平衡节点：给定电压幅值V和相角δ（通常设为1∠0°），求注入功率P、Q

参考教材：陈珩《电力系统稳态分析》第三章 3.1节
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class BusType(Enum):
    """节点类型枚举

    Attributes:
        PQ: PQ节点（负荷节点）
        PV: PV节点（发电机节点）
        SLACK: 平衡节点（参考节点）
    """
    PQ = "PQ"       # 负荷节点
    PV = "PV"       # 发电机节点，控制电压
    SLACK = "SLACK" # 平衡节点


@dataclass
class Bus:
    """电力系统节点模型

    存储节点的编号、类型、电压及功率参数。

    Attributes:
        number: 节点编号（从1开始）
        name: 节点名称（可选）
        bus_type: 节点类型（PQ/PV/SLACK）
        V: 电压幅值（标幺值），默认1.0
        delta: 电压相角（弧度），默认0.0
        P: 有功注入功率（标幺值），发电为正，负荷为负
        Q: 无功注入功率（标幺值），发电为正，负荷为负
        P_specified: 指定的有功功率（PV和PQ节点）
        Q_specified: 指定的无功功率（PQ节点）
        V_specified: 指定的电压幅值（PV节点和平衡节点）
        Q_min: 无功出力下限（PV节点）
        Q_max: 无功出力上限（PV节点）

    Example:
        >>> # 创建平衡节点
        >>> slack = Bus(number=1, name="Slack", bus_type=BusType.SLACK,
        ...             V=1.05, delta=0.0)
        >>> # 创建PV节点
        >>> gen = Bus(number=2, name="Gen1", bus_type=BusType.PV,
        ...           V_specified=1.02, P_specified=0.5)
        >>> # 创建PQ节点
        >>> load = Bus(number=3, name="Load1", bus_type=BusType.PQ,
        ...            P_specified=-0.8, Q_specified=-0.3)
    """

    number: int                          # 节点编号
    name: Optional[str] = None           # 节点名称
    bus_type: BusType = BusType.PQ        # 节点类型

    # 电压（计算结果）
    V: float = 1.0                       # 电压幅值（标幺值）
    delta: float = 0.0                   # 电压相角（弧度）

    # 功率（计算结果）
    P: float = 0.0                       # 有功注入功率（标幺值）
    Q: float = 0.0                       # 无功注入功率（标幺值）

    # 指定参数（根据节点类型不同含义不同）
    P_specified: float = 0.0             # 指定有功功率
    Q_specified: float = 0.0             # 指定无功功率
    V_specified: float = 1.0              # 指定电压幅值

    # PV节点无功出力限制
    Q_min: float = -999.0                # 无功出力下限
    Q_max: float = 999.0                 # 无功出力上限

    def __post_init__(self):
        """初始化后处理：根据节点类型设置默认值"""
        if self.bus_type == BusType.SLACK:
            # 平衡节点：电压幅值和相角已知
            self.V = self.V_specified
            self.delta = 0.0  # 平衡节点相角通常设为0
        elif self.bus_type == BusType.PV:
            # PV节点：电压幅值已知
            self.V = self.V_specified

    def get_active_power_mismatch(self) -> float:
        """计算有功功率偏差

        Returns:
            ΔP = P_specified - P（指定值减去计算值）

        Note:
            仅对PQ节点和PV节点有效
        """
        if self.bus_type == BusType.SLACK:
            return 0.0
        return self.P_specified - self.P

    def get_reactive_power_mismatch(self) -> float:
        """计算无功功率偏差

        Returns:
            ΔQ = Q_specified - Q（指定值减去计算值）

        Note:
            仅对PQ节点有效
        """
        if self.bus_type != BusType.PQ:
            return 0.0
        return self.Q_specified - self.Q

    def check_q_limit(self) -> bool:
        """检查PV节点无功出力是否越限

        Returns:
            True表示越限，需要转换为PQ节点

        Note:
            仅对PV节点有效
        """
        if self.bus_type != BusType.PV:
            return False
        return self.Q < self.Q_min or self.Q > self.Q_max

    def __repr__(self) -> str:
        return (f"Bus({self.number}, '{self.name or ''}', {self.bus_type.value}, "
                f"V={self.V:.4f}∠{self.delta*180/3.14159:.2f}°)")