"""
序网模型 (Sequence Networks)
=============================

构造正序、负序和零序网络。

数学模型
--------
不对称短路分析需要将三相网络分解为三个序网络：

1. 正序网络：与正常运行网络相同
   - 发电机用次暂态电势E"和次暂态电抗Xd"
   - 负荷用恒定阻抗

2. 负序网络：
   - 发电机用负序电抗X2
   - 线路参数与正序相同
   - 变压器负序参数等于正序参数

3. 零序网络：
   - 发电机零序电抗X0（通常很小或为∞，表示无中性点接地）
   - 线路零序参数与正序不同（需单独给出）
   - 变压器零序参数取决于接线方式

不对称短路边界条件：
    单相接地（a相）：Ib = 0, Ic = 0, Va = 0
        → 正序、负序、零序网络串联

    两相短路（b、c相）：Ia = 0, Ib + Ic = 0, Vb = Vc
        → 正序、负序网络并联

    两相接地（b、c相）：Ia = 0, Vb = 0, Vc = 0
        → 正序网络与负序、零序网络并联

参考教材：
    - 陈珩《电力系统稳态分析》第六章 6.4-6.5节
    - 李光琦《电力系统暂态分析》第二章 2.2-2.4节
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator
from psa4teaching.network.ybus import build_ybus, YBusResult


@dataclass
class SequenceNetworks:
    """序网模型集合

    Attributes:
        positive: 正序网络（Ybus1）
        negative: 负序网络（Ybus2）
        zero: 零序网络（Ybus0）
        Zbus1: 正序阻抗矩阵
        Zbus2: 负序阻抗矩阵
        Zbus0: 零序阻抗矩阵
        bus_indices: 节点编号到索引的映射
    """
    positive: YBusResult
    negative: YBusResult
    zero: YBusResult
    Zbus1: NDArray[np.complex128]
    Zbus2: NDArray[np.complex128]
    Zbus0: NDArray[np.complex128]
    bus_indices: Dict[int, int]


@dataclass
class ZeroSequenceLine:
    """零序线路参数

    Attributes:
        from_bus: 送端节点编号
        to_bus: 受端节点编号
        R0: 零序电阻（标幺值）
        X0: 零序电抗（标幺值）
        B0: 零序电纳（标幺值）
    """
    from_bus: int
    to_bus: int
    R0: float
    X0: float
    B0: float


@dataclass
class ZeroSequenceTransformer:
    """零序变压器参数

    Attributes:
        from_bus: 高压侧节点编号
        to_bus: 低压侧节点编号
        XT0: 零序短路电抗（标幺值）
        RT0: 零序短路电阻（标幺值）
        connection: 接线方式，影响零序通路
            "Ynd" - YN/d接线（零序可通过）
            "Yyn" - YN/yn接线（零序可通过，需中性点接地）
            "Yy"  - Y/y接线（零序不通）
            "Dy"  - D/y接线（零序在D侧不通）
    """
    from_bus: int
    to_bus: int
    XT0: float = 0.1
    RT0: float = 0.0
    connection: str = "Ynd"


def build_positive_sequence_network(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    bus_numbers: Optional[List[int]] = None
) -> Tuple[YBusResult, NDArray[np.complex128]]:
    """构造正序网络并求Zbus

    Args:
        buses: 节点列表
        lines: 输电线路列表（正序参数）
        transformers: 变压器列表
        generators: 发电机列表（用于修改发电机节点导纳）
        bus_numbers: 节点编号列表（可选）

    Returns:
        (Ybus_result, Zbus1): 正序导纳矩阵结果和正序阻抗矩阵

    Note:
        正序网络与正常运行的潮流网络基本相同。
        区别在于发电机用次暂态电势和次暂态电抗表示：
        发电机节点导纳增加 1/jXd"
    """
    if bus_numbers is None:
        bus_numbers = [bus.number for bus in buses]

    # 构造正序Ybus
    ybus_result = build_ybus(lines, transformers, bus_numbers=bus_numbers)

    # 修改发电机节点导纳（加入次暂态电抗）
    Ybus1 = ybus_result.Ybus.copy()
    for gen in generators:
        if gen.bus in ybus_result.bus_indices:
            idx = ybus_result.bus_indices[gen.bus]
            # 发电机导纳 1/jXd"
            Y_gen = 1.0 / (1j * gen.Xd_doubleprime)
            Ybus1[idx, idx] += Y_gen

    # 更新Ybus_result
    updated_result = YBusResult(
        Ybus=Ybus1,
        n_bus=ybus_result.n_bus,
        bus_indices=ybus_result.bus_indices,
        G=Ybus1.real,
        B=Ybus1.imag
    )

    # 求Zbus1
    try:
        Zbus1 = np.linalg.inv(Ybus1)
    except np.linalg.LinAlgError:
        raise ValueError("正序Ybus矩阵奇异")

    return updated_result, Zbus1


def build_negative_sequence_network(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    bus_numbers: Optional[List[int]] = None
) -> Tuple[YBusResult, NDArray[np.complex128]]:
    """构造负序网络并求Zbus

    Args:
        buses: 节点列表
        lines: 输电线路列表（负序参数通常等于正序参数）
        transformers: 变压器列表
        generators: 发电机列表
        bus_numbers: 节点编号列表（可选）

    Returns:
        (Ybus_result, Zbus2): 负序导纳矩阵结果和负序阻抗矩阵

    Note:
        负序网络特点：
        - 线路的负序参数等于正序参数（R2=R1, X2=X1）
        - 变压器的负序参数等于正序参数
        - 发电机用负序电抗X2（通常X2 ≈ Xd"）
        - 负序网络中无电源电势
    """
    if bus_numbers is None:
        bus_numbers = [bus.number for bus in buses]

    # 负序线路参数等于正序参数
    ybus_result = build_ybus(lines, transformers, bus_numbers=bus_numbers)

    # 修改发电机节点导纳（使用负序电抗）
    Ybus2 = ybus_result.Ybus.copy()
    for gen in generators:
        if gen.bus in ybus_result.bus_indices:
            idx = ybus_result.bus_indices[gen.bus]
            # 发电机负序导纳 1/jX2
            X2 = gen.Xq_doubleprime if gen.Xq_doubleprime else gen.Xd_doubleprime
            Y_gen = 1.0 / (1j * X2)
            Ybus2[idx, idx] += Y_gen

    updated_result = YBusResult(
        Ybus=Ybus2,
        n_bus=ybus_result.n_bus,
        bus_indices=ybus_result.bus_indices,
        G=Ybus2.real,
        B=Ybus2.imag
    )

    try:
        Zbus2 = np.linalg.inv(Ybus2)
    except np.linalg.LinAlgError:
        raise ValueError("负序Ybus矩阵奇异")

    return updated_result, Zbus2


def build_zero_sequence_network(
    buses: List[Bus],
    z_lines: List['ZeroSequenceLine'],
    z_transformers: List['ZeroSequenceTransformer'],
    generators: List[Generator],
    bus_numbers: Optional[List[int]] = None
) -> Tuple[YBusResult, NDArray[np.complex128]]:
    """构造零序网络并求Zbus

    Args:
        buses: 节点列表
        z_lines: 零序线路参数列表
        z_transformers: 零序变压器参数列表
        generators: 发电机列表（用于确定零序通路）
        bus_numbers: 节点编号列表（可选）

    Returns:
        (Ybus_result, Zbus0): 零序导纳矩阵结果和零序阻抗矩阵

    Note:
        零序网络特点：
        - 线路零序参数通常大于正序参数（X0 ≈ 2~3.5 × X1）
        - 变压器零序通路取决于接线方式
        - 发电机零序电抗通常很小（中性点经阻抗接地时）
        - 零序网络中无电源电势
    """
    # 收集所有零序线路和变压器
    zero_lines = []
    for zl in z_lines:
        line = Line(from_bus=zl.from_bus, to_bus=zl.to_bus,
                    R=zl.R0, X=zl.X0, B=zl.B0)
        zero_lines.append(line)

    # 只保留有零序通路的变压器
    zero_transformers = []
    for ztx in z_transformers:
        if ztx.connection in ["Ynd", "Yyn"]:
            tx = Transformer(from_bus=ztx.from_bus, to_bus=ztx.to_bus,
                           RT=ztx.RT0, XT=ztx.XT0)
            zero_transformers.append(tx)

    if bus_numbers is None:
        bus_numbers = [bus.number for bus in buses]

    ybus_result = build_ybus(zero_lines, zero_transformers, bus_numbers=bus_numbers)

    # 修改发电机节点导纳（使用零序电抗，如果有的话）
    Ybus0 = ybus_result.Ybus.copy()
    for gen in generators:
        if gen.bus in ybus_result.bus_indices:
            # 零序电抗通常很大（表示无零序通路），除非中性点接地
            # 这里假设发电机零序电抗为无穷大（不加入导纳）
            # 如果需要中性点接地，用户应单独指定
            pass

    updated_result = YBusResult(
        Ybus=Ybus0,
        n_bus=ybus_result.n_bus,
        bus_indices=ybus_result.bus_indices,
        G=Ybus0.real,
        B=Ybus0.imag
    )

    try:
        Zbus0 = np.linalg.inv(Ybus0)
    except np.linalg.LinAlgError:
        # 零序网络可能不连通，部分节点无零序通路
        # 使用伪逆
        Zbus0 = np.linalg.pinv(Ybus0)

    return updated_result, Zbus0


def build_all_sequence_networks(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    z_lines: Optional[List['ZeroSequenceLine']] = None,
    z_transformers: Optional[List['ZeroSequenceTransformer']] = None,
    bus_numbers: Optional[List[int]] = None
) -> SequenceNetworks:
    """构造完整的序网模型

    Args:
        buses: 节点列表
        lines: 输电线路列表（正序参数）
        transformers: 变压器列表
        generators: 发电机列表
        z_lines: 零序线路参数（可选，默认等于正序参数）
        z_transformers: 零序变压器参数（可选）
        bus_numbers: 节点编号列表（可选）

    Returns:
        SequenceNetworks: 包含正序、负序、零序网络的完整模型
    """
    # 构造各序网络
    pos_result, Zbus1 = build_positive_sequence_network(
        buses, lines, transformers, generators, bus_numbers)
    neg_result, Zbus2 = build_negative_sequence_network(
        buses, lines, transformers, generators, bus_numbers)

    if z_lines is None:
        # 默认零序线路参数等于正序参数（简化处理）
        z_lines = [ZeroSequenceLine(l.from_bus, l.to_bus, l.R, l.X, l.B)
                   for l in lines]

    if z_transformers is None:
        z_transformers = [ZeroSequenceTransformer(t.from_bus, t.to_bus, t.XT, t.RT)
                         for t in transformers]

    zero_result, Zbus0 = build_zero_sequence_network(
        buses, z_lines, z_transformers, generators, bus_numbers)

    return SequenceNetworks(
        positive=pos_result,
        negative=neg_result,
        zero=zero_result,
        Zbus1=Zbus1,
        Zbus2=Zbus2,
        Zbus0=Zbus0,
        bus_indices=pos_result.bus_indices
    )