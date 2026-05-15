"""
对称短路计算 (Symmetric Fault / Three-Phase Fault)
=================================================

基于节点阻抗矩阵的三相短路电流计算。

数学模型
--------
三相短路时，短路点的边界条件为：
    Va_f = 0

短路电流计算：
    If = Vf(0) / Zff

其中：
    Vf(0) —— 短路前故障点电压（通常取1.0）
    Zff —— 故障点的自阻抗（Zbus[f,f]）

各节点电压：
    Vi = Vi(0) - Zif × If

各支路电流：
    Iij = (Vi - Vj) / Zij

转移阻抗计算：
    转移阻抗 Zkf 表示发电机k对短路点f的转移阻抗
    转移阻抗可通过Zbus矩阵直接获得：Zkf = Zbus[k,f]

参考教材：
    - 陈珩《电力系统稳态分析》第六章 6.2-6.3节
    - 李光琦《电力系统暂态分析》第二章 2.1节
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
from psa4teaching.network.zbus import build_zbus, ZBusResult


@dataclass
class SymmetricFaultResult:
    """三相短路计算结果

    Attributes:
        fault_bus: 短路点节点编号
        fault_current: 短路点短路电流（标幺值）
        fault_current_ka: 短路点短路电流（kA）
        Zff: 短路点自阻抗
        V_pu: 各节点短路后电压（标幺值）
        V_before: 各节点短路前电压
        branch_currents: 各支路短路电流
        transfer_impedances: 各发电机到短路点的转移阻抗
    """
    fault_bus: int
    fault_current: complex
    fault_current_ka: float
    Zff: complex
    V_pu: Dict[int, complex]
    V_before: Dict[int, complex]
    branch_currents: Dict[Tuple[int, int], complex]
    transfer_impedances: Dict[int, complex]


def calculate_three_phase_fault(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    fault_bus: int,
    fault_impedance: Optional[complex] = None,
    S_base: float = 100.0,
    V_base: float = None,
    verbose: bool = False
) -> SymmetricFaultResult:
    """计算三相短路电流

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        generators: 发电机列表
        fault_bus: 短路点节点编号
        fault_impedance: 故障点过渡阻抗（标幺值），默认为0（金属性短路）
        S_base: 系统基准容量（MVA）
        V_base: 短路点基准电压（kV），若为None则自动取节点电压基准
        verbose: 是否打印详细信息

    Returns:
        SymmetricFaultResult: 三相短路计算结果

    Example:
        >>> result = calculate_three_phase_fault(buses, lines, transformers,
        ...                                       generators, fault_bus=3)
        >>> print(f"短路电流: {result.fault_current_ka:.2f} kA")
    """
    # 构造Zbus
    bus_numbers = [bus.number for bus in buses]
    ybus_result = build_ybus(lines, transformers, bus_numbers=bus_numbers)
    zbus_result = build_zbus(lines, transformers, bus_numbers=bus_numbers)

    Zbus = zbus_result.Zbus
    bus_indices = zbus_result.bus_indices
    n_bus = zbus_result.n_bus

    fault_idx = bus_indices[fault_bus]

    # 短路前各节点电压（通常取1.0）
    V_before = {}
    for bus in buses:
        idx = bus_indices[bus.number]
        if bus.bus_type == BusType.SLACK:
            V_before[bus.number] = complex(bus.V_specified, 0)
        else:
            V_before[bus.number] = complex(1.0, 0)

    # 短路点短路前电压
    Vf0 = V_before[fault_bus]

    # 短路点自阻抗
    Zff = Zbus[fault_idx, fault_idx]

    # 考虑故障阻抗
    if fault_impedance is not None:
        Z_total = Zff + fault_impedance
    else:
        Z_total = Zff

    # 短路电流 If = Vf(0) / Zff
    fault_current = Vf0 / Z_total

    # 计算各节点短路后电压
    V_after = {}
    for bus in buses:
        idx = bus_indices[bus.number]
        V_after[bus.number] = V_before[bus.number] - Zbus[idx, fault_idx] * fault_current

    # 计算支路电流
    branch_currents = {}
    for line in lines:
        i, j = line.from_bus, line.to_bus
        idx_i, idx_j = bus_indices[i], bus_indices[j]
        Vi = V_after[i]
        Vj = V_after[j]
        Z_series = line.Z
        if abs(Z_series) > 1e-10:
            Iij = (Vi - Vj) / Z_series
            branch_currents[(i, j)] = Iij
            branch_currents[(j, i)] = -Iij

    # 计算转移阻抗
    transfer_impedances = {}
    for gen in generators:
        idx_g = bus_indices[gen.bus]
        Zkf = Zbus[idx_g, fault_idx]
        transfer_impedances[gen.bus] = Zkf

    # 计算有名值（kA）
    if V_base is None:
        V_base = 10.5  # 默认10.5kV
    I_base = S_base / (np.sqrt(3) * V_base)
    fault_current_ka = abs(fault_current) * I_base

    if verbose:
        print(f"短路点: 节点{fault_bus}")
        print(f"短路前电压: {Vf0:.4f} p.u.")
        print(f"短路点自阻抗: {Zff:.4f} p.u.")
        print(f"短路电流: {fault_current:.4f} p.u. = {fault_current_ka:.2f} kA")

    return SymmetricFaultResult(
        fault_bus=fault_bus,
        fault_current=fault_current,
        fault_current_ka=fault_current_ka,
        Zff=Zff,
        V_pu=V_after,
        V_before=V_before,
        branch_currents=branch_currents,
        transfer_impedances=transfer_impedances
    )


def calculate_transfer_impedances(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    fault_bus: int,
    generator_buses: Optional[List[int]] = None,
    bus_numbers: Optional[List[int]] = None
) -> Dict[int, complex]:
    """计算各发电机到短路点的转移阻抗

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        fault_bus: 短路点节点编号
        generator_buses: 发电机节点编号列表（可选）
        bus_numbers: 节点编号列表（可选）

    Returns:
        字典，键为发电机节点编号，值为转移阻抗Zkf

    Note:
        转移阻抗的物理意义：
        - Zkf = Zbus[k,f]
        - 发电机k对短路点的电流贡献：Ig = Eg / Zkf
        - 转移阻抗越大，发电机贡献越小

        利用转移阻抗可分析各发电机对短路电流的贡献。
    """
    zbus_result = build_zbus(lines, transformers, bus_numbers=bus_numbers)
    Zbus = zbus_result.Zbus
    bus_indices = zbus_result.bus_indices

    if generator_buses is None:
        generator_buses = [bus.number for bus in buses
                          if bus.bus_type in [BusType.PV, BusType.SLACK]]

    fault_idx = bus_indices[fault_bus]
    result = {}

    for gen_bus in generator_buses:
        idx_g = bus_indices[gen_bus]
        result[gen_bus] = Zbus[idx_g, fault_idx]

    return result