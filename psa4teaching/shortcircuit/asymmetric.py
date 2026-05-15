"""
不对称短路计算 (Asymmetric Fault)
=================================

基于对称分量法的各种不对称短路电流计算。

数学模型
--------
利用对称分量法，将不对称短路问题分解为三个序网络的组合：

1. 单相接地短路（a相接地）：
    边界条件：Ib=0, Ic=0, Va=0
    序网连接：正序-负序-零序 串联
    计算公式：
        If = Ia1 = Ea / (Z1 + Z2 + Z0)
        其中 Z1, Z2, Z0 分别为短路点的正序、负序、零序等效阻抗

2. 两相短路（bc相短路）：
    边界条件：Ia=0, Ib=-Ic, Vb=Vc
    序网连接：正序-负序 并联
    计算公式：
        If = √3 × Ia1 = √3 × Ea / (Z1 + Z2)

3. 两相接地短路（bc相接地）：
    边界条件：Ia=0, Vb=0, Vc=0
    序网连接：正序与(负序∥零序) 并联
    计算公式：
        Ia1 = Ea / (Z1 + Z2∥Z0)
        If = 3 × Ia0 = 3 × Ia1 × Z2/(Z2+Z0)

参考教材：
    - 陈珩《电力系统稳态分析》第六章 6.4-6.5节
    - 李光琦《电力系统暂态分析》第二章 2.2-2.4节
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator
from psa4teaching.shortcircuit.sequence_network import (
    SequenceNetworks, build_all_sequence_networks,
    ZeroSequenceLine, ZeroSequenceTransformer
)


# 对称分量变换矩阵
_a = complex(-0.5, np.sqrt(3) / 2)  # 旋转因子 a = e^(j120°)

def phase_to_sequence(Ia, Ib, Ic):
    """三相相量 → 对称分量

    Args:
        Ia, Ib, Ic: 三相电流/电压相量

    Returns:
        (I0, I1, I2): 零序、正序、负序分量
    """
    I0 = (Ia + Ib + Ic) / 3
    I1 = (Ia + _a * Ib + _a**2 * Ic) / 3
    I2 = (Ia + _a**2 * Ib + _a * Ic) / 3
    return I0, I1, I2


def sequence_to_phase(I0, I1, I2):
    """对称分量 → 三相相量

    Args:
        I0, I1, I2: 零序、正序、负序分量

    Returns:
        (Ia, Ib, Ic): 三相电流/电压相量
    """
    Ia = I0 + I1 + I2
    Ib = I0 + _a**2 * I1 + _a * I2
    Ic = I0 + _a * I1 + _a**2 * I2
    return Ia, Ib, Ic


@dataclass
class AsymmetricFaultResult:
    """不对称短路计算结果

    Attributes:
        fault_type: 故障类型
        fault_bus: 故障点节点编号
        fault_current: 故障点电流（a相电流）
        fault_currents_3phase: 三相故障电流 (Ia, Ib, Ic)
        sequence_currents: 序电流 (I0, I1, I2)
        sequence_voltages: 故障点序电压 (V0, V1, V2)
        Z1: 正序等效阻抗
        Z2: 负序等效阻抗
        Z0: 零序等效阻抗
    """
    fault_type: str
    fault_bus: int
    fault_current: complex
    fault_currents_3phase: Tuple[complex, complex, complex]
    sequence_currents: Tuple[complex, complex, complex]
    sequence_voltages: Tuple[complex, complex, complex]
    Z1: complex
    Z2: complex
    Z0: complex


def calculate_single_line_to_ground(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    fault_bus: int,
    Ea: complex = 1.0 + 0j,
    z_lines: Optional[List] = None,
    z_transformers: Optional[List] = None,
    fault_impedance: Optional[complex] = None
) -> AsymmetricFaultResult:
    """计算单相接地短路电流（a相接地）

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        generators: 发电机列表
        fault_bus: 故障点节点编号
        Ea: 正序等值电势（标幺值），默认1.0
        z_lines: 零序线路参数（可选）
        z_transformers: 零序变压器参数（可选）
        fault_impedance: 故障点过渡阻抗（可选）

    Returns:
        AsymmetricFaultResult: 单相接地短路计算结果

    Note:
        序网连接：正序-负序-零序 串联

        计算公式：
            Ia1 = Ea / (Z1 + Z2 + Z0)
            Ia = Ia0 = Ia1 = Ia2  （串联，各序电流相等）
            If = Ia = 3 × Ia0

        故障点序电压：
            Va1 = Ea - Ia1 × Z1
            Va2 = -Ia2 × Z2
            Va0 = -Ia0 × Z0
    """
    # 构造序网
    seq_net = build_all_sequence_networks(
        buses, lines, transformers, generators,
        z_lines, z_transformers
    )

    bus_indices = seq_net.bus_indices
    fault_idx = bus_indices[fault_bus]

    # 从Zbus获取故障点各序阻抗
    Z1 = seq_net.Zbus1[fault_idx, fault_idx]
    Z2 = seq_net.Zbus2[fault_idx, fault_idx]
    Z0 = seq_net.Zbus0[fault_idx, fault_idx]

    # 考虑故障阻抗
    if fault_impedance is not None:
        Z1_total = Z1 + fault_impedance
    else:
        Z1_total = Z1

    # 单相接地：三序串联
    # Ia1 = Ea / (Z1 + Z2 + Z0)
    Ia1 = Ea / (Z1_total + Z2 + Z0)
    Ia2 = Ia1
    Ia0 = Ia1

    # 故障电流
    Ia = Ia0 + Ia1 + Ia2  # = 3 × Ia0

    # 三相电流
    Ia_3ph, Ib_3ph, Ic_3ph = sequence_to_phase(Ia0, Ia1, Ia2)

    # 故障点序电压
    Va1 = Ea - Ia1 * Z1_total
    Va2 = -Ia2 * Z2
    Va0 = -Ia0 * Z0

    return AsymmetricFaultResult(
        fault_type="单相接地(a相)",
        fault_bus=fault_bus,
        fault_current=Ia,
        fault_currents_3phase=(Ia_3ph, Ib_3ph, Ic_3ph),
        sequence_currents=(Ia0, Ia1, Ia2),
        sequence_voltages=(Va0, Va1, Va2),
        Z1=Z1, Z2=Z2, Z0=Z0
    )


def calculate_line_to_line(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    fault_bus: int,
    Ea: complex = 1.0 + 0j,
    z_lines: Optional[List] = None,
    z_transformers: Optional[List] = None,
    fault_impedance: Optional[complex] = None
) -> AsymmetricFaultResult:
    """计算两相短路电流（b、c相短路）

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        generators: 发电机列表
        fault_bus: 故障点节点编号
        Ea: 正序等值电势
        z_lines: 零序线路参数（可选）
        z_transformers: 零序变压器参数（可选）
        fault_impedance: 故障点过渡阻抗（可选）

    Returns:
        AsymmetricFaultResult: 两相短路计算结果

    Note:
        序网连接：正序-负序 并联（零序不通）

        计算公式：
            Ia1 = Ea / (Z1 + Z2)
            Ia2 = -Ia1
            Ia0 = 0

        故障电流：
            Ib = (a² - a) × Ia1 = -j√3 × Ia1
            Ic = -(Ib) = j√3 × Ia1
    """
    seq_net = build_all_sequence_networks(
        buses, lines, transformers, generators,
        z_lines, z_transformers
    )

    bus_indices = seq_net.bus_indices
    fault_idx = bus_indices[fault_bus]

    Z1 = seq_net.Zbus1[fault_idx, fault_idx]
    Z2 = seq_net.Zbus2[fault_idx, fault_idx]
    Z0 = seq_net.Zbus0[fault_idx, fault_idx]

    if fault_impedance is not None:
        Z1_total = Z1 + fault_impedance
    else:
        Z1_total = Z1

    # 两相短路：正序-负序并联
    Ia1 = Ea / (Z1_total + Z2)
    Ia2 = -Ia1
    Ia0 = complex(0, 0)

    # 三相电流
    Ia_3ph, Ib_3ph, Ic_3ph = sequence_to_phase(Ia0, Ia1, Ia2)

    # 故障电流（a相电流）
    Ia = Ia0 + Ia1 + Ia2  # = 0

    # 故障点序电压
    Va1 = Ea - Ia1 * Z1_total
    Va2 = -Ia2 * Z2
    Va0 = complex(0, 0)

    return AsymmetricFaultResult(
        fault_type="两相短路(bc相)",
        fault_bus=fault_bus,
        fault_current=Ia,
        fault_currents_3phase=(Ia_3ph, Ib_3ph, Ic_3ph),
        sequence_currents=(Ia0, Ia1, Ia2),
        sequence_voltages=(Va0, Va1, Va2),
        Z1=Z1, Z2=Z2, Z0=Z0
    )


def calculate_double_line_to_ground(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    fault_bus: int,
    Ea: complex = 1.0 + 0j,
    z_lines: Optional[List] = None,
    z_transformers: Optional[List] = None,
    fault_impedance: Optional[complex] = None
) -> AsymmetricFaultResult:
    """计算两相接地短路电流（b、c相接地）

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        generators: 发电机列表
        fault_bus: 故障点节点编号
        Ea: 正序等值电势
        z_lines: 零序线路参数（可选）
        z_transformers: 零序变压器参数（可选）
        fault_impedance: 故障点过渡阻抗（可选）

    Returns:
        AsymmetricFaultResult: 两相接地短路计算结果

    Note:
        序网连接：正序与(负序∥零序) 并联

        计算公式：
            Z2∥Z0 = Z2 × Z0 / (Z2 + Z0)
            Ia1 = Ea / (Z1 + Z2∥Z0)
            Ia2 = -Ia1 × Z0/(Z2+Z0)
            Ia0 = -Ia1 × Z2/(Z2+Z0)

        故障电流：
            If = 3 × Ia0
    """
    seq_net = build_all_sequence_networks(
        buses, lines, transformers, generators,
        z_lines, z_transformers
    )

    bus_indices = seq_net.bus_indices
    fault_idx = bus_indices[fault_bus]

    Z1 = seq_net.Zbus1[fault_idx, fault_idx]
    Z2 = seq_net.Zbus2[fault_idx, fault_idx]
    Z0 = seq_net.Zbus0[fault_idx, fault_idx]

    if fault_impedance is not None:
        Z1_total = Z1 + fault_impedance
    else:
        Z1_total = Z1

    # 两相接地：正序与(负序∥零序)并联
    Z_parallel = Z2 * Z0 / (Z2 + Z0)
    Ia1 = Ea / (Z1_total + Z_parallel)
    Ia2 = -Ia1 * Z0 / (Z2 + Z0)
    Ia0 = -Ia1 * Z2 / (Z2 + Z0)

    # 三相电流
    Ia_3ph, Ib_3ph, Ic_3ph = sequence_to_phase(Ia0, Ia1, Ia2)

    # 故障电流
    Ia = Ia0 + Ia1 + Ia2

    # 故障点序电压
    Va1 = Ea - Ia1 * Z1_total
    Va2 = -Ia2 * Z2
    Va0 = -Ia0 * Z0

    return AsymmetricFaultResult(
        fault_type="两相接地(bc相)",
        fault_bus=fault_bus,
        fault_current=Ia,
        fault_currents_3phase=(Ia_3ph, Ib_3ph, Ic_3ph),
        sequence_currents=(Ia0, Ia1, Ia2),
        sequence_voltages=(Va0, Va1, Va2),
        Z1=Z1, Z2=Z2, Z0=Z0
    )