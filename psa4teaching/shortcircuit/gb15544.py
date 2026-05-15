"""
GB/T 15544 三相交流系统短路电流计算
=====================================

基于GB/T 15544（等同于IEC 60909）的三相短路电流计算方法。

标准要点
--------
GB/T 15544 规定了三相交流系统短路电流的计算方法，主要特点：

1. 等效电压源法：
    - 引入等效电压源 c × Un / √3
    - c 为电压系数，取决于系统电压等级

2. 电压系数 c：
    - 380V ~ 1kV：c_max = 1.10, c_min = 0.95
    - 1kV ~ 35kV：c_max = 1.10, c_min = 1.00
    - 35kV ~ 220kV：c_max = 1.10, c_min = 1.00
    - 220kV 以上：c_max = 1.10, c_min = 1.00

3. 短路电流类型：
    - 初始对称短路电流 I"k（次暂态短路电流）
    - 峰值短路电流 ip
    - 开断短路电流 Ib
    - 稳态短路电流 Ik

4. 计算方法：
    - 最大短路电流：c = c_max
    - 最小短路电流：c = c_min

5. 短路阻抗修正：
    - 发电机阻抗需乘以修正系数KG
    - 变压器阻抗需考虑分接头位置
    - 线路阻抗温度修正

参考标准：GB/T 15544-1995 / IEC 60909

参考教材：李光琦《电力系统暂态分析》第二章
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator


# 电压系数表（GB/T 15544 表1）
VOLTAGE_FACTORS = {
    'LV':    {'c_max': 1.10, 'c_min': 0.95, 'range': '0~1kV'},
    'MV':    {'c_max': 1.10, 'c_min': 1.00, 'range': '1~35kV'},
    'HV_35': {'c_max': 1.10, 'c_min': 1.00, 'range': '35~220kV'},
    'HV_220':{'c_max': 1.10, 'c_min': 1.00, 'range': '220kV以上'},
}


@dataclass
class GB15544Result:
    """GB/T 15544 短路计算结果

    Attributes:
        fault_bus: 短路点节点编号
        voltage_level: 电压等级类型
        c_used: 使用的电压系数
        Ik_initial: 初始对称短路电流 I"k（kA）
        ip_peak: 峰值短路电流 ip（kA）
        Ik_steady: 稳态短路电流 Ik（kA）
        Ik_min: 最小短路电流（kA）
        X_R_ratio: 短路点等效X/R比
        Z_eq: 等效短路阻抗（标幺值）
    """
    fault_bus: int
    voltage_level: str
    c_used: float
    Ik_initial: float        # 初始对称短路电流（kA）
    ip_peak: float           # 峰值短路电流（kA）
    Ik_steady: float         # 稳态短路电流（kA）
    Ik_min: float            # 最小短路电流（kA）
    X_R_ratio: float         # X/R比
    Z_eq: complex            # 等效阻抗


def get_voltage_factor(
    V_nominal: float,
    max_current: bool = True
) -> float:
    """获取电压系数c

    Args:
        V_nominal: 系统额定电压（kV）
        max_current: True为计算最大短路电流（c_max），False为最小（c_min）

    Returns:
        电压系数c

    Note:
        电压等级划分：
        - 低压（LV）：Un ≤ 1kV，c_max=1.10, c_min=0.95
        - 中压（MV）：1kV < Un ≤ 35kV，c_max=1.10, c_min=1.00
        - 高压（HV）：Un > 35kV，c_max=1.10, c_min=1.00
    """
    if V_nominal <= 1.0:
        level = 'LV'
    elif V_nominal <= 35.0:
        level = 'MV'
    else:
        level = 'HV_220'

    if max_current:
        return VOLTAGE_FACTORS[level]['c_max']
    else:
        return VOLTAGE_FACTORS[level]['c_min']


def get_correction_factor(
    generator: Generator,
    is_max: bool = True
) -> Dict[str, float]:
    """计算发电机短路阻抗修正系数

    Args:
        generator: 发电机模型
        is_max: True为最大短路电流修正

    Returns:
        包含修正系数KG和修正后阻抗的字典

    Note:
        GB/T 15544 规定发电机阻抗修正系数：
        KG = UnG / UrG × 1 / tgφr

        修正后的发电机阻抗：
        X"G_corrected = KG × X"G

        对于最大短路电流计算：
        KG = VnG / UrG × cmax

        简化计算中，可取 KG ≈ 1.1（发电机额定电压等于系统额定电压时）
    """
    # 简化修正系数
    if is_max:
        KG = 1.1  # 典型值
    else:
        KG = 0.95

    Xd_pp_corrected = KG * generator.Xd_doubleprime

    return {
        'KG': KG,
        'Xd_doubleprime_corrected': Xd_pp_corrected,
        'X2_corrected': KG * (generator.Xq_doubleprime or generator.Xd_doubleprime)
    }


def calculate_gb15544(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    fault_bus: int,
    V_nominal: float = 10.5,
    S_base: float = 100.0,
    max_current: bool = True,
    verbose: bool = False
) -> GB15544Result:
    """基于GB/T 15544计算三相短路电流

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        generators: 发电机列表
        fault_bus: 短路点节点编号
        V_nominal: 系统额定电压（kV）
        S_base: 基准容量（MVA）
        max_current: True为计算最大短路电流
        verbose: 是否打印详细信息

    Returns:
        GB15544Result: GB/T 15544短路计算结果

    Note:
        计算步骤：
        1. 获取电压系数c
        2. 对发电机阻抗进行修正
        3. 构造修正后的Zbus
        4. 计算短路点等效阻抗
        5. 计算初始对称短路电流 I"k
        6. 计算峰值短路电流 ip = κ × √2 × I"k
        7. 计算稳态短路电流 Ik

        其中 κ 为冲击系数：
        κ = 1.02 + 0.98 × e^(-3/(X/R))
        或简化取 κ = 1.8（X/R ≥ 3时）
    """
    from psa4teaching.network.ybus import build_ybus

    # 获取电压系数
    c = get_voltage_factor(V_nominal, max_current)

    # 确定电压等级类型
    if V_nominal <= 1.0:
        voltage_level = 'LV'
    elif V_nominal <= 35.0:
        voltage_level = 'MV'
    else:
        voltage_level = 'HV'

    # 构造修正后的网络导纳矩阵
    bus_numbers = [bus.number for bus in buses]
    ybus_result = build_ybus(lines, transformers, bus_numbers=bus_numbers)
    Ybus = ybus_result.Ybus.copy()
    bus_indices = ybus_result.bus_indices

    # 对发电机阻抗进行修正
    for gen in generators:
        if gen.bus in bus_indices:
            idx = bus_indices[gen.bus]
            correction = get_correction_factor(gen, max_current)
            KG = correction['KG']
            Xd_pp_corr = correction['Xd_doubleprime_corrected']
            # 修正发电机节点导纳
            Y_gen = KG / (1j * Xd_pp_corr)
            Ybus[idx, idx] += Y_gen

    # 求Zbus
    try:
        Zbus = np.linalg.inv(Ybus)
    except np.linalg.LinAlgError:
        raise ValueError("修正后Ybus矩阵奇异，无法计算短路电流")

    fault_idx = bus_indices[fault_bus]
    Z_eq = Zbus[fault_idx, fault_idx]

    # 基准电流
    I_base = S_base / (np.sqrt(3) * V_nominal)

    # 初始对称短路电流 I"k = c × Vn / (√3 × Z_eq)
    # 在标幺值中：I"k = c / Z_eq
    Ik_pu = c / Z_eq
    Ik_initial = abs(Ik_pu) * I_base

    # X/R比
    X_R_ratio = Z_eq.imag / Z_eq.real if abs(Z_eq.real) > 1e-10 else 100.0

    # 冲击系数 κ
    if X_R_ratio >= 3:
        kappa = 1.8
    else:
        kappa = 1.02 + 0.98 * np.exp(-3.0 / X_R_ratio)

    # 峰值短路电流 ip = κ × √2 × I"k
    ip_peak = kappa * np.sqrt(2) * Ik_initial

    # 稳态短路电流（简化取等于初始值，实际需考虑发电机特性）
    Ik_steady = Ik_initial

    # 最小短路电流
    c_min = get_voltage_factor(V_nominal, max_current=False)
    Ik_min_pu = c_min / Z_eq
    Ik_min = abs(Ik_min_pu) * I_base

    if verbose:
        print(f"=== GB/T 15544 短路计算结果 ===")
        print(f"短路点: 节点 {fault_bus}")
        print(f"电压等级: {V_nominal}kV ({voltage_level})")
        print(f"电压系数 c = {c:.2f}")
        print(f"等效阻抗 Z = {abs(Z_eq):.6f} p.u. (∠{np.degrees(np.angle(Z_eq)):.2f}°)")
        print(f"X/R = {X_R_ratio:.2f}")
        print(f"冲击系数 κ = {kappa:.3f}")
        print(f"初始对称短路电流 I\"k = {Ik_initial:.2f} kA")
        print(f"峰值短路电流 ip = {ip_peak:.2f} kA")
        print(f"稳态短路电流 Ik = {Ik_steady:.2f} kA")
        print(f"最小短路电流 Ik_min = {Ik_min:.2f} kA")

    return GB15544Result(
        fault_bus=fault_bus,
        voltage_level=voltage_level,
        c_used=c,
        Ik_initial=Ik_initial,
        ip_peak=ip_peak,
        Ik_steady=Ik_steady,
        Ik_min=Ik_min,
        X_R_ratio=X_R_ratio,
        Z_eq=Z_eq
    )