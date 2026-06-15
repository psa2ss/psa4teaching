"""
电压稳定性分析 (Voltage Stability Analysis)
=========================================

提供 PV 曲线和 QV 曲线的计算功能，用于电压稳定性评估。

数学模型
--------

### PV 曲线 (P-V Curve)
通过逐步增加负荷功率，追踪关键节点电压随负荷变化的曲线。
曲线的"鼻点"(nose point) 对应电压崩溃临界点。

    方法: 重复牛顿-拉夫逊潮流计算
    负荷模型: P_load(lambda) = lambda * P_base
              Q_load(lambda) = lambda * Q_base

### QV 曲线 (Q-V Curve)
固定负荷有功，扫描目标节点电压，计算所需的无功注入。
曲线的最低点给出无功裕度。

参考:
    - Kundur P. Power System Stability and Control, Ch.14
    - Taylor C.W. "Power System Voltage Stability", McGraw-Hill, 1994
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator
from psa4teaching.models.load import Load
from psa4teaching.network.ybus import build_ybus, YBusResult
from psa4teaching.powerflow.newton_raphson import run_newton_raphson


@dataclass
class PVCurveResult:
    """PV 曲线计算结果

    Attributes:
        lambda_values: 负荷增长因子序列
        V_curve: 各节点电压 (n_points × n_bus)
        P_total: 系统总负荷 (n_points,)
        nose_point_index: 鼻点（临界点）索引
        critical_lambda: 临界负荷因子
        converged: 曲线计算是否成功完成
    """
    lambda_values: NDArray[np.float64]
    V_curve: NDArray[np.float64]
    P_total: NDArray[np.float64]
    nose_point_index: int
    critical_lambda: float
    converged: bool


@dataclass
class QVCurveResult:
    """QV 曲线计算结果

    Attributes:
        V_values: 电压扫描值 (n_points,)
        Q_required: 所需无功注入 (n_points,)
        Q_min: 最小无功需求
        V_at_Qmin: Q 最小时的电压
        reactive_margin: 当前运行点到 Q_min 的无功裕度
        converged: 曲线计算是否成功
    """
    V_values: NDArray[np.float64]
    Q_required: NDArray[np.float64]
    Q_min: float
    V_at_Qmin: float
    reactive_margin: float
    converged: bool


def compute_pv_curve(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    loads: List[Load],
    target_bus: int = None,
    lambda_max: float = 3.0,
    n_points: int = 50,
    P_direction: Optional[NDArray[np.float64]] = None,
    Q_direction: Optional[NDArray[np.float64]] = None,
    verbose: bool = False,
) -> PVCurveResult:
    """计算 PV 曲线（重复潮流法）

    通过逐步增加系统负荷，追踪节点电压的变化，
    定位电压崩溃的鼻点。

    Args:
        buses: 节点列表（会被修改，建议传入副本）
        lines: 线路列表
        transformers: 变压器列表
        generators: 发电机列表
        loads: 负荷列表
        target_bus: 关键观测节点（如为 None，使用最大负荷节点）
        lambda_max: 最大负荷增长因子
        n_points: 扫描点数
        P_direction: 各节点有功增长方向（如为 None，按原始负荷比例）
        Q_direction: 各节点无功增长方向
        verbose: 打印进度

    Returns:
        PVCurveResult: PV 曲线计算结果

    Example:
        >>> result = compute_pv_curve(buses, lines, transformers,
        ...                           generators, loads, target_bus=3)
        >>> print(f"Critical lambda = {result.critical_lambda:.2f}")
    """
    n_bus = len(buses)

    # 建立节点编号到索引的映射
    bus_indices = {b.number: i for i, b in enumerate(buses)}

    # 保存原始负荷
    original_P = np.array([b.P_specified for b in buses])
    original_Q = np.array([b.Q_specified for b in buses])

    # 确定负荷增长方向
    if P_direction is None:
        P_direction = np.array([
            abs(b.P_specified) if b.P_specified < 0 else 0.0
            for b in buses
        ])
        if np.sum(P_direction) < 1e-9:
            P_direction = np.ones(n_bus) / n_bus  # fallback

    if Q_direction is None:
        Q_direction = np.array([
            abs(b.Q_specified) if b.Q_specified < 0 else 0.0
            for b in buses
        ])
        if np.sum(Q_direction) < 1e-9:
            Q_direction = np.ones(n_bus) / n_bus * 0.3

    # 归一化
    P_direction = P_direction / np.sum(np.abs(P_direction))
    Q_direction = Q_direction / np.sum(np.abs(Q_direction))

    lambda_vals = np.linspace(0, lambda_max, n_points)
    V_curve = np.zeros((n_points, n_bus))
    P_total = np.zeros(n_points)

    nose_idx = n_points - 1
    crit_lambda = lambda_max

    if target_bus is None:
        # 选择负荷最大的 PQ 节点作为观测节点
        pq_buses = [(i, abs(b.P_specified)) for i, b in enumerate(buses)
                    if b.bus_type == BusType.PQ and b.P_specified < 0]
        if pq_buses:
            target_bus = max(pq_buses, key=lambda x: x[1])[0]
        else:
            target_bus = 0

    for k, lam in enumerate(lambda_vals):
        # 修改负荷
        for i, b in enumerate(buses):
            if b.bus_type == BusType.PQ:
                b.P_specified = original_P[i] + lam * P_direction[i] * (-1)
                b.Q_specified = original_Q[i] + lam * Q_direction[i] * (-1)
            # PV 节点增加发电以平衡负荷增长 (简化: 按比例分配)
            if b.bus_type == BusType.PV and b.P_specified > 0:
                b.P_specified = max(original_P[i],
                    original_P[i] + lam * np.sum(P_direction) / n_bus)

        # 构造导纳矩阵
        ybus_result = build_ybus(lines, transformers)

        # 执行潮流计算
        result = run_newton_raphson(
            buses, ybus_result, max_iterations=30, tolerance=1e-6, verbose=False
        )

        if result.converged:
            V_curve[k, :] = result.V
            P_total[k] = lam * np.sum(np.abs(P_direction))
        else:
            # 潮流不收敛 → 到达鼻点
            nose_idx = max(0, k - 1)
            crit_lambda = lambda_vals[nose_idx] if nose_idx > 0 else 0.0
            if verbose and k > 0:
                print(f"  Nose point at lambda={crit_lambda:.3f} (k={k})")
            break
    else:
        # 全部收敛
        nose_idx = n_points - 1
        crit_lambda = lambda_max

    # 恢复原始负荷
    for i, b in enumerate(buses):
        b.P_specified = original_P[i]
        b.Q_specified = original_Q[i]

    return PVCurveResult(
        lambda_values=lambda_vals[:nose_idx + 1],
        V_curve=V_curve[:nose_idx + 1, :],
        P_total=P_total[:nose_idx + 1],
        nose_point_index=nose_idx,
        critical_lambda=crit_lambda,
        converged=nose_idx > 0,
    )


def compute_qv_curve(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    generators: List[Generator],
    loads: List[Load],
    target_bus: int,
    V_range: Tuple[float, float] = (0.4, 1.2),
    n_points: int = 40,
    verbose: bool = False,
) -> QVCurveResult:
    """计算 QV 曲线

    将目标节点转换为 PV 节点，扫描电压值，
    通过潮流计算获得维持该电压所需的无功注入。

    Args:
        buses: 节点列表（会被修改，建议传入副本）
        lines: 线路列表
        transformers: 变压器列表
        generators: 发电机列表
        loads: 负荷列表
        target_bus: 目标节点编号
        V_range: 电压扫描范围 (V_min, V_max) pu
        n_points: 扫描点数
        verbose: 打印进度

    Returns:
        QVCurveResult: QV 曲线计算结果

    Note:
        如果目标节点无法维持指定电压（潮流不收敛），
        对应的 Q 值设为 NaN，曲线在该点中断。
    """
    # 保存节点原始类型
    orig_types = [b.bus_type for b in buses]
    orig_V = [b.V_specified for b in buses]
    orig_P = [b.P_specified for b in buses]
    orig_Q = [b.Q_specified for b in buses]

    # 找到目标节点索引
    bus_indices = {b.number: i for i, b in enumerate(buses)}
    tgt_idx = bus_indices[target_bus]

    V_vals = np.linspace(V_range[0], V_range[1], n_points)
    Q_required = np.full(n_points, np.nan)
    converged = False

    for k, V_set in enumerate(V_vals):
        # 将目标节点改为 PV (控制电压 = V_set)
        buses[tgt_idx].bus_type = BusType.PV
        buses[tgt_idx].V_specified = V_set
        buses[tgt_idx].P_specified = orig_P[tgt_idx]  # 保持有功不变

        # 构造导纳矩阵
        ybus_result = build_ybus(lines, transformers)

        # 执行潮流计算
        result = run_newton_raphson(
            buses, ybus_result, max_iterations=30, tolerance=1e-6, verbose=False
        )

        if result.converged:
            # 潮流收敛 → 记录目标节点的无功注入
            tgt_result_idx = list(ybus_result.bus_indices.keys()).index(target_bus)
            Q_required[k] = result.Q[tgt_result_idx]
            converged = True

    # 恢复原始节点类型
    for i, b in enumerate(buses):
        b.bus_type = orig_types[i]
        b.V_specified = orig_V[i]
        b.P_specified = orig_P[i]
        b.Q_specified = orig_Q[i]

    # 分析结果
    valid_mask = ~np.isnan(Q_required)
    if np.any(valid_mask):
        Q_min = np.min(Q_required[valid_mask])
        idx_min = np.argmin(Q_required[valid_mask])
        V_at_Qmin = V_vals[valid_mask][idx_min]
        # 无功裕度: 当前运行 Q 与 Q_min 的差值
        current_Q = buses[tgt_idx].Q
        reactive_margin = current_Q - Q_min if current_Q is not None else 0.0
    else:
        Q_min = 0.0
        V_at_Qmin = 1.0
        reactive_margin = 0.0

    return QVCurveResult(
        V_values=V_vals,
        Q_required=Q_required,
        Q_min=Q_min,
        V_at_Qmin=V_at_Qmin,
        reactive_margin=reactive_margin,
        converged=converged,
    )


__all__ = [
    "PVCurveResult",
    "QVCurveResult",
    "compute_pv_curve",
    "compute_qv_curve",
]
