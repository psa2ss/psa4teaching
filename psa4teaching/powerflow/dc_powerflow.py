"""
直流潮流计算 (DC Power Flow)
============================

使用直流潮流法进行快速潮流计算。

数学模型
--------
直流潮流基于以下简化假设：
1. 忽略电阻，只考虑电抗（R << X）
2. 忽略并联支路
3. 节点电压幅值近似为1.0
4. 电压相角差很小，sin(θij) ≈ θij

简化后的潮流方程：
    P = B × θ
    θ = B⁻¹ × P

其中 Bij = -1/Xij。

参考教材：陈珩《电力系统稳态分析》第四章 4.5节
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer


@dataclass
class DCPowerFlowResult:
    """直流潮流计算结果

    Attributes:
        theta: 节点电压相角向量（弧度）
        P: 节点有功注入向量
        branch_flows: 支路有功功率流
        losses: 有功损耗
    """
    theta: NDArray[np.float64]
    P: NDArray[np.float64]
    branch_flows: Dict[tuple, float]
    losses: float
    bus_indices: Dict[int, int]


def run_dc_powerflow(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    bus_numbers: Optional[List[int]] = None
) -> DCPowerFlowResult:
    """执行直流潮流计算

    Args:
        buses: 节点列表
        lines: 输电线路列表
        transformers: 变压器列表
        bus_numbers: 节点编号列表（可选）

    Returns:
        DCPowerFlowResult: 直流潮流计算结果

    Example:
        >>> result = run_dc_powerflow(buses, lines, transformers)
        >>> print(f"节点相角: {result.theta}")
    """
    # 收集节点编号
    bus_set = set()
    for bus in buses:
        bus_set.add(bus.number)
    for line in lines:
        bus_set.add(line.from_bus)
        bus_set.add(line.to_bus)
    for tx in transformers:
        bus_set.add(tx.from_bus)
        bus_set.add(tx.to_bus)

    if bus_numbers:
        all_buses = bus_numbers
    else:
        all_buses = sorted(list(bus_set))

    n_bus = len(all_buses)
    bus_indices = {bus: idx for idx, bus in enumerate(all_buses)}

    # 构造B矩阵（直流潮流）
    B = np.zeros((n_bus, n_bus))

    for line in lines:
        i, j = line.from_bus, line.to_bus
        idx_i, idx_j = bus_indices[i], bus_indices[j]

        Bij = -1.0 / line.X
        B[idx_i, idx_j] += Bij
        B[idx_j, idx_i] += Bij
        B[idx_i, idx_i] -= Bij
        B[idx_j, idx_j] -= Bij

    for tx in transformers:
        i, j = tx.from_bus, tx.to_bus
        idx_i, idx_j = bus_indices[i], bus_indices[j]

        Bij = -1.0 / tx.XT
        B[idx_i, idx_j] += Bij
        B[idx_j, idx_i] += Bij
        B[idx_i, idx_i] -= Bij
        B[idx_j, idx_j] -= Bij

    # 找到平衡节点
    slack_bus = None
    slack_idx = -1
    for bus in buses:
        if bus.bus_type == BusType.SLACK:
            slack_bus = bus
            slack_idx = bus_indices[bus.number]
            break

    # 构造P向量
    P = np.zeros(n_bus)
    for bus in buses:
        idx = bus_indices[bus.number]
        if bus.bus_type != BusType.SLACK:
            P[idx] = bus.P_specified

    # 消去平衡节点
    if slack_idx >= 0:
        non_slack = [i for i in range(n_bus) if i != slack_idx]
        B_reduced = B[np.ix_(non_slack, non_slack)]
        P_reduced = P[non_slack]

        try:
            theta_reduced = np.linalg.solve(B_reduced, P_reduced)
        except np.linalg.LinAlgError:
            raise ValueError("B矩阵奇异，无法求解直流潮流")

        theta = np.zeros(n_bus)
        for i, idx in enumerate(non_slack):
            theta[idx] = theta_reduced[i]
    else:
        raise ValueError("未找到平衡节点")

    # 计算支路功率流
    branch_flows = {}
    for line in lines:
        i, j = line.from_bus, line.to_bus
        idx_i, idx_j = bus_indices[i], bus_indices[j]
        # Pij = (θi - θj) / Xij
        Pij = (theta[idx_i] - theta[idx_j]) / line.X
        branch_flows[(i, j)] = Pij

    for tx in transformers:
        i, j = tx.from_bus, tx.to_bus
        idx_i, idx_j = bus_indices[i], bus_indices[j]
        Pij = (theta[idx_i] - theta[idx_j]) / tx.XT
        branch_flows[(i, j)] = Pij

    # 直流潮流中损耗近似为0
    losses = 0.0

    return DCPowerFlowResult(
        theta=theta,
        P=P,
        branch_flows=branch_flows,
        losses=losses,
        bus_indices=bus_indices
    )
