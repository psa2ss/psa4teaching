"""
P-Q分解法潮流计算 (Fast Decoupled Power Flow)
=============================================

使用P-Q分解法（快速解耦潮流）求解电力系统潮流。

数学模型
--------
P-Q分解法基于两个假设：
1. 电网中电抗远大于电阻（X >> R）
2. 节点电压相角差很小

基于上述假设，雅可比矩阵可简化为：
    [ΔP/V]   [B'   0 ] [Δθ]
    [     ] = [       ] [   ]
    [ΔQ/V]   [0   B''] [ΔV]

其中：
    B' = -Bij（简化为仅考虑电纳）
    B'' = -Bij（简化为仅考虑电纳）

迭代公式：
    Δθ = B'⁻¹ × (ΔP/V)
    ΔV = B''⁻¹ × (ΔQ/V)

优点：
    - 雅可比矩阵为常数矩阵，只需求逆一次
    - 迭代速度快，存储空间小
    - 收敛性较好（线性收敛）

参考教材：陈珩《电力系统稳态分析》第四章 4.4节
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.network.ybus import YBusResult


@dataclass
class FastDecoupledResult:
    """P-Q分解法潮流计算结果

    Attributes:
        converged: 是否收敛
        iterations: 迭代次数
        V: 最终电压幅值向量
        delta: 最终电压相角向量
        P: 节点有功注入向量
        Q: 节点无功注入向量
        max_mismatch: 最大功率偏差
        losses: 网络总有功损耗
    """
    converged: bool
    iterations: int
    V: NDArray[np.float64]
    delta: NDArray[np.float64]
    P: NDArray[np.float64]
    Q: NDArray[np.float64]
    max_mismatch: float
    losses: float
    history: List[Dict] = None


def run_fast_decoupled(
    buses: List[Bus],
    ybus_result: YBusResult,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> FastDecoupledResult:
    """执行P-Q分解法潮流计算

    Args:
        buses: 节点列表
        ybus_result: 节点导纳矩阵结果
        max_iterations: 最大迭代次数
        tolerance: 收敛精度
        verbose: 是否打印迭代过程

    Returns:
        FastDecoupledResult: 潮流计算结果

    Example:
        >>> result = run_fast_decoupled(buses, ybus_result)
        >>> print(f"收敛: {result.converged}, 迭代次数: {result.iterations}")
    """
    Ybus = ybus_result.Ybus
    B = ybus_result.B
    n_bus = ybus_result.n_bus
    bus_indices = ybus_result.bus_indices

    # 分类节点
    pq_buses = []
    pv_buses = []
    slack_bus = None

    for bus in buses:
        if bus.bus_type == BusType.PQ:
            pq_buses.append(bus)
        elif bus.bus_type == BusType.PV:
            pv_buses.append(bus)
        else:
            slack_bus = bus

    n_pq = len(pq_buses)
    n_pv = len(pv_buses)

    # 初始化状态变量
    V = np.ones(n_bus)
    delta = np.zeros(n_bus)

    for bus in buses:
        idx = bus_indices[bus.number]
        V[idx] = bus.V_specified if bus.V_specified > 0 else 1.0

    # 构造B'和B''矩阵
    B1, B2 = build_b_matrices(Ybus, B, pq_buses, pv_buses, bus_indices)

    # 预先求逆
    try:
        B1_inv = np.linalg.inv(B1)
        B2_inv = np.linalg.inv(B2) if n_pq > 0 else None
    except np.linalg.LinAlgError:
        raise ValueError("B'或B''矩阵奇异，无法求逆")

    # 迭代求解
    history = []
    converged = False

    for iteration in range(max_iterations):
        # 计算功率偏差
        P_calc, Q_calc = _calculate_power(Ybus, V, delta, n_bus)

        # P偏差
        P_mismatch = np.array([
            bus.P_specified - P_calc[bus_indices[bus.number]]
            for bus in pv_buses + pq_buses
        ])

        # Q偏差
        Q_mismatch = np.array([
            bus.Q_specified - Q_calc[bus_indices[bus.number]]
            for bus in pq_buses
        ]) if n_pq > 0 else np.array([])

        max_mismatch = max(
            np.max(np.abs(P_mismatch)),
            np.max(np.abs(Q_mismatch)) if n_pq > 0 else 0.0
        )

        if verbose:
            print(f"迭代 {iteration + 1}: 最大偏差 = {max_mismatch:.2e}")

        history.append({
            'iteration': iteration + 1,
            'max_mismatch': max_mismatch
        })

        # 检查收敛
        if max_mismatch < tolerance:
            converged = True
            break

        # P-θ迭代
        P_mismatch_normalized = P_mismatch / np.array([
            V[bus_indices[bus.number]] for bus in pv_buses + pq_buses
        ])
        delta_correction = B1_inv @ P_mismatch_normalized

        # 更新相角
        for i, bus in enumerate(pv_buses + pq_buses):
            idx = bus_indices[bus.number]
            delta[idx] += delta_correction[i]

        # Q-V迭代
        if n_pq > 0:
            # 重新计算Q偏差
            _, Q_calc = _calculate_power(Ybus, V, delta, n_bus)
            Q_mismatch = np.array([
                bus.Q_specified - Q_calc[bus_indices[bus.number]]
                for bus in pq_buses
            ])

            Q_mismatch_normalized = Q_mismatch / np.array([
                V[bus_indices[bus.number]] for bus in pq_buses
            ])
            V_correction = B2_inv @ Q_mismatch_normalized

            # 更新电压
            for i, bus in enumerate(pq_buses):
                idx = bus_indices[bus.number]
                V[idx] += V_correction[i]
                V[idx] = np.clip(V[idx], 0.5, 1.5)

    # 最终功率计算
    P_final, Q_final = _calculate_power(Ybus, V, delta, n_bus)

    # 计算损耗
    losses = _calculate_losses(Ybus, V, delta, n_bus)

    return FastDecoupledResult(
        converged=converged,
        iterations=iteration + 1,
        V=V,
        delta=delta,
        P=P_final,
        Q=Q_final,
        max_mismatch=max_mismatch,
        losses=losses,
        history=history
    )


def build_b_matrices(
    Ybus: NDArray[np.complex128],
    B: NDArray[np.float64],
    pq_buses: List[Bus],
    pv_buses: List[Bus],
    bus_indices: Dict[int, int]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """构造B'和B''矩阵

    Args:
        Ybus: 节点导纳矩阵
        B: 电纳矩阵
        pq_buses: PQ节点列表
        pv_buses: PV节点列表
        bus_indices: 节点编号到索引的映射

    Returns:
        (B1, B2): B'矩阵和B''矩阵

    Note:
        B'矩阵构造方法（XB型）：
        - 忽略电阻，仅考虑电纳
        - 忽略并联支路
        - B'_ij = -Bij
        - B'_ii = Σ|Bij|

        B''矩阵构造方法：
        - B''_ij = -Bij
        - B''_ii = Σ|Bij|（仅PQ节点）
    """
    n_pq = len(pq_buses)
    n_pv = len(pv_buses)
    n_p = n_pv + n_pq

    # 节点索引
    p_indices = [bus_indices[bus.number] for bus in pv_buses + pq_buses]
    q_indices = [bus_indices[bus.number] for bus in pq_buses]

    # 构造B'矩阵
    B1 = np.zeros((n_p, n_p))
    for i, idx_i in enumerate(p_indices):
        for j, idx_j in enumerate(p_indices):
            if i == j:
                # 对角元
                B1[i, j] = -B[idx_i, idx_i]
                # 或者用 Σ|Bij|
                # B1[i, j] = sum(abs(B[idx_i, k]) for k in p_indices)
            else:
                B1[i, j] = -B[idx_i, idx_j]

    # 构造B''矩阵
    B2 = np.zeros((n_pq, n_pq))
    if n_pq > 0:
        for i, idx_i in enumerate(q_indices):
            for j, idx_j in enumerate(q_indices):
                if i == j:
                    B2[i, j] = -B[idx_i, idx_i]
                else:
                    B2[i, j] = -B[idx_i, idx_j]

    return B1, B2


def _calculate_power(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    n_bus: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """计算节点注入功率"""
    P = np.zeros(n_bus)
    Q = np.zeros(n_bus)

    for i in range(n_bus):
        for j in range(n_bus):
            theta_ij = delta[i] - delta[j]
            Gij = Ybus[i, j].real
            Bij = Ybus[i, j].imag

            P[i] += V[i] * V[j] * (Gij * np.cos(theta_ij) + Bij * np.sin(theta_ij))
            Q[i] += V[i] * V[j] * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij))

    return P, Q


def _calculate_losses(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    n_bus: int
) -> float:
    """计算网络总有功损耗"""
    losses = 0.0
    for i in range(n_bus):
        for j in range(i + 1, n_bus):
            if Ybus[i, j] != 0:
                theta_ij = delta[i] - delta[j]
                conductance = -Ybus[i, j].real
                losses += conductance * (V[i]**2 + V[j]**2 - 2*V[i]*V[j]*np.cos(theta_ij))

    return losses