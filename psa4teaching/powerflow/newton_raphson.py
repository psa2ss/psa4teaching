"""
牛顿-拉夫逊潮流计算 (Newton-Raphson Power Flow)
===============================================

使用牛顿-拉夫逊法求解电力系统潮流。

数学模型
--------
极坐标形式的潮流方程：

有功功率方程：
    Pi = Vi × Σⱼ Vj × (Gij×cos(θi-θj) + Bij×sin(θi-θj))

无功功率方程：
    Qi = Vi × Σⱼ Vj × (Gij×sin(θi-θj) - Bij×cos(θi-θj))

牛顿-拉夫逊迭代：
    [ΔP]   [H  N] [Δθ]
    [  ] = [    ] [  ]
    [ΔQ]   [J  L] [ΔV]

雅可比矩阵元素：
    H_ii = ∂Pi/∂θi = -Qi - Vi²×Bii
    H_ij = ∂Pi/∂θj = Vi×Vj×(Gij×sin(θij) - Bij×cos(θij))
    N_ii = ∂Pi/∂Vi = Pi/Vi + Vi×Gii
    N_ij = ∂Pi/∂Vj = Vi×(Gij×cos(θij) + Bij×sin(θij))
    J_ii = ∂Qi/∂θi = Pi - Vi²×Gii
    J_ij = ∂Qi/∂θj = -Vi×Vj×(Gij×cos(θij) + Bij×sin(θij))
    L_ii = ∂Qi/∂Vi = Qi/Vi - Vi×Bii
    L_ij = ∂Qi/∂Vj = Vi×(Gij×sin(θij) - Bij×cos(θij))

参考教材：陈珩《电力系统稳态分析》第四章 4.2-4.3节
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.network.ybus import YBusResult


@dataclass
class NewtonRaphsonResult:
    """牛顿-拉夫逊潮流计算结果

    Attributes:
        converged: 是否收敛
        iterations: 迭代次数
        V: 最终电压幅值向量（标幺值）
        delta: 最终电压相角向量（弧度）
        P: 节点有功注入向量
        Q: 节点无功注入向量
        P_mismatch: 有功功率偏差向量
        Q_mismatch: 无功功率偏差向量
        max_mismatch: 最大功率偏差
        losses: 网络总有功损耗
    """
    converged: bool
    iterations: int
    V: NDArray[np.float64]
    delta: NDArray[np.float64]
    P: NDArray[np.float64]
    Q: NDArray[np.float64]
    P_mismatch: NDArray[np.float64]
    Q_mismatch: NDArray[np.float64]
    max_mismatch: float
    losses: float
    history: List[Dict] = None


def run_newton_raphson(
    buses: List[Bus],
    ybus_result: YBusResult,
    max_iterations: int = 50,
    tolerance: float = 1e-8,
    verbose: bool = False
) -> NewtonRaphsonResult:
    """执行牛顿-拉夫逊潮流计算

    Args:
        buses: 节点列表
        ybus_result: 节点导纳矩阵结果
        max_iterations: 最大迭代次数
        tolerance: 收敛精度（最大功率偏差）
        verbose: 是否打印迭代过程

    Returns:
        NewtonRaphsonResult: 潮流计算结果

    Example:
        >>> buses = [Bus(1, bus_type=BusType.SLACK, V_specified=1.05),
        ...          Bus(2, bus_type=BusType.PV, P_specified=0.5, V_specified=1.02),
        ...          Bus(3, bus_type=BusType.PQ, P_specified=-0.8, Q_specified=-0.3)]
        >>> result = run_newton_raphson(buses, ybus_result)
        >>> print(f"收敛: {result.converged}, 迭代次数: {result.iterations}")
    """
    Ybus = ybus_result.Ybus
    G = ybus_result.G
    B = ybus_result.B
    n_bus = ybus_result.n_bus
    bus_indices = ybus_result.bus_indices

    # 初始化状态变量
    V = np.ones(n_bus)
    delta = np.zeros(n_bus)

    # 设置初始值和指定值
    pq_buses = []
    pv_buses = []
    slack_bus = None

    for bus in buses:
        idx = bus_indices[bus.number]
        V[idx] = bus.V_specified if bus.V_specified > 0 else 1.0
        delta[idx] = 0.0

        if bus.bus_type == BusType.PQ:
            pq_buses.append(bus)
        elif bus.bus_type == BusType.PV:
            pv_buses.append(bus)
        else:  # SLACK
            slack_bus = bus

    n_pq = len(pq_buses)
    n_pv = len(pv_buses)

    # 迭代历史
    history = []
    converged = False

    for iteration in range(max_iterations):
        # 计算功率偏差
        P_calc, Q_calc = _calculate_power(Ybus, V, delta, n_bus)
        P_mismatch, Q_mismatch = _calculate_mismatch(
            buses, bus_indices, P_calc, Q_calc, pq_buses, pv_buses
        )

        max_mismatch = max(
            np.max(np.abs(P_mismatch)),
            np.max(np.abs(Q_mismatch)) if n_pq > 0 else 0.0
        )

        if verbose:
            print(f"迭代 {iteration + 1}: 最大偏差 = {max_mismatch:.2e}")

        history.append({
            'iteration': iteration + 1,
            'max_mismatch': max_mismatch,
            'P_mismatch': P_mismatch.copy(),
            'Q_mismatch': Q_mismatch.copy() if n_pq > 0 else np.array([])
        })

        # 检查收敛
        if max_mismatch < tolerance:
            converged = True
            break

        # 构造雅可比矩阵
        J = build_jacobian(Ybus, G, B, V, delta, pq_buses, pv_buses, bus_indices)

        # 构造偏差向量
        mismatch = np.concatenate([P_mismatch, Q_mismatch])

        # 求解修正方程
        try:
            correction = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            if verbose:
                print("雅可比矩阵奇异，无法求解")
            break

        # 更新状态变量
        n_p = n_pv + n_pq
        delta_correction = correction[:n_p]
        V_correction = correction[n_p:] if n_pq > 0 else np.array([])

        _update_state_variables(
            buses, bus_indices, V, delta,
            delta_correction, V_correction, pq_buses, pv_buses, slack_bus
        )

        # PV节点无功越限处理
        _handle_q_limits(buses, bus_indices, Ybus, V, delta, pv_buses)

    # 最终功率计算
    P_final, Q_final = _calculate_power(Ybus, V, delta, n_bus)

    # 计算网络损耗
    losses = _calculate_losses(Ybus, V, delta, n_bus)

    return NewtonRaphsonResult(
        converged=converged,
        iterations=iteration + 1,
        V=V,
        delta=delta,
        P=P_final,
        Q=Q_final,
        P_mismatch=P_mismatch,
        Q_mismatch=Q_mismatch,
        max_mismatch=max_mismatch,
        losses=losses,
        history=history
    )


def build_jacobian(
    Ybus: NDArray[np.complex128],
    G: NDArray[np.float64],
    B: NDArray[np.float64],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    pq_buses: List[Bus],
    pv_buses: List[Bus],
    bus_indices: Dict[int, int]
) -> NDArray[np.float64]:
    """构造雅可比矩阵

    Args:
        Ybus: 节点导纳矩阵
        G: 电导矩阵
        B: 电纳矩阵
        V: 电压幅值向量
        delta: 电压相角向量
        pq_buses: PQ节点列表
        pv_buses: PV节点列表
        bus_indices: 节点编号到索引的映射

    Returns:
        J: 雅可比矩阵，形状为 (n_P + n_Q, n_P + n_Q)
        其中 n_P = n_pv + n_pq, n_Q = n_pq

    Note:
        雅可比矩阵结构：
            [H  N]
            [J  L]
        H: ∂P/∂θ, 大小 n_P × n_P
        N: ∂P/∂V, 大小 n_P × n_Q
        J: ∂Q/∂θ, 大小 n_Q × n_P
        L: ∂Q/∂V, 大小 n_Q × n_Q
    """
    n_pq = len(pq_buses)
    n_pv = len(pv_buses)
    n_p = n_pv + n_pq
    n_q = n_pq

    J = np.zeros((n_p + n_q, n_p + n_q))

    # 获取P和Q对应的节点索引
    p_indices = [bus_indices[bus.number] for bus in pv_buses + pq_buses]
    q_indices = [bus_indices[bus.number] for bus in pq_buses]

    n_bus = len(V)

    # 构造H矩阵 (∂P/∂θ)
    for i, idx_i in enumerate(p_indices):
        for j, idx_j in enumerate(p_indices):
            if i == j:
                # 对角元: H_ii = -Qi - Vi²×Bii
                Qi = V[idx_i] * sum(
                    V[k] * (G[idx_i, k] * np.sin(delta[idx_i] - delta[k]) -
                            B[idx_i, k] * np.cos(delta[idx_i] - delta[k]))
                    for k in range(n_bus)
                )
                J[i, j] = -Qi - V[idx_i]**2 * B[idx_i, idx_i]
            else:
                # 非对角元: H_ij = Vi×Vj×(Gij×sin(θij) - Bij×cos(θij))
                theta_ij = delta[idx_i] - delta[idx_j]
                J[i, j] = (V[idx_i] * V[idx_j] *
                          (G[idx_i, idx_j] * np.sin(theta_ij) -
                           B[idx_i, idx_j] * np.cos(theta_ij)))

    # 构造N矩阵 (∂P/∂V)
    if n_q > 0:
        for i, idx_i in enumerate(p_indices):
            for j, idx_j in enumerate(q_indices):
                if idx_i == idx_j:
                    # 对角元: N_ii = Pi/Vi + Vi×Gii
                    Pi = V[idx_i] * sum(
                        V[k] * (G[idx_i, k] * np.cos(delta[idx_i] - delta[k]) +
                                B[idx_i, k] * np.sin(delta[idx_i] - delta[k]))
                        for k in range(n_bus)
                    )
                    J[i, n_p + j] = Pi / V[idx_i] + V[idx_i] * G[idx_i, idx_i]
                else:
                    # 非对角元: N_ij = Vi×(Gij×cos(θij) + Bij×sin(θij))
                    theta_ij = delta[idx_i] - delta[idx_j]
                    J[i, n_p + j] = V[idx_i] * (
                        G[idx_i, idx_j] * np.cos(theta_ij) +
                        B[idx_i, idx_j] * np.sin(theta_ij)
                    )

    # 构造J矩阵 (∂Q/∂θ)
    if n_q > 0:
        for i, idx_i in enumerate(q_indices):
            for j, idx_j in enumerate(p_indices):
                if idx_i == idx_j:
                    # 对角元: J_ii = Pi - Vi²×Gii
                    Pi = V[idx_i] * sum(
                        V[k] * (G[idx_i, k] * np.cos(delta[idx_i] - delta[k]) +
                                B[idx_i, k] * np.sin(delta[idx_i] - delta[k]))
                        for k in range(n_bus)
                    )
                    J[n_p + i, j] = Pi - V[idx_i]**2 * G[idx_i, idx_i]
                else:
                    # 非对角元: J_ij = -Vi×Vj×(Gij×cos(θij) + Bij×sin(θij))
                    theta_ij = delta[idx_i] - delta[idx_j]
                    J[n_p + i, j] = -V[idx_i] * V[idx_j] * (
                        G[idx_i, idx_j] * np.cos(theta_ij) +
                        B[idx_i, idx_j] * np.sin(theta_ij)
                    )

    # 构造L矩阵 (∂Q/∂V)
    if n_q > 0:
        for i, idx_i in enumerate(q_indices):
            for j, idx_j in enumerate(q_indices):
                if i == j:
                    # 对角元: L_ii = Qi/Vi - Vi×Bii
                    Qi = V[idx_i] * sum(
                        V[k] * (G[idx_i, k] * np.sin(delta[idx_i] - delta[k]) -
                                B[idx_i, k] * np.cos(delta[idx_i] - delta[k]))
                        for k in range(n_bus)
                    )
                    J[n_p + i, n_p + j] = Qi / V[idx_i] - V[idx_i] * B[idx_i, idx_i]
                else:
                    # 非对角元: L_ij = Vi×(Gij×sin(θij) - Bij×cos(θij))
                    theta_ij = delta[idx_i] - delta[idx_j]
                    J[n_p + i, n_p + j] = V[idx_i] * (
                        G[idx_i, idx_j] * np.sin(theta_ij) -
                        B[idx_i, idx_j] * np.cos(theta_ij)
                    )

    return J


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


def _calculate_mismatch(
    buses: List[Bus],
    bus_indices: Dict[int, int],
    P_calc: NDArray[np.float64],
    Q_calc: NDArray[np.float64],
    pq_buses: List[Bus],
    pv_buses: List[Bus]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """计算功率偏差"""
    # P偏差（PQ和PV节点）
    P_mismatch = np.array([
        buses[i].P_specified - P_calc[bus_indices[buses[i].number]]
        for i in range(len(buses))
        if buses[i].bus_type in [BusType.PQ, BusType.PV]
    ]) if len(pq_buses) + len(pv_buses) > 0 else np.array([])

    # 重新按pv_buses + pq_buses顺序计算
    P_mismatch = np.array([
        bus.P_specified - P_calc[bus_indices[bus.number]]
        for bus in pv_buses + pq_buses
    ])

    # Q偏差（仅PQ节点）
    Q_mismatch = np.array([
        bus.Q_specified - Q_calc[bus_indices[bus.number]]
        for bus in pq_buses
    ]) if len(pq_buses) > 0 else np.array([])

    return P_mismatch, Q_mismatch


def _update_state_variables(
    buses: List[Bus],
    bus_indices: Dict[int, int],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    delta_correction: NDArray[np.float64],
    V_correction: NDArray[np.float64],
    pq_buses: List[Bus],
    pv_buses: List[Bus],
    slack_bus: Bus
):
    """更新状态变量"""
    # 更新相角（PV和PQ节点）
    for i, bus in enumerate(pv_buses + pq_buses):
        idx = bus_indices[bus.number]
        delta[idx] += delta_correction[i]

    # 更新电压幅值（仅PQ节点）
    for i, bus in enumerate(pq_buses):
        idx = bus_indices[bus.number]
        V[idx] += V_correction[i]
        if V[idx] < 0.5:  # 电压下限
            V[idx] = 0.5
        elif V[idx] > 1.5:  # 电压上限
            V[idx] = 1.5


def _handle_q_limits(
    buses: List[Bus],
    bus_indices: Dict[int, int],
    Ybus: NDArray[np.complex128],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    pv_buses: List[Bus]
):
    """处理PV节点无功越限"""
    n_bus = len(V)

    for bus in pv_buses:
        idx = bus_indices[bus.number]

        # 计算无功出力
        Q = 0.0
        for j in range(n_bus):
            theta_ij = delta[idx] - delta[j]
            Gij = Ybus[idx, j].real
            Bij = Ybus[idx, j].imag
            Q += V[idx] * V[j] * (Gij * np.sin(theta_ij) - Bij * np.cos(theta_ij))

        # 检查越限
        if Q < bus.Q_min or Q > bus.Q_max:
            # 转换为PQ节点
            bus.bus_type = BusType.PQ
            bus.Q_specified = min(max(Q, bus.Q_min), bus.Q_max)


def _calculate_losses(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.float64],
    delta: NDArray[np.float64],
    n_bus: int
) -> float:
    """计算网络总有功损耗"""
    # 总注入功率
    P_total = 0.0
    for i in range(n_bus):
        for j in range(n_bus):
            theta_ij = delta[i] - delta[j]
            Gij = Ybus[i, j].real
            Bij = Ybus[i, j].imag
            P_total += V[i] * V[j] * (Gij * np.cos(theta_ij) + Bij * np.sin(theta_ij))

    # 网络损耗 = 总发电 - 总负荷 ≈ 总注入（平衡节点吸收）
    # 更准确的方法是计算所有支路损耗之和
    losses = 0.0
    for i in range(n_bus):
        for j in range(i + 1, n_bus):
            if Ybus[i, j] != 0:
                theta_ij = delta[i] - delta[j]
                Gij = Ybus[i, j].real
                Iij = (V[i] * np.cos(delta[i]) - V[j] * np.cos(delta[j])) * (-Gij)
                # 简化计算
                conductance = -Ybus[i, j].real
                losses += conductance * (V[i]**2 + V[j]**2 - 2*V[i]*V[j]*np.cos(theta_ij))

    return losses