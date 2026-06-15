"""
小干扰稳定分析 (Small Signal Stability Analysis)
=================================================

通过线性化状态方程和特征值分析，判断系统的小干扰稳定性。

数学模型
--------

### 单机无穷大系统

经典模型线性化：
    M × d²Δδ/dt² + D × dΔδ/dt + Ks × Δδ = 0

    其中：
    M = 2H/ωs —— 惯性系数
    D —— 阻尼系数
    Ks = ∂Pe/∂δ|δ₀ = (E'V∞/XΣ) × cos(δ₀) —— 同步转矩系数

    转换为状态空间形式：
    d[Δδ]/dt = [0       1    ] [Δδ]
    d[Δω]/dt = [-Ks/M  -D/M ] [Δω]

    特征方程：M×s² + D×s + Ks = 0
    特征值：s = (-D ± √(D²-4M×Ks)) / (2M)

    稳定判据：
    - 所有特征值实部 < 0 → 稳定
    - 阻尼比 ζ = -Re(λ) / |λ| > 0 → 稳定
    - 阻尼比越大，振荡衰减越快

详细模型线性化：
    增加Eq'和励磁系统状态变量，状态方程扩展为4×4或更高维。

### 多机系统

    dΔX/dt = A × ΔX

    其中：
    X —— 状态变量向量 [Δδ₁, Δω₁, ..., Δδₙ, Δωₙ]ᵀ
    A —— 系统状态矩阵（雅可比矩阵）

    稳定判据：
    - 所有特征值实部 < 0 → 稳定
    - 参与因子分析可找出振荡模式与状态变量的关联

参考教材：李光琦《电力系统暂态分析》第四章
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


@dataclass
class SmallSignalResult:
    """小干扰稳定分析结果

    Attributes:
        stable: 系统是否小干扰稳定
        eigenvalues: 特征值向量
        damping_ratios: 各振荡模式的阻尼比
        frequencies: 各振荡模式的振荡频率（Hz）
        participation_factors: 参与因子矩阵
        state_matrix: 系统状态矩阵A
    """
    stable: bool
    eigenvalues: NDArray[np.complex128]
    damping_ratios: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    participation_factors: Optional[NDArray[np.float64]]
    state_matrix: NDArray[np.float64]


def analyze_single_machine_infinite_bus(
    E_prime: float,
    V_infinity: float,
    X_total: float,
    H: float,
    D: float,
    delta_0: float,
    Pm: float,
    detailed: bool = False,
    Xd: float = 1.8,
    Xd_prime: float = 0.3,
    Xq: float = 1.7,
    Td0_prime: float = 8.0,
    Ka: float = 50.0,
    Te: float = 0.3,
    verbose: bool = False
) -> SmallSignalResult:
    """单机无穷大系统小干扰稳定分析

    Args:
        E_prime: 暂态电势
        V_infinity: 无穷大母线电压
        X_total: 等值电抗
        H: 惯性时间常数
        D: 阻尼系数
        delta_0: 运行功角（弧度）
        Pm: 机械功率
        detailed: 是否使用详细模型（否则使用经典模型）
        Xd: 同步电抗（详细模型）
        Xd_prime: 暂态电抗（详细模型）
        Xq: q轴同步电抗（详细模型）
        Td0_prime: 暂态时间常数（详细模型）
        Ka: 励磁放大倍数（详细模型）
        Te: 励磁机时间常数（详细模型）
        verbose: 是否打印详细信息

    Returns:
        SmallSignalResult: 小干扰稳定分析结果

    Note:
        经典模型（2阶）：
        状态矩阵：
            A = [0         1      ]
                [-Ks/(2H)  -D/(2H)]

        其中 Ks = (E'V∞/XΣ) × cos(δ₀)

        详细模型（4阶）：
        状态变量：[Δδ, Δω, ΔEq', ΔEfd]
        需要线性化更复杂的方程
    """
    omega_s = 2 * np.pi * 50  # 同步角频率（50Hz系统）

    if not detailed:
        # ===== 经典模型 =====
        # 同步转矩系数 Ks = ∂Pe/∂δ = (E'V∞/XΣ) × cos(δ₀)
        Ks = (E_prime * V_infinity / X_total) * np.cos(delta_0)
        M = 2 * H / omega_s

        # 状态矩阵
        A = np.array([
            [0,            1],
            [-Ks / M,     -D / M]
        ])

        eigenvalues = np.linalg.eigvals(A)
    else:
        # ===== 详细模型 =====
        # 状态变量: [Δδ, Δω, ΔEq', ΔEfd]

        # 计算初始运行点
        Id0 = (E_prime - V_infinity * np.cos(delta_0)) / (Xd_prime + X_total)
        Iq0 = V_infinity * np.sin(delta_0) / (Xq + X_total)
        Vd0 = Xq * Iq0  # d轴电压分量
        Vq0 = V_infinity * np.cos(delta_0) + X_total * Id0  # q轴电压分量
        Vt0 = np.sqrt(Vd0**2 + Vq0**2)

        # 偏导数计算
        # ∂Pe/∂δ
        dPe_dd = (E_prime * V_infinity * np.cos(delta_0) * (Xd_prime + X_total) +
                  V_infinity**2 * np.sin(delta_0) * (Xd_prime - Xq) * np.sin(delta_0)) / (Xd_prime + X_total)**2

        # ∂Pe/∂Eq'
        dPe_dEq = V_infinity * np.sin(delta_0) / (Xd_prime + X_total)

        # ∂Pe/∂Id（近似）
        dPe_dId = (Xd_prime - Xq) * Iq0

        # ∂Vt/∂δ
        dVt_dd = (Vd0 * dVd_dd(delta_0, V_infinity, X_total, Xq) +
                  Vq0 * dVq_dd(delta_0, V_infinity, X_total, Xd_prime)) / Vt0

        # ∂Vt/∂Eq'
        dVt_dEq = (Vd0 * dVd_dEq(X_total, Xq) +
                   Vq0 * dVq_dEq(X_total, Xd_prime)) / Vt0

        # ∂Id/∂δ
        dId_dd = V_infinity * np.sin(delta_0) / (Xd_prime + X_total)

        # ∂Id/∂Eq'
        dId_dEq = 1.0 / (Xd_prime + X_total)

        # 构造状态矩阵
        # [Δδ']     [0      ωs         0           0       ]
        # [Δω']   = [dPe_dd  -Dωs/(2H)  dPe_dEq    0       ] [Δδ]
        # [ΔEq']    [0       0          -1/Td0'     1/Td0'  ] [Δω]
        # [ΔEfd']   [0       0          -Ka*dVt_dEq/Te -1/Te ] [ΔEq']
        #                                                          [ΔEfd]
        A = np.array([
            [0,  omega_s,  0,           0],
            [dPe_dd/(2*H)*omega_s, -D*omega_s/(2*H),  dPe_dEq*omega_s/(2*H), 0],
            [0,  0,  -(1+(Xd-Xd_prime)*dId_dEq)/Td0_prime,  1/Td0_prime],
            [0,  0,  -Ka*dVt_dEq/Te,  -1/Te]
        ])

        eigenvalues = np.linalg.eigvals(A)

    # 分析特征值
    stable = all(e.real < 0 for e in eigenvalues)

    # 计算阻尼比和振荡频率
    damping_ratios = []
    frequencies = []
    for ev in eigenvalues:
        if abs(ev.imag) > 1e-6:  # 复数特征值（振荡模式）
            zeta = -ev.real / abs(ev)
            f = abs(ev.imag) / (2 * np.pi)
            damping_ratios.append(zeta)
            frequencies.append(f)
        else:  # 实数特征值（非振荡模式）
            damping_ratios.append(1.0 if ev.real < 0 else -1.0)
            frequencies.append(0.0)

    if verbose:
        print(f"=== 小干扰稳定分析结果 ===")
        print(f"模型: {'详细模型' if detailed else '经典模型'}")
        print(f"系统{'稳定' if stable else '不稳定'}")
        print(f"特征值:")
        for i, ev in enumerate(eigenvalues):
            if abs(ev.imag) > 1e-6:
                print(f"  λ{i+1} = {ev.real:.4f} + j{ev.imag:.4f} "
                      f"(ζ={damping_ratios[i]:.4f}, f={frequencies[i]:.2f}Hz)")
            else:
                print(f"  λ{i+1} = {ev.real:.4f} (非振荡)")

    return SmallSignalResult(
        stable=stable,
        eigenvalues=eigenvalues,
        damping_ratios=np.array(damping_ratios),
        frequencies=np.array(frequencies),
        participation_factors=None,
        state_matrix=A
    )


def dVd_dd(delta, V_inf, X_total, Xq):
    """∂Vd/∂δ"""
    return V_inf * np.cos(delta) * Xq / (Xq + X_total)

def dVd_dEq(X_total, Xq):
    """∂Vd/∂Eq'"""
    return 0.0

def dVq_dd(delta, V_inf, X_total, Xd_prime):
    """∂Vq/∂δ"""
    return -V_inf * np.sin(delta)

def dVq_dEq(X_total, Xd_prime):
    """∂Vq/∂Eq'"""
    return 1.0


def analyze_multi_machine(
    E_primes: List[float],
    H_list: List[float],
    D_list: List[float],
    delta_0_list: List[float],
    Ybus_reduced: NDArray[np.complex128],
    Pm_list: Optional[List[float]] = None,
    verbose: bool = False
) -> SmallSignalResult:
    """多机系统小干扰稳定分析

    Args:
        E_primes: 各发电机暂态电势列表
        H_list: 各发电机惯性时间常数列表
        D_list: 各发电机阻尼系数列表
        delta_0_list: 各发电机初始功角列表（弧度）
        Ybus_reduced: 导纳矩阵（只含发电机节点）
        Pm_list: 各发电机机械功率列表（可选）
        verbose: 是否打印详细信息

    Returns:
        SmallSignalResult: 小干扰稳定分析结果

    Note:
        多机系统状态方程：
        d[Δδ]/dt = [0      I    ] [Δδ]
        d[Δω]/dt = [M⁻¹K  M⁻¹D'] [Δω]

        其中：
        Kij = ∂Pei/∂δj —— 同步转矩系数矩阵
        M = diag(2Hi/ωs) —— 惯性矩阵
        D' = -diag(Di) —— 阻尼矩阵
    """
    n_gen = len(E_primes)
    omega_s = 2 * np.pi * 50

    G = Ybus_reduced.real
    B = Ybus_reduced.imag

    # 构造同步转矩系数矩阵
    K = np.zeros((n_gen, n_gen))
    for i in range(n_gen):
        for j in range(n_gen):
            alpha_ij = np.arctan2(G[i, j], B[i, j])
            Y_mag = np.sqrt(G[i, j]**2 + B[i, j]**2)

            if i == j:
                # 自转矩系数
                for k in range(n_gen):
                    alpha_ik = np.arctan2(G[i, k], B[i, k])
                    Y_mag_ik = np.sqrt(G[i, k]**2 + B[i, k]**2)
                    K[i, j] += E_primes[i] * E_primes[k] * Y_mag_ik * np.cos(
                        delta_0_list[i] - delta_0_list[k] - alpha_ik
                    )
            else:
                # 互转矩系数
                K[i, j] = -E_primes[i] * E_primes[j] * Y_mag * np.cos(
                    delta_0_list[i] - delta_0_list[j] - alpha_ij
                )

    # 惯性矩阵
    M_inv = np.diag([omega_s / (2 * H_list[i]) for i in range(n_gen)])
    D_diag = np.diag([-D_list[i] * omega_s / (2 * H_list[i]) for i in range(n_gen)])

    # 构造状态矩阵 (2n × 2n)
    A = np.zeros((2 * n_gen, 2 * n_gen))
    A[:n_gen, n_gen:] = np.eye(n_gen)          # dΔδ/dt = Δω
    A[n_gen:, :n_gen] = M_inv @ K              # dΔω/dt = M⁻¹K Δδ
    A[n_gen:, n_gen:] = D_diag                 # + D' Δω

    # 特征值分析
    eigenvalues = np.linalg.eigvals(A)
    stable = all(e.real < 0 for e in eigenvalues)

    # 计算阻尼比和频率
    damping_ratios = []
    frequencies = []
    for ev in eigenvalues:
        if abs(ev.imag) > 1e-6:
            zeta = -ev.real / abs(ev)
            f = abs(ev.imag) / (2 * np.pi)
            damping_ratios.append(zeta)
            frequencies.append(f)
        else:
            damping_ratios.append(1.0 if ev.real < 0 else -1.0)
            frequencies.append(0.0)

    # 计算参与因子（使用左右特征向量）
    # 参与因子 p_ij = |w_ij * v_ji|，其中 w 是左特征向量，v 是右特征向量
    # 左特征向量 = V^{-1} 的行向量（满足 w_i^T * A = λ_i * w_i^T）
    eigvals, right_eigvecs = np.linalg.eig(A)
    n_states = 2 * n_gen
    participation = np.zeros((n_states, n_states))

    try:
        # 左特征向量矩阵 L = inv(V)^T，行向量为左特征向量
        left_eigvecs = np.linalg.inv(right_eigvecs).T
        for i in range(n_states):
            for j in range(n_states):
                # p_ij = |left_eigvecs[i,j] * right_eigvecs[j,i]|
                participation[i, j] = abs(left_eigvecs[i, j] * right_eigvecs[j, i])
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，回退到右特征向量近似
        for i in range(n_states):
            for j in range(n_states):
                participation[i, j] = abs(right_eigvecs[j, i]) ** 2

    if verbose:
        print(f"=== 多机系统小干扰稳定分析 ===")
        print(f"发电机数: {n_gen}")
        print(f"系统{'稳定' if stable else '不稳定'}")
        print(f"\n特征值:")
        for i, ev in enumerate(eigenvalues):
            if abs(ev.imag) > 1e-6:
                idx = len([e for e in eigenvalues[:i+1] if abs(e.imag) > 1e-6]) - 1
                print(f"  λ{i+1} = {ev.real:.4f} + j{ev.imag:.4f} "
                      f"(ζ={damping_ratios[idx]:.4f}, f={frequencies[idx]:.2f}Hz)")
            else:
                print(f"  λ{i+1} = {ev.real:.4f} (非振荡)")

    return SmallSignalResult(
        stable=stable,
        eigenvalues=eigenvalues,
        damping_ratios=np.array(damping_ratios),
        frequencies=np.array(frequencies),
        participation_factors=participation,
        state_matrix=A
    )


def analyze_multi_machine_detailed(
    E_primes: List[float],
    H_list: List[float],
    D_list: List[float],
    delta_0_list: List[float],
    Ybus_reduced: NDArray[np.complex128],
    Xd_list: Optional[List[float]] = None,
    Xd_prime_list: Optional[List[float]] = None,
    Xq_list: Optional[List[float]] = None,
    Td0_prime_list: Optional[List[float]] = None,
    Ka_list: Optional[List[float]] = None,
    Te_list: Optional[List[float]] = None,
    exciter_params_list: Optional[List] = None,
    verbose: bool = False
) -> SmallSignalResult:
    """多机系统小干扰稳定分析（详细模型，含励磁系统动态）

    支持多机系统中每台发电机的磁链衰减动态（Eq'）和励磁系统动态，
    构造完整的线性化状态矩阵并进行特征值分析。

    Args:
        E_primes: 各发电机暂态电势 E'（标幺值）
        H_list: 各发电机惯性时间常数（秒）
        D_list: 各发电机阻尼系数
        delta_0_list: 各发电机初始功角（弧度）
        Ybus_reduced: 缩减导纳矩阵（仅含发电机内部节点）
        Xd_list: 各发电机 d 轴同步电抗（标幺值）
        Xd_prime_list: 各发电机 d 轴暂态电抗（标幺值）
        Xq_list: 各发电机 q 轴同步电抗（标幺值）
        Td0_prime_list: 各发电机 d 轴暂态开路时间常数（秒）
        Ka_list: 各励磁系统增益（简化励磁模型）
        Te_list: 各励磁系统时间常数（秒，简化励磁模型）
        exciter_params_list: 各励磁系统 IEEET1Params 列表（如有则优先使用）
        verbose: 是否打印详细信息

    Returns:
        SmallSignalResult: 小干扰稳定分析结果

    Note:
        状态变量（每台发电机 4 阶，使用简化励磁）:
            [Δδ₁, ..., Δδₙ, Δω₁, ..., Δωₙ, ΔEq'₁, ..., ΔEq'ₙ, ΔEfd₁, ..., ΔEfdₙ]ᵀ

        若提供 exciter_params_list (IEEET1)，则每台发电机增加至 6 阶:
            [..., ΔVR_i, ΔEfd_i, ΔRf_i]ᵀ

        线性化基于多机系统 dq 轴网络方程，包含:
        - 同步转矩耦合（K1 矩阵）
        - 磁链衰减效应（K3, K4 矩阵）
        - 励磁调压效应（K5, K6 矩阵）

        参考: Kundur Ch.12, Sec. 12.5-12.6
    """
    n_gen = len(E_primes)
    omega_s = 2 * np.pi * 50

    # 默认参数
    if Xd_list is None:
        Xd_list = [1.8] * n_gen
    if Xd_prime_list is None:
        Xd_prime_list = [0.3] * n_gen
    if Xq_list is None:
        Xq_list = [1.7] * n_gen
    if Td0_prime_list is None:
        Td0_prime_list = [8.0] * n_gen
    if Ka_list is None:
        Ka_list = [50.0] * n_gen
    if Te_list is None:
        Te_list = [0.3] * n_gen

    G = Ybus_reduced.real
    B = Ybus_reduced.imag

    # ================================================================
    # Step 1: 计算初始运行点的 dq 轴电气量
    # ================================================================
    Id0 = np.zeros(n_gen)
    Iq0 = np.zeros(n_gen)
    Pe0 = np.zeros(n_gen)
    Vd0 = np.zeros(n_gen)
    Vq0 = np.zeros(n_gen)
    Vt0 = np.zeros(n_gen)

    for i in range(n_gen):
        # dq 轴电流（多机网络方程）
        for k in range(n_gen):
            delta_ik = delta_0_list[k] - delta_0_list[i]
            Id0[i] += E_primes[k] * (
                G[i, k] * np.cos(delta_ik) - B[i, k] * np.sin(delta_ik)
            )
            Iq0[i] += E_primes[k] * (
                G[i, k] * np.sin(delta_ik) + B[i, k] * np.cos(delta_ik)
            )

        # 电气功率（含凸极效应）
        Pe0[i] = E_primes[i] * Iq0[i] + (
            Xq_list[i] - Xd_prime_list[i]
        ) * Id0[i] * Iq0[i]

        # 端电压 dq 分量
        Vd0[i] = Xq_list[i] * Iq0[i]
        Vq0[i] = E_primes[i] - Xd_prime_list[i] * Id0[i]
        Vt0[i] = np.sqrt(Vd0[i]**2 + Vq0[i]**2)

    # ================================================================
    # Step 2: 计算偏导数矩阵
    # ================================================================

    # dId/dδ, dIq/dδ, dId/dEq', dIq/dEq' (每个 n×n)
    dId_ddelta = np.zeros((n_gen, n_gen))
    dIq_ddelta = np.zeros((n_gen, n_gen))
    dId_dEq = np.zeros((n_gen, n_gen))
    dIq_dEq = np.zeros((n_gen, n_gen))

    for i in range(n_gen):
        for j in range(n_gen):
            delta_ji = delta_0_list[j] - delta_0_list[i]
            cos_ji = np.cos(delta_ji)
            sin_ji = np.sin(delta_ji)

            if i == j:
                # 对角线项: ∂Id_i/∂δ_i, ∂Iq_i/∂δ_i
                dId_ddelta[i, i] = E_primes[i] * B[i, i]
                dIq_ddelta[i, i] = -E_primes[i] * G[i, i]
                for k in range(n_gen):
                    delta_ki = delta_0_list[k] - delta_0_list[i]
                    cos_ki = np.cos(delta_ki)
                    sin_ki = np.sin(delta_ki)
                    dId_ddelta[i, i] += E_primes[k] * (
                        G[i, k] * sin_ki + B[i, k] * cos_ki
                    )
                    dIq_ddelta[i, i] += E_primes[k] * (
                        -G[i, k] * cos_ki + B[i, k] * sin_ki
                    )
                # 修正: 之前累加的对角项不对
                # 实际上:
                # dId_i/dδ_i = Σ_k E'_k * (G_ik*sin(δ_k-δ_i) + B_ik*cos(δ_k-δ_i))
                # dIq_i/dδ_i = Σ_k E'_k * (-G_ik*cos(δ_k-δ_i) + B_ik*sin(δ_k-δ_i))
                # 当 k=i: sin(0)=0, cos(0)=1
                # dId_i/dδ_i 含 E'_i * B_ii
                # dIq_i/dδ_i 含 -E'_i * G_ii
                # 这个计算是对的，但上面已经加了 E_primes[i]*B[i,i]，再在循环中又加了一次
                # 需要修正，去掉重复
                pass
            else:
                # 非对角线项
                dId_ddelta[i, j] = E_primes[j] * (
                    -G[i, j] * sin_ji - B[i, j] * cos_ji
                )
                dIq_ddelta[i, j] = E_primes[j] * (
                    G[i, j] * cos_ji - B[i, j] * sin_ji
                )

            # ∂Id_i/∂Eq'_j, ∂Iq_i/∂Eq'_j
            dId_dEq[i, j] = G[i, j] * cos_ji - B[i, j] * sin_ji
            dIq_dEq[i, j] = G[i, j] * sin_ji + B[i, j] * cos_ji

    # 修正对角线 dId/dδ 和 dIq/dδ（上面的双重计算需要修正）
    for i in range(n_gen):
        dId_ddelta[i, i] = 0.0
        dIq_ddelta[i, i] = 0.0
        for k in range(n_gen):
            delta_ki = delta_0_list[k] - delta_0_list[i]
            dId_ddelta[i, i] += E_primes[k] * (
                G[i, k] * np.sin(delta_ki) + B[i, k] * np.cos(delta_ki)
            )
            dIq_ddelta[i, i] += E_primes[k] * (
                -G[i, k] * np.cos(delta_ki) + B[i, k] * np.sin(delta_ki)
            )

    # dPe/dδ, dPe/dEq', dVt/dδ, dVt/dEq' (每个 n×n)
    dPe_ddelta = np.zeros((n_gen, n_gen))
    dPe_dEq = np.zeros((n_gen, n_gen))
    dVt_ddelta = np.zeros((n_gen, n_gen))
    dVt_dEq = np.zeros((n_gen, n_gen))

    for i in range(n_gen):
        Xq_i = Xq_list[i]
        Xdp_i = Xd_prime_list[i]

        for j in range(n_gen):
            # dPe_i/dδ_j
            dPe_ddelta[i, j] = (
                E_primes[i] * dIq_ddelta[i, j]
                + (Xq_i - Xdp_i) * (
                    dId_ddelta[i, j] * Iq0[i] + Id0[i] * dIq_ddelta[i, j]
                )
            )

            # dPe_i/dEq'_j
            kronecker = 1.0 if i == j else 0.0
            dPe_dEq[i, j] = (
                kronecker * Iq0[i]
                + E_primes[i] * dIq_dEq[i, j]
                + (Xq_i - Xdp_i) * (
                    dId_dEq[i, j] * Iq0[i] + Id0[i] * dIq_dEq[i, j]
                )
            )

            # dVt_i/dδ_j
            dVd_ddelta_ij = Xq_i * dIq_ddelta[i, j]
            dVq_ddelta_ij = -Xdp_i * dId_ddelta[i, j]
            if Vt0[i] > 1e-9:
                dVt_ddelta[i, j] = (
                    Vd0[i] * dVd_ddelta_ij + Vq0[i] * dVq_ddelta_ij
                ) / Vt0[i]

            # dVt_i/dEq'_j
            dVd_dEq_ij = Xq_i * dIq_dEq[i, j]
            dVq_dEq_ij = kronecker - Xdp_i * dId_dEq[i, j]
            if Vt0[i] > 1e-9:
                dVt_dEq[i, j] = (
                    Vd0[i] * dVd_dEq_ij + Vq0[i] * dVq_dEq_ij
                ) / Vt0[i]

    # ================================================================
    # Step 3: 确定是否使用 IEEET1
    # ================================================================
    use_ieeet1 = exciter_params_list is not None and len(exciter_params_list) == n_gen

    if use_ieeet1:
        # IEEET1: 每台发电机 6 阶
        # 状态: [δ(×n), ω(×n), Eq'(×n), VR(×n), Efd(×n), Rf(×n)]
        n_states_per_gen = 6
        n_states = n_gen * n_states_per_gen
        A = np.zeros((n_states, n_states))

        # 索引偏移
        i_delta = 0
        i_omega = n_gen
        i_Eqp = 2 * n_gen
        i_VR = 3 * n_gen
        i_Efd = 4 * n_gen
        i_Rf = 5 * n_gen

        for i in range(n_gen):
            # dΔδ_i/dt = Δω_i (Δω in rad/s)
            A[i_delta + i, i_omega + i] = 1.0

            # dΔω_i/dt = (ωs/(2H_i)) * (Σ_dPe_dδ·Δδ + Σ_dPe_dEq·ΔEq' - D_i·Δω_i)
            M_coeff = omega_s / (2 * H_list[i])
            for j in range(n_gen):
                A[i_omega + i, i_delta + j] = dPe_ddelta[i, j] * M_coeff
                A[i_omega + i, i_Eqp + j] = dPe_dEq[i, j] * M_coeff
            A[i_omega + i, i_omega + i] = -D_list[i] * M_coeff

            # dΔEq'_i/dt = (ΔEfd_i - ΔEq'_i - (Xd-Xd')·ΔId_i) / Td0'
            Td0 = Td0_prime_list[i]
            Xd_Xdp = Xd_list[i] - Xd_prime_list[i]
            for j in range(n_gen):
                A[i_Eqp + i, i_delta + j] = -Xd_Xdp * dId_ddelta[i, j] / Td0
                A[i_Eqp + i, i_Eqp + j] = (
                    -1.0 - Xd_Xdp * dId_dEq[i, j]
                ) / Td0
            A[i_Eqp + i, i_Efd + i] = 1.0 / Td0

            # IEEET1 励磁动态（3 阶）
            exc = exciter_params_list[i]
            # 饱和在工作点处的等效 KE
            Efd0 = Vt0[i]  # 近似：稳态励磁电压 ≈ 端电压标幺值
            dSE = exc._A_sat * exc._B_sat * np.exp(exc._B_sat * Efd0)
            KE_eff = exc.KE + dSE

            TA = exc.TA if exc.TA > 0 else 1e-6
            TE = exc.TE if exc.TE > 0 else 1e-6
            TF = exc.TF if exc.TF > 0 else 1e-6

            # dΔVR/dt = (-KA/TA)·ΔVt - ΔVR/TA
            for j in range(n_gen):
                A[i_VR + i, i_delta + j] = -exc.KA * dVt_ddelta[i, j] / TA
                A[i_VR + i, i_Eqp + j] = -exc.KA * dVt_dEq[i, j] / TA
            A[i_VR + i, i_VR + i] = -1.0 / TA

            # dΔEfd/dt = ΔVR/TE - KE_eff/TE·ΔEfd
            A[i_Efd + i, i_VR + i] = 1.0 / TE
            A[i_Efd + i, i_Efd + i] = -KE_eff / TE

            # dΔRf/dt = (KF/(TF·TE))·ΔVR - (KF·KE_eff)/(TF·TE)·ΔEfd - ΔRf/TF
            A[i_Rf + i, i_VR + i] = exc.KF / (TF * TE)
            A[i_Rf + i, i_Efd + i] = -exc.KF * KE_eff / (TF * TE)
            A[i_Rf + i, i_Rf + i] = -1.0 / TF

    else:
        # 简化励磁: 每台发电机 4 阶
        # 状态: [δ(×n), ω(×n), Eq'(×n), Efd(×n)]
        n_states_per_gen = 4
        n_states = n_gen * n_states_per_gen
        A = np.zeros((n_states, n_states))

        i_delta = 0
        i_omega = n_gen
        i_Eqp = 2 * n_gen
        i_Efd = 3 * n_gen

        for i in range(n_gen):
            # dΔδ_i/dt = Δω_i (Δω in rad/s)
            A[i_delta + i, i_omega + i] = 1.0

            # dΔω_i/dt = (ωs/(2H_i)) * (...)
            M_coeff_i = omega_s / (2 * H_list[i])
            for j in range(n_gen):
                A[i_omega + i, i_delta + j] = dPe_ddelta[i, j] * M_coeff_i
                A[i_omega + i, i_Eqp + j] = dPe_dEq[i, j] * M_coeff_i
            A[i_omega + i, i_omega + i] = -D_list[i] * M_coeff_i

            # dΔEq'_i/dt
            Td0 = Td0_prime_list[i]
            Xd_Xdp = Xd_list[i] - Xd_prime_list[i]
            for j in range(n_gen):
                A[i_Eqp + i, i_delta + j] = -Xd_Xdp * dId_ddelta[i, j] / Td0
                A[i_Eqp + i, i_Eqp + j] = (
                    -1.0 - Xd_Xdp * dId_dEq[i, j]
                ) / Td0
            A[i_Eqp + i, i_Efd + i] = 1.0 / Td0

            # dΔEfd_i/dt = (-KA_i/Te_i)·ΔVt_i - ΔEfd_i/Te_i
            Ka = Ka_list[i]
            Te = Te_list[i]
            for j in range(n_gen):
                A[i_Efd + i, i_delta + j] = -Ka * dVt_ddelta[i, j] / Te
                A[i_Efd + i, i_Eqp + j] = -Ka * dVt_dEq[i, j] / Te
            A[i_Efd + i, i_Efd + i] = -1.0 / Te

    # ================================================================
    # Step 4: 特征值分析
    # ================================================================
    eigenvalues = np.linalg.eigvals(A)
    stable = all(e.real < 0 for e in eigenvalues)

    # 阻尼比和频率
    damping_ratios = []
    frequencies = []
    for ev in eigenvalues:
        if abs(ev.imag) > 1e-6:
            zeta = -ev.real / abs(ev)
            f = abs(ev.imag) / (2 * np.pi)
            damping_ratios.append(zeta)
            frequencies.append(f)
        else:
            damping_ratios.append(1.0 if ev.real < 0 else -1.0)
            frequencies.append(0.0)

    # 参与因子
    eigvals, right_eigvecs = np.linalg.eig(A)
    participation = np.zeros((n_states, n_states))
    try:
        left_eigvecs = np.linalg.inv(right_eigvecs).T
        for i in range(n_states):
            for j in range(n_states):
                participation[i, j] = abs(left_eigvecs[i, j] * right_eigvecs[j, i])
    except np.linalg.LinAlgError:
        for i in range(n_states):
            for j in range(n_states):
                participation[i, j] = abs(right_eigvecs[j, i]) ** 2

    if verbose:
        print(f"=== 多机系统小干扰稳定分析（详细模型）===")
        print(f"发电机数: {n_gen}")
        if use_ieeet1:
            print(f"励磁系统: IEEET1 ({n_states} 阶状态矩阵)")
        else:
            print(f"励磁系统: 简化励磁 KA/Te ({n_states} 阶状态矩阵)")
        print(f"系统{'稳定' if stable else '不稳定'}")
        print(f"\n初始运行点:")
        for i in range(n_gen):
            print(f"  发电机{i+1}: Pe={Pe0[i]:.4f}, Vt={Vt0[i]:.4f}, "
                  f"Id={Id0[i]:.4f}, Iq={Iq0[i]:.4f}")
        print(f"\n特征值（仅显示振荡模式）:")
        osc_modes = [(abs(ev.imag), ev, damping_ratios[idx], frequencies[idx])
                     for idx, ev in enumerate(eigenvalues) if abs(ev.imag) > 1e-6]
        osc_modes.sort(key=lambda x: x[3])  # 按频率排序
        for _, ev, zeta, f in osc_modes:
            print(f"  λ = {ev.real:+.4f} ± j{abs(ev.imag):.4f}  "
                  f"(ζ={zeta:+.4f}, f={f:.4f} Hz)")

    return SmallSignalResult(
        stable=stable,
        eigenvalues=eigenvalues,
        damping_ratios=np.array(damping_ratios),
        frequencies=np.array(frequencies),
        participation_factors=participation,
        state_matrix=A
    )