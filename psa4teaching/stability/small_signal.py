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

    # 计算参与因子
    eigvals, eigvecs = np.linalg.eig(A)
    n_states = 2 * n_gen
    participation = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            participation[i, j] = abs(eigvecs[j, i]) * abs(eigvecs[j, i])

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