"""
Heffron-Phillips 模型 (K1-K6 常数)
================================

实现单机无穷大系统的 Heffron-Phillips 线性化模型，计算 K1-K6
常数，并构造完整的状态矩阵用于小干扰稳定分析。

数学模型
--------

Heffron-Phillips 模型将同步发电机-无穷大系统在工作点附近线性化，
得到如下形式的 Phillips-Heffron 框图（Kundur Figure 12.12）：

    ΔPm ─→[+]─→[ 1/(Ms+D) ]─→ Δω ─→[ ωs/s ]─→ Δδ
            ↑                                │
            │  K1                            │ K4
            │                                ↓
    ΔPe ←──┘                  ΔEq' ←──[ K3/(1+sK3·Td0') ]←──[+]─ ΔEfd
            ↑                    ↑                          ↑
            │ K2                 │                          │
            └────────────────────┘              [-KA/(1+sTe)]←──[-K5·Δδ - K6·ΔEq']

K-常数公式 (Kundur Eqs. 12.47-12.52):
    K1 = ∂Pe/∂δ  — 同步转矩系数（固定励磁）
    K2 = ∂Pe/∂Eq' — 磁场磁链变化对电磁转矩的影响
    K3 = 阻抗系数 — 外电抗对励磁回路的影响
    K4 = ∂Eq/∂δ  — 功角变化对磁场电势的去磁效应
    K5 = ∂Vt/∂δ  — 功角变化对端电压的影响
    K6 = ∂Vt/∂Eq' — 磁场磁链变化对端电压的影响

状态变量（经典 3 阶，无励磁）:
    [Δδ, Δω, ΔEq']^T

状态变量（4 阶，简化励磁 KA/Te）:
    [Δδ, Δω, ΔEq', ΔEfd]^T

参考:
    - Kundur P. Power System Stability and Control, Ch.12, Sec. 12.3
    - Heffron W.G., Phillips R.A. "Effect of Modern Amplidyne Voltage
      Regulators on Underexcited Operation of Large Turbine Generators"
      AIEE Trans., 1952
    - DeMello F.P., Concordia C. "Concepts of Synchronous Machine
      Stability as Affected by Excitation Control" IEEE Trans. PAS, 1969
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class HeffronPhillipsResult:
    """Heffron-Phillips 线性化分析结果

    Attributes:
        K1: 同步转矩系数 ∂Pe/∂δ（p.u.）
        K2: ∂Pe/∂Eq'（p.u.）
        K3: 励磁回路阻抗系数（无量纲）
        K4: 去磁效应系数 ∂Eq/∂δ（p.u.）
        K5: ∂Vt/∂δ（p.u.）
        K6: ∂Vt/∂Eq'（p.u.）
        eigenvalues: 状态矩阵特征值
        damping_ratios: 各振荡模式的阻尼比
        frequencies: 各振荡模式的振荡频率（Hz）
        state_matrix: 系统状态矩阵 A
        stable: 系统是否小干扰稳定
        states: 状态变量名称列表
        participation_factors: 参与因子矩阵（可选）
        operating_point: 工作点信息字典
    """
    K1: float
    K2: float
    K3: float
    K4: float
    K5: float
    K6: float
    eigenvalues: NDArray[np.complex128]
    damping_ratios: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    state_matrix: NDArray[np.float64]
    stable: bool
    states: List[str] = field(default_factory=list)
    participation_factors: Optional[NDArray[np.float64]] = None
    operating_point: Dict = field(default_factory=dict)


def compute_heffron_phillips_constants(
    E_prime: float,
    V_infinity: float,
    X_total: float,
    Xd: float = 1.8,
    Xd_prime: float = 0.3,
    Xq: float = 1.7,
    Td0_prime: float = 8.0,
    H: float = 5.0,
    D: float = 0.0,
    delta_0: float = None,
    Pm: float = None,
    Ka: float = 0.0,
    Te: float = 0.3,
    exciter_params=None,
    governor_params=None,
    pss_params=None,
    verbose: bool = False,
) -> HeffronPhillipsResult:
    """计算 SMIB 系统的 Heffron-Phillips K1-K6 常数

    线性化单机无穷大系统，计算 K1-K6 常数，并构造状态矩阵
    进行小干扰稳定分析。

    Args:
        E_prime: 暂态电势 Eq'（标幺值）
        V_infinity: 无穷大母线电压（标幺值）
        X_total: 外部等值电抗 Xe（变压器 + 线路，标幺值）
        Xd: d 轴同步电抗（标幺值）
        Xd_prime: d 轴暂态电抗（标幺值）
        Xq: q 轴同步电抗（标幺值）
        Td0_prime: d 轴暂态开路时间常数（秒）
        H: 惯性时间常数（秒）
        D: 阻尼系数
        delta_0: 初始功角（弧度），若不给出则从 Pm 反算
        Pm: 机械功率（标幺值），用于求解 δ₀
        Ka: 励磁系统增益（简化模型，若为 0 则无励磁动态）
        Te: 励磁系统时间常数（秒）
        exciter_params: IEEET1Params 励磁系统参数（如提供，优先使用）
        governor_params: TGOV1Params 调速器参数（可选）
        pss_params: PSS2AParams 稳定器参数（可选）
        verbose: 是否打印详细信息

    Returns:
        HeffronPhillipsResult: 包含 K1-K6 常数和完整分析结果

    Example:
        >>> # Kundur Ex 12.1: SMIB 系统
        >>> result = compute_heffron_phillips_constants(
        ...     E_prime=1.2, V_infinity=1.0, X_total=0.5,
        ...     Xd=1.8, Xd_prime=0.3, Xq=1.7,
        ...     Td0_prime=8.0, H=5.0, D=0.0,
        ...     delta_0=np.radians(30), Ka=50.0, Te=0.3,
        ...     verbose=True)
        >>> print(f"K1={result.K1:.3f}, K2={result.K2:.3f}")
    """
    omega_s = 2 * np.pi * 50  # 同步角频率

    # ================================================================
    # Step 1: 确定初值条件和工作点
    # ================================================================
    Xe = X_total
    X_dSigma = Xe + Xd_prime  # d轴总电抗（E'到V∞）
    X_qSigma = Xe + Xq        # q轴总电抗

    if delta_0 is None and Pm is not None:
        # 从机械功率求解功角
        # Pe = (E'*V∞/X_dΣ)*sin(δ) + V∞²/2*(1/X_qΣ-1/X_dΣ)*sin(2δ) = Pm
        # 使用牛顿法求解
        delta_0 = _solve_delta_from_power(
            E_prime, V_infinity, X_dSigma, X_qSigma, Pm
        )
    elif delta_0 is None:
        delta_0 = np.radians(30)  # 默认

    # 计算工作点电流
    cos_d0 = np.cos(delta_0)
    sin_d0 = np.sin(delta_0)

    Id0 = (E_prime - V_infinity * cos_d0) / X_dSigma
    Iq0 = V_infinity * sin_d0 / X_qSigma

    # 端电压分量
    Vd0 = Xq * Iq0
    Vq0 = V_infinity * cos_d0 + Xe * Id0  # = E' - Xd'*Id0
    Vt0 = np.sqrt(Vd0**2 + Vq0**2)

    # 电气功率
    Pe0 = (E_prime * V_infinity / X_dSigma) * sin_d0 + \
          (V_infinity**2 / 2) * (1.0/X_qSigma - 1.0/X_dSigma) * np.sin(2*delta_0)

    if Pm is None:
        Pm = Pe0  # 稳态时 Pm = Pe

    # ================================================================
    # Step 2: 计算 K1-K6 常数 (Kundur Eqs. 12.47-12.52)
    # ================================================================

    # K1 = ∂Pe/∂δ (固定 Eq')
    K1 = (E_prime * V_infinity / X_dSigma) * cos_d0 + \
         V_infinity**2 * (Xd_prime - Xq) / (X_dSigma * X_qSigma) * np.cos(2*delta_0)

    # K2 = ∂Pe/∂Eq'
    K2 = V_infinity * sin_d0 / X_dSigma

    # K3 = 励磁回路阻抗系数
    K3 = X_dSigma / (Xe + Xd) if (Xe + Xd) > 0 else 0.0

    # K4 = ∂Eq/∂δ (去磁效应)
    K4 = V_infinity * (Xd - Xd_prime) * sin_d0 / X_dSigma

    # K5 = ∂Vt/∂δ
    # 偏导数
    dVd_ddelta = Xq * V_infinity * cos_d0 / X_qSigma
    dVq_ddelta = -V_infinity * sin_d0 * (1.0 + Xe / X_dSigma)

    if Vt0 > 1e-9:
        K5 = (Vd0 * dVd_ddelta + Vq0 * dVq_ddelta) / Vt0
    else:
        K5 = 0.0

    # K6 = ∂Vt/∂Eq'
    dVd_dEq = 0.0  # Iq 不依赖 Eq'
    dVq_dEq = Xe / X_dSigma  # from Vq = Eq' - Xd'*Id

    if Vt0 > 1e-9:
        K6 = (Vd0 * dVd_dEq + Vq0 * dVq_dEq) / Vt0
    else:
        K6 = 0.0

    # ================================================================
    # Step 3: 构造状态矩阵
    # ================================================================
    # 状态变量: [Δδ (rad), Δω (rad/s), ΔEq', ...]
    # 摇摆方程: (2H/ωs)·dΔω/dt = -K1·Δδ - K2·ΔEq' - D·Δω
    #          → dΔω/dt = (ωs/(2H))·(-K1·Δδ - K2·ΔEq' - D·Δω)
    M_coeff = omega_s / (2.0 * H)  # ωs/(2H) — 摇摆方程系数

    use_exciter = Ka > 0 or exciter_params is not None
    use_ieeet1 = exciter_params is not None

    if use_ieeet1 and exciter_params is not None:
        # IEEET1: 6 阶模型
        # 状态: [Δδ, Δω, ΔEq', ΔVR, ΔEfd, ΔRf]
        states = ['Δδ', 'Δω', "ΔEq'", 'ΔVR', 'ΔEfd', 'ΔRf']
        n_states = 6

        A = np.zeros((n_states, n_states))

        # 励磁参数
        exc = exciter_params
        Efd0 = Vt0
        dSE = exc._A_sat * exc._B_sat * np.exp(exc._B_sat * Efd0)
        KE_eff = exc.KE + dSE
        TA_val = exc.TA if exc.TA > 0 else 1e-6
        TE_val = exc.TE if exc.TE > 0 else 1e-6
        TF_val = exc.TF if exc.TF > 0 else 1e-6

        # dΔδ/dt = Δω
        A[0, 1] = 1.0

        # dΔω/dt = (ωs/(2H))·(-K1·Δδ - K2·ΔEq' - D·Δω)
        A[1, 0] = -K1 * M_coeff
        A[1, 1] = -D * M_coeff
        A[1, 2] = -K2 * M_coeff

        # dΔEq'/dt = (1/Td0')(-K4·Δδ - (1/K3)·ΔEq' + ΔEfd)
        A[2, 0] = -K4 / Td0_prime
        A[2, 2] = -1.0 / (K3 * Td0_prime) if K3 > 0 else 0.0
        A[2, 4] = 1.0 / Td0_prime

        # dΔVR/dt = (-KA/TA)·(K5·Δδ + K6·ΔEq') - ΔVR/TA
        A[3, 0] = -exc.KA * K5 / TA_val
        A[3, 2] = -exc.KA * K6 / TA_val
        A[3, 3] = -1.0 / TA_val

        # dΔEfd/dt = ΔVR/TE - KE_eff·ΔEfd/TE
        A[4, 3] = 1.0 / TE_val
        A[4, 4] = -KE_eff / TE_val

        # dΔRf/dt
        A[5, 3] = exc.KF / (TF_val * TE_val)
        A[5, 4] = -exc.KF * KE_eff / (TF_val * TE_val)
        A[5, 5] = -1.0 / TF_val

    elif use_exciter:
        # 简化励磁: 4 阶模型
        # 状态: [Δδ, Δω, ΔEq', ΔEfd]
        states = ['Δδ', 'Δω', "ΔEq'", 'ΔEfd']
        n_states = 4

        A = np.zeros((n_states, n_states))

        # dΔδ/dt = Δω
        A[0, 1] = 1.0

        # dΔω/dt = (ωs/(2H))·(-K1·Δδ - K2·ΔEq' - D·Δω)
        A[1, 0] = -K1 * M_coeff
        A[1, 1] = -D * M_coeff
        A[1, 2] = -K2 * M_coeff

        # dΔEq'/dt
        A[2, 0] = -K4 / Td0_prime
        A[2, 2] = -1.0 / (K3 * Td0_prime) if K3 > 0 else 0.0
        A[2, 3] = 1.0 / Td0_prime

        # dΔEfd/dt
        A[3, 0] = -Ka * K5 / Te
        A[3, 2] = -Ka * K6 / Te
        A[3, 3] = -1.0 / Te

    else:
        # 经典 3 阶（磁链衰减，无励磁动态）
        # 状态: [Δδ, Δω, ΔEq']
        states = ['Δδ', 'Δω', "ΔEq'"]
        n_states = 3

        A = np.zeros((n_states, n_states))

        # dΔδ/dt = Δω
        A[0, 1] = 1.0

        # dΔω/dt = (ωs/(2H))·(-K1·Δδ - K2·ΔEq' - D·Δω)
        A[1, 0] = -K1 * M_coeff
        A[1, 1] = -D * M_coeff
        A[1, 2] = -K2 * M_coeff

        # dΔEq'/dt
        A[2, 0] = -K4 / Td0_prime
        A[2, 2] = -1.0 / (K3 * Td0_prime) if K3 > 0 else 0.0

    # ================================================================
    # Step 4: 特征值分析
    # ================================================================
    eigenvalues = np.linalg.eigvals(A)
    stable = all(e.real < 0 for e in eigenvalues)

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

    # ================================================================
    # Step 5: 输出
    # ================================================================
    operating_point = {
        'delta_0_rad': delta_0,
        'delta_0_deg': np.degrees(delta_0),
        'Id0': Id0, 'Iq0': Iq0,
        'Vd0': Vd0, 'Vq0': Vq0, 'Vt0': Vt0,
        'Pe0': Pe0, 'Pm': Pm,
        'E_prime': E_prime,
    }

    if verbose:
        print("=" * 55)
        print("  Heffron-Phillips K1-K6 Analysis")
        print("=" * 55)
        print(f"\n  Operating point: delta0 = {np.degrees(delta_0):.2f} deg, "
              f"Pe = {Pe0:.4f} pu, Vt = {Vt0:.4f} pu")
        print(f"\n  K-constants:")
        print(f"    K1 = {K1:+.4f}  (sync torque coeff)")
        print(f"    K2 = {K2:+.4f}  (dPe/dEq')")
        print(f"    K3 = {K3:+.4f}  (field impedance factor)")
        print(f"    K4 = {K4:+.4f}  (demagnetizing effect)")
        print(f"    K5 = {K5:+.4f}  (dVt/ddelta)")
        print(f"    K6 = {K6:+.4f}  (dVt/dEq')")
        print(f"\n  State matrix ({n_states}x{n_states}):")
        print(f"    Exciter: {'IEEET1' if use_ieeet1 else 'KA/Te' if use_exciter else 'None (3rd order)'}")
        print(f"    System {'STABLE' if stable else 'UNSTABLE'}")
        print(f"\n  Eigenvalues (oscillatory modes only):")
        osc_found = False
        for i, ev in enumerate(eigenvalues):
            if abs(ev.imag) > 1e-6:
                osc_found = True
                print(f"    lambda = {ev.real:+.4f} +/- j{abs(ev.imag):.4f}  "
                      f"(zeta={damping_ratios[i]:+.4f}, f={frequencies[i]:.2f} Hz)")
        if not osc_found:
            for i, ev in enumerate(eigenvalues):
                print(f"    lambda = {ev.real:+.4f}  (non-oscillatory)")
        print("=" * 55)

    return HeffronPhillipsResult(
        K1=K1, K2=K2, K3=K3, K4=K4, K5=K5, K6=K6,
        eigenvalues=eigenvalues,
        damping_ratios=np.array(damping_ratios),
        frequencies=np.array(frequencies),
        state_matrix=A,
        stable=stable,
        states=states,
        participation_factors=participation,
        operating_point=operating_point,
    )


def _solve_delta_from_power(
    E_prime: float,
    V_infinity: float,
    X_dSigma: float,
    X_qSigma: float,
    P_target: float,
    delta_guess: float = None
) -> float:
    """从目标有功功率反解功角 δ（牛顿法）

    Pe(δ) = (E'*V∞/X_dΣ)*sin(δ) + V∞²/2*(1/X_qΣ-1/X_dΣ)*sin(2δ)
    """
    if delta_guess is None:
        # 初值：忽略凸极效应的线性近似
        delta_guess = np.arcsin(np.clip(
            P_target * X_dSigma / (E_prime * V_infinity), -1.0, 1.0
        ))

    delta = delta_guess
    for _ in range(20):
        sin_d = np.sin(delta)
        cos_d = np.cos(delta)
        sin_2d = np.sin(2 * delta)
        cos_2d = np.cos(2 * delta)

        a = E_prime * V_infinity / X_dSigma
        b = V_infinity**2 / 2 * (1.0 / X_qSigma - 1.0 / X_dSigma)

        Pe = a * sin_d + b * sin_2d
        dPe = a * cos_d + 2 * b * cos_2d

        if abs(dPe) < 1e-12:
            break

        f = Pe - P_target
        delta = delta - f / dPe

        if abs(f) < 1e-10:
            break

    return delta


def sweep_k_constants(
    E_prime: float,
    V_infinity: float,
    X_total: float,
    delta_range: Tuple[float, float] = (5, 85),
    n_points: int = 50,
    Xd: float = 1.8,
    Xd_prime: float = 0.3,
    Xq: float = 1.7,
    Td0_prime: float = 8.0,
) -> Dict[str, NDArray]:
    """扫描不同功角下的 K1-K6 常数

    绘制 K1-K6 随功角变化的曲线，直观展示运行点对系统阻尼的影响。

    Args:
        E_prime: 暂态电势
        V_infinity: 无穷大母线电压
        X_total: 外部等值电抗
        delta_range: 功角范围（度）
        n_points: 扫描点数
        Xd, Xd_prime, Xq, Td0_prime: 发电机参数

    Returns:
        Dict，包含 'delta_deg', 'K1'-'K6' 各数组

    Example:
        >>> curves = sweep_k_constants(1.2, 1.0, 0.5, (5, 85), 50)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(curves['delta_deg'], curves['K5'], label='K5')
    """
    delta_vals = np.radians(np.linspace(delta_range[0], delta_range[1], n_points))
    K1_arr = np.zeros(n_points)
    K2_arr = np.zeros(n_points)
    K3_arr = np.zeros(n_points)
    K4_arr = np.zeros(n_points)
    K5_arr = np.zeros(n_points)
    K6_arr = np.zeros(n_points)

    for k, delta_0 in enumerate(delta_vals):
        result = compute_heffron_phillips_constants(
            E_prime=E_prime, V_infinity=V_infinity, X_total=X_total,
            Xd=Xd, Xd_prime=Xd_prime, Xq=Xq, Td0_prime=Td0_prime,
            delta_0=delta_0, H=5.0, D=0.0,
            verbose=False,
        )
        K1_arr[k] = result.K1
        K2_arr[k] = result.K2
        K3_arr[k] = result.K3
        K4_arr[k] = result.K4
        K5_arr[k] = result.K5
        K6_arr[k] = result.K6

    return {
        'delta_deg': np.linspace(delta_range[0], delta_range[1], n_points),
        'delta_rad': delta_vals,
        'K1': K1_arr, 'K2': K2_arr, 'K3': K3_arr,
        'K4': K4_arr, 'K5': K5_arr, 'K6': K6_arr,
    }


__all__ = [
    "HeffronPhillipsResult",
    "compute_heffron_phillips_constants",
    "sweep_k_constants",
]
