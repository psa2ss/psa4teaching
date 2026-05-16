"""
暂态稳定时域仿真 (Transient Stability Simulation)
=================================================

通过数值积分求解发电机转子运动方程，判断系统暂态稳定性。

数学模型
--------

### 经典模型（Single Machine Infinite Bus - Classic Model）

发电机采用二阶模型：
    dδ/dt = (ω - 1) × ωs = Δω × ωs      [rad/s]
    dω/dt = (Pm - Pe - D×Δω) / (2H)        [p.u.]

其中：
    δ —— 发电机转子功角（弧度）
    ω —— 发电机转速（标幺值，同步速ωs=1）
    Pm —— 机械功率（标幺值）
    Pe —— 电磁功率（标幺值）
    H —— 惯性时间常数（秒）
    D —— 阻尼系数

电磁功率（单机无穷大）：
    Pe = (E' × V∞ / XΣ) × sin(δ)

其中：
    E' —— 暂态电势（经典模型中恒定）
    V∞ —— 无穷大母线电压
    XΣ —— 等值电抗

### 详细模型（Detailed Model）

发电机采用三阶或更高阶模型，包含：
    - 转子运动方程（δ, ω）
    - d轴暂态方程（Eq'）
    - 励磁系统方程（Efd）
    - 调速系统方程（Pm）

d轴暂态方程：
    dEq'/dt = (Efd - Eq') / Td0'

励磁系统（简化一阶）：
    dEfd/dt = -(Efd - Efd0) / Te + Ka × (Vref - Vt) / Te

调速系统（简化一阶）：
    dPm/dt = -(Pm - Pm0) / Tg

参考教材：李光琦《电力系统暂态分析》第三、四章
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class TransientStabilityResult:
    """暂态稳定仿真结果

    Attributes:
        converged: 仿真是否正常完成
        stable: 系统是否保持暂态稳定
        time: 时间向量（秒）
        delta: 功角轨迹（弧度或度）
        omega: 转速轨迹（标幺值）
        Pe: 电磁功率轨迹
        Pm: 机械功率轨迹
        max_delta: 最大功角
        critical_clearing_time: 临界清除时间估计（秒），可选
    """
    converged: bool
    stable: bool
    time: NDArray[np.float64]
    delta: NDArray[np.float64]
    omega: NDArray[np.float64]
    Pe: NDArray[np.float64]
    Pm: NDArray[np.float64]
    max_delta: float
    critical_clearing_time: Optional[float] = None
    Eq_prime: Optional[NDArray[np.float64]] = None
    Efd: Optional[NDArray[np.float64]] = None
    Vt: Optional[NDArray[np.float64]] = None


def simulate_single_machine_infinite_bus_classic(
    E_prime: float,
    V_infinity: float,
    X_total: float,
    H: float,
    D: float,
    Pm: float,
    delta_0: float,
    fault_time: float,
    fault_clearing_time: float,
    X_total_fault: Optional[float] = None,
    X_total_post: Optional[float] = None,
    t_end: float = 5.0,
    dt: float = 0.005,
    method: str = "rk4",
    stability_limit: float = 150.0,
    verbose: bool = False
) -> TransientStabilityResult:
    """单机无穷大系统暂态稳定仿真 - 经典模型

    Args:
        E_prime: 暂态电势 E'（标幺值）
        V_infinity: 无穷大母线电压（标幺值）
        X_total: 正常运行时的等值电抗（标幺值）
        H: 惯性时间常数（秒）
        D: 阻尼系数（标幺值）
        Pm: 机械功率（标幺值）
        delta_0: 初始功角（弧度）
        fault_time: 故障发生时刻（秒）
        fault_clearing_time: 故障清除时刻（秒）
        X_total_fault: 故障期间等值电抗（标幺值，默认X_total的2倍）
        X_total_post: 故障后等值电抗（标幺值，默认与X_total相同）
        t_end: 仿真结束时间（秒）
        dt: 仿真步长（秒）
        method: 积分方法，"rk4"（龙格-库塔）或 "euler"（改进欧拉）
        stability_limit: 稳定性判据，功角超过此值视为失稳（度）
        verbose: 是否打印详细信息

    Returns:
        TransientStabilityResult: 暂态稳定仿真结果

    Note:
        仿真过程：
        1. 故障前：使用X_total计算Pe
        2. 故障期间（t_fault ≤ t ≤ t_clear）：使用X_total_fault计算Pe
        3. 故障后：使用X_total_post计算Pe（切除故障线路后网络变化）
        4. 判断是否稳定：功角δ是否持续增长或超过稳定性限制

    Example:
        >>> result = simulate_single_machine_infinite_bus_classic(
        ...     E_prime=1.2, V_infinity=1.0, X_total=0.5,
        ...     H=5.0, D=0.0, Pm=0.8, delta_0=np.radians(30),
        ...     fault_time=0.0, fault_clearing_time=0.15
        ... )
        >>> print(f"系统{'稳定' if result.stable else '不稳定'}")
    """
    if X_total_fault is None:
        X_total_fault = X_total * 2.0
    if X_total_post is None:
        X_total_post = X_total

    # 功率系数（三段工况）
    P_max_pre = E_prime * V_infinity / X_total
    P_max_fault = E_prime * V_infinity / X_total_fault
    P_max_post = E_prime * V_infinity / X_total_post

    # 初始条件
    delta = delta_0
    omega = 1.0  # 标幺值，同步速

    # 数组初始化
    n_steps = int(t_end / dt) + 1
    time_arr = np.zeros(n_steps)
    delta_arr = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)
    Pe_arr = np.zeros(n_steps)
    Pm_arr = np.zeros(n_steps)

    delta_arr[0] = delta
    omega_arr[0] = omega

    stable = True
    converged = True

    def derivatives(delta_val, omega_val, t):
        """计算导数"""
        # 确定当前使用哪组参数
        if t < fault_time:
            P_max = P_max_pre
        elif t < fault_clearing_time:
            P_max = P_max_fault
        else:
            P_max = P_max_post

        Pe = P_max * np.sin(delta_val)
        d_delta = (omega_val - 1.0)  # dδ/dt = Δω × ωs（标幺值）
        d_omega = (Pm - Pe - D * (omega_val - 1.0)) / (2 * H)

        return d_delta, d_omega, Pe

    for i in range(n_steps - 1):
        t = i * dt
        time_arr[i] = t

        if method == "rk4":
            # 四阶龙格-库塔法
            k1_d, k1_w, Pe = derivatives(delta, omega, t)

            k2_d, k2_w, _ = derivatives(
                delta + 0.5 * dt * k1_d,
                omega + 0.5 * dt * k1_w,
                t + 0.5 * dt
            )

            k3_d, k3_w, _ = derivatives(
                delta + 0.5 * dt * k2_d,
                omega + 0.5 * dt * k2_w,
                t + 0.5 * dt
            )

            k4_d, k4_w, _ = derivatives(
                delta + dt * k3_d,
                omega + dt * k3_w,
                t + dt
            )

            delta += dt / 6 * (k1_d + 2*k2_d + 2*k3_d + k4_d)
            omega += dt / 6 * (k1_w + 2*k2_w + 2*k3_w + k4_w)

        elif method == "euler":
            # 改进欧拉法
            d_delta, d_omega, Pe = derivatives(delta, omega, t)

            delta_pred = delta + dt * d_delta
            omega_pred = omega + dt * d_omega

            d_delta_pred, d_omega_pred, _ = derivatives(delta_pred, omega_pred, t + dt)

            delta += dt / 2 * (d_delta + d_delta_pred)
            omega += dt / 2 * (d_omega + d_omega_pred)

        else:
            raise ValueError(f"未知的积分方法: {method}")

        delta_arr[i] = delta
        omega_arr[i] = omega
        Pe_arr[i] = Pe
        Pm_arr[i] = Pm

        # 稳定性检查
        if np.degrees(delta) > stability_limit:
            stable = False
            converged = True
            time_arr[i + 1] = (i + 1) * dt
            delta_arr = delta_arr[:i + 2]
            omega_arr = omega_arr[:i + 2]
            Pe_arr = Pe_arr[:i + 2]
            Pm_arr = Pm_arr[:i + 2]
            time_arr = time_arr[:i + 2]
            break

    time_arr[-1] = (n_steps - 1) * dt
    delta_arr[-1] = delta
    omega_arr[-1] = omega

    if verbose:
        print(f"仿真完成: {'稳定' if stable else '不稳定'}")
        print(f"最大功角: {np.max(np.degrees(delta_arr)):.2f}°")

    return TransientStabilityResult(
        converged=converged,
        stable=stable,
        time=time_arr,
        delta=np.degrees(delta_arr),
        omega=omega_arr,
        Pe=Pe_arr,
        Pm=Pm_arr,
        max_delta=np.max(np.degrees(delta_arr))
    )


def simulate_single_machine_infinite_bus_detailed(
    E_prime_0: float,
    V_infinity: float,
    X_total: float,
    Xd: float,
    Xd_prime: float,
    Xq: float,
    Td0_prime: float,
    H: float,
    D: float,
    Pm_0: float,
    delta_0: float,
    Efd_0: float,
    # 励磁系统参数
    Ka: float = 50.0,
    Ta: float = 0.05,
    Te: float = 0.3,
    Efd_min: float = 0.0,
    Efd_max: float = 7.0,
    # 调速系统参数
    Tg: float = 0.5,
    Pm_min: float = 0.0,
    Pm_max: float = 1.5,
    # 仿真参数
    fault_time: float = 0.0,
    fault_clearing_time: float = 0.15,
    X_total_fault: Optional[float] = None,
    t_end: float = 5.0,
    dt: float = 0.005,
    stability_limit: float = 150.0
) -> TransientStabilityResult:
    """单机无穷大系统暂态稳定仿真 - 详细模型

    Args:
        E_prime_0: 初始暂态电势
        V_infinity: 无穷大母线电压
        X_total: 等值电抗
        Xd: 同步电抗
        Xd_prime: 暂态电抗
        Xq: q轴同步电抗
        Td0_prime: d轴暂态开路时间常数（秒）
        H: 惯性时间常数
        D: 阻尼系数
        Pm_0: 初始机械功率
        delta_0: 初始功角（弧度）
        Efd_0: 初始励磁电压
        Ka: 励磁系统放大倍数
        Ta: 励磁调节器时间常数
        Te: 励磁机时间常数
        Efd_min/Efd_max: 励磁电压限幅
        Tg: 调速器时间常数
        Pm_min/Pm_max: 机械功率限幅
        fault_time: 故障发生时刻
        fault_clearing_time: 故障清除时刻
        X_total_fault: 故障期间等值电抗
        t_end: 仿真结束时间
        dt: 仿真步长
        stability_limit: 稳定性判断功角限制（度）

    Returns:
        TransientStabilityResult: 详细模型仿真结果

    Note:
        详细模型状态方程（四阶）：
        1. dδ/dt = ω - 1
        2. dω/dt = (Pm - Pe - D(ω-1)) / (2H)
        3. dEq'/dt = (Efd - Eq' - (Xd-Xd')×Id) / Td0'
        4. dEfd/dt = (-(Efd-Efd0)/Te + Ka(Ta))/(Ta×Te) × (Vref-Vt)

        其中：
        Id = (Eq' - V∞cos(δ)) / (Xd'+X_total)
        Iq = V∞sin(δ) / (Xq+X_total)
        Pe = Eq'×Iq + (Xd'-Xq)×Id×Iq
        Vt = √((V∞+X_total×Iq)² + (X_total×Id)²)
    """
    if X_total_fault is None:
        X_total_fault = X_total * 2.0

    n_steps = int(t_end / dt) + 1
    time_arr = np.zeros(n_steps)
    delta_arr = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)
    Pe_arr = np.zeros(n_steps)
    Pm_arr = np.zeros(n_steps)
    Eq_prime_arr = np.zeros(n_steps)
    Efd_arr = np.zeros(n_steps)
    Vt_arr = np.zeros(n_steps)

    # 初始状态
    delta = delta_0
    omega = 1.0
    Eq_prime = E_prime_0
    Efd = Efd_0
    Pm = Pm_0

    # 初始电压参考值（使系统在初始点平衡）
    # 计算初始电流
    Id_0 = (Eq_prime - V_infinity * np.cos(delta)) / (Xd_prime + X_total)
    Iq_0 = V_infinity * np.sin(delta) / (Xq + X_total)
    Vt_0 = np.sqrt((V_infinity + X_total * Iq_0)**2 + (X_total * Id_0)**2)
    Vref = Vt_0  # 初始电压参考值

    Eq_prime_arr[0] = Eq_prime
    Efd_arr[0] = Efd
    Vt_arr[0] = Vt_0

    stable = True

    def compute_power(delta_val, Eq_val, X_total_val):
        """计算电磁功率"""
        Id = (Eq_val - V_infinity * np.cos(delta_val)) / (Xd_prime + X_total_val)
        Iq = V_infinity * np.sin(delta_val) / (Xq + X_total_val)
        Pe = Eq_val * Iq + (Xd_prime - Xq) * Id * Iq
        Vt = np.sqrt((V_infinity + X_total_val * Iq)**2 +
                      (X_total_val * Id)**2)
        return Pe, Id, Iq, Vt

    def derivatives(state, t):
        """计算状态导数"""
        delta_v, omega_v, Eq_v, Efd_v, Pm_v = state

        # 确定等值电抗
        if t < fault_time:
            X_t = X_total
        elif t < fault_clearing_time:
            X_t = X_total_fault
        else:
            X_t = X_total

        # 计算电流和功率
        Pe, Id, Iq, Vt = compute_power(delta_v, Eq_v, X_t)

        # 转子运动方程
        d_delta = omega_v - 1.0
        d_omega = (Pm_v - Pe - D * (omega_v - 1.0)) / (2 * H)

        # d轴暂态方程
        dEq = (Efd_v - Eq_v - (Xd - Xd_prime) * Id) / Td0_prime

        # 励磁系统方程
        dEfd = (Ka * (Vref - Vt) - Efd_v) / Te

        # 励磁限幅
        dEfd = np.clip(dEfd, Efd_min, Efd_max)
        Efd_v = np.clip(Efd_v, Efd_min, Efd_max)

        # 调速系统方程
        dPm = -(Pm_v - Pm_0) / Tg

        return np.array([d_delta, d_omega, dEq, dEfd, dPm]), Pe, Vt

    for i in range(n_steps - 1):
        t = i * dt
        time_arr[i] = t

        state = np.array([delta, omega, Eq_prime, Efd, Pm])

        # 四阶龙格-库塔法
        k1, Pe, Vt = derivatives(state, t)
        k2, _, _ = derivatives(state + 0.5*dt*k1, t + 0.5*dt)
        k3, _, _ = derivatives(state + 0.5*dt*k2, t + 0.5*dt)
        k4, _, _ = derivatives(state + dt*k3, t + dt)

        state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        delta, omega, Eq_prime, Efd, Pm = state

        delta_arr[i] = np.degrees(delta)
        omega_arr[i] = omega
        Pe_arr[i] = Pe
        Pm_arr[i] = Pm
        Eq_prime_arr[i] = Eq_prime
        Efd_arr[i] = Efd
        Vt_arr[i] = Vt

        if np.degrees(delta) > stability_limit:
            stable = False
            delta_arr = delta_arr[:i+2]
            omega_arr = omega_arr[:i+2]
            Pe_arr = Pe_arr[:i+2]
            Pm_arr = Pm_arr[:i+2]
            Eq_prime_arr = Eq_prime_arr[:i+2]
            Efd_arr = Efd_arr[:i+2]
            Vt_arr = Vt_arr[:i+2]
            time_arr = time_arr[:i+2]
            break

    time_arr[-1] = t_end
    delta_arr[-1] = np.degrees(delta)
    omega_arr[-1] = omega

    return TransientStabilityResult(
        converged=True,
        stable=stable,
        time=time_arr,
        delta=delta_arr,
        omega=omega_arr,
        Pe=Pe_arr,
        Pm=Pm_arr,
        max_delta=np.max(np.degrees(delta_arr)),
        Eq_prime=Eq_prime_arr,
        Efd=Efd_arr,
        Vt=Vt_arr
    )


def simulate_multi_machine_classic(
    E_primes: List[float],
    H_list: List[float],
    D_list: List[float],
    Pm_list: List[float],
    delta_0_list: List[float],
    Ybus_reduced: NDArray[np.complex128],
    fault_time: float = 0.0,
    fault_clearing_time: float = 0.15,
    Ybus_fault: Optional[NDArray[np.complex128]] = None,
    t_end: float = 5.0,
    dt: float = 0.005,
    stability_limit: float = 150.0,
    verbose: bool = False
) -> TransientStabilityResult:
    """多机系统暂态稳定仿真 - 经典模型

    Args:
        E_primes: 各发电机暂态电势列表
        H_list: 各发电机惯性时间常数列表
        D_list: 各发电机阻尼系数列表
        Pm_list: 各发电机机械功率列表
        delta_0_list: 各发电机初始功角列表（弧度）
        Ybus_reduced: 导纳矩阵（只含发电机节点，已消去负荷节点）
        fault_time: 故障发生时刻
        fault_clearing_time: 故障清除时刻
        Ybus_fault: 故障期间导纳矩阵（可选，默认增大对角元模拟故障）
        t_end: 仿真结束时间
        dt: 仿真步长
        stability_limit: 稳定性判断功角限制（度）
        verbose: 是否打印详细信息

    Returns:
        TransientStabilityResult: 多机系统暂态稳定仿真结果

    Note:
        多机系统经典模型的运动方程：
        Mi × d²δi/dt² = Pmi - Pei - Di × dδi/dt

        其中电磁功率：
        Pei = Ei' × Σ Ej' × |Yij| × sin(δi - δj - αij)

        αij = arctan(Gij/Bij) 为导纳矩阵元素的阻抗角

        参考：李光琦《电力系统暂态分析》第四章
    """
    n_gen = len(E_primes)
    Y_normal = Ybus_reduced
    G_normal = Y_normal.real
    B_normal = Y_normal.imag

    if Ybus_fault is None:
        # 默认故障处理：增大各节点自导纳（模拟三相短路）
        Y_fault = Y_normal.copy()
        for i in range(n_gen):
            Y_fault[i, i] += 10.0 + 10.0j
        G_fault = Y_fault.real
        B_fault = Y_fault.imag
    else:
        G_fault = Ybus_fault.real
        B_fault = Ybus_fault.imag

    n_steps = int(t_end / dt) + 1
    time_arr = np.zeros(n_steps)
    delta_arr = np.zeros((n_steps, n_gen))
    omega_arr = np.zeros((n_steps, n_gen))
    Pe_arr = np.zeros((n_steps, n_gen))
    Pm_arr = np.zeros((n_steps, n_gen))

    # 初始状态
    deltas = np.array(delta_0_list, dtype=float)
    omegas = np.ones(n_gen)

    delta_arr[0] = np.degrees(deltas)
    omega_arr[0] = omegas

    stable = True

    def compute_electrical_power(deltas_val, G, B):
        """计算各发电机电磁功率"""
        Pe = np.zeros(n_gen)
        for i in range(n_gen):
            for j in range(n_gen):
                alpha_ij = np.arctan2(G[i, j], B[i, j])
                Y_mag = np.sqrt(G[i, j]**2 + B[i, j]**2)
                Pe[i] += E_primes[i] * E_primes[j] * Y_mag * np.sin(
                    deltas_val[i] - deltas_val[j] - alpha_ij
                )
        return Pe

    for i in range(n_steps - 1):
        t = i * dt
        time_arr[i] = t

        # 确定使用的导纳矩阵
        if t < fault_time:
            G, B = G_normal, B_normal
        elif t < fault_clearing_time:
            G, B = G_fault, B_fault
        else:
            G, B = G_normal, B_normal

        Pe = compute_electrical_power(deltas, G, B)

        # RK4积分
        def deriv(d, w):
            pe = compute_electrical_power(d, G, B)
            dd = w - 1.0
            dw = np.array([(Pm_list[k] - pe[k] - D_list[k]*(w[k]-1.0)) / (2*H_list[k])
                          for k in range(n_gen)])
            return dd, dw

        k1d, k1w = deriv(deltas, omegas)
        k2d, k2w = deriv(deltas + 0.5*dt*k1d, omegas + 0.5*dt*k1w)
        k3d, k3w = deriv(deltas + 0.5*dt*k2d, omegas + 0.5*dt*k2w)
        k4d, k4w = deriv(deltas + dt*k3d, omegas + dt*k3w)

        deltas = deltas + dt/6*(k1d + 2*k2d + 2*k3d + k4d)
        omegas = omegas + dt/6*(k1w + 2*k2w + 2*k3w + k4w)

        delta_arr[i] = np.degrees(deltas)
        omega_arr[i] = omegas
        Pe_arr[i] = Pe
        Pm_arr[i] = np.array(Pm_list)

        if np.max(np.degrees(deltas)) > stability_limit:
            stable = False
            delta_arr = delta_arr[:i+2]
            omega_arr = omega_arr[:i+2]
            Pe_arr = Pe_arr[:i+2]
            Pm_arr = Pm_arr[:i+2]
            time_arr = time_arr[:i+2]
            break

    return TransientStabilityResult(
        converged=True,
        stable=stable,
        time=time_arr,
        delta=delta_arr,
        omega=omega_arr,
        Pe=Pe_arr,
        Pm=Pm_arr,
        max_delta=np.max(delta_arr)
    )