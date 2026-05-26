"""
ENTSO-E SMIB 标准测试系统 - 动态仿真
=============================================

基于 ENTSO-E SG SPD 报告《Documentation on Controller Tests in Test Grid Configurations》
实现的单机无穷大系统（SMIB）动态仿真。

修正（v2）：
1. 正确的 SMIB 网络方程：Vt = f(Eq', δ, X_ext)，不再是常量
2. 正确的状态变量：δ, ω, Eq' 为发电机状态；Efd/PMECH 为控制器输出
3. 正确的初始条件：空载时 Efd=1.0, PMECH=0
4. 三测试案例使用不同的网络拓扑

参考：
- ENTSO-E SG SPD Report (2013-11-26)
- Kundur《Power System Stability and Control》
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from psa4teaching.models import (
    Bus, BusType, Line, Transformer, Generator, Load, LoadModel
)
from psa4teaching.models.governor import TGOV1Params
from psa4teaching.models.exciter import SEXSParams
from psa4teaching.models.pss import PSS2AParams


# ============================================================
# 系统构建
# ============================================================

def build_entsoe_smib_system() -> Dict:
    """构建 ENTSO-E SMIB 标准测试系统

    NGEN(21kV) ─ NTLV(21kV) ─ NTHV(380kV) ─ NGRID(380kV)
        │                        │
       GEN                      GRID(无穷大, Sk=2500MVA)
                                 GRIDL(恒阻抗负荷)
    """
    buses = [
        Bus(1, "NGEN", BusType.PV, V_specified=1.0, base_kv=21.0),
        Bus(2, "NTLV", BusType.PQ, base_kv=21.0),
        Bus(3, "NTHV", BusType.PQ, base_kv=380.0),
        Bus(4, "NGRID", BusType.SLACK, V_specified=1.05, base_kv=380.0),
    ]

    line1 = Line(from_bus=1, to_bus=2, R=0.001, X=0.01, B=0.0, name="L1_NGEN_NTLV")
    line2 = Line(from_bus=3, to_bus=4, R=0.02, X=0.199, B=0.0, name="L2_NTHV_NGRID")

    transformer = Transformer(
        from_bus=2, to_bus=3,
        RT=0.0, XT=0.16,
        name="T1_GEN_GRID", k=419.0/21.0
    )

    lines = [line1, line2]

    generator = Generator(
        bus=1, name="GEN1",
        Sb=500.0, Vb=21.0,
        Xd=2.0, Xd_prime=0.35, Xd_doubleprime=0.25,
        Xq=1.8, Xq_prime=0.5, Xq_doubleprime=0.3,
        Td0_prime=0.9, Td0_doubleprime=0.03,
        Tq0_prime=0.6, Tq0_doubleprime=0.05,
        H=4.0, D=0.0, model_type='detail',
    )

    governor = TGOV1Params(
        R=0.05, T1=0.5, T2=3.0, T3=10.0,
        Dt=0.0, VMIN=0.0, VMAX=1.0, P_base_MW=475.0
    )

    exciter = SEXSParams(
        K=200.0, TA=3.0, TB=10.0, TE=0.05,
        EMIN=0.0, EMAX=4.0
    )

    pss = PSS2AParams(
        KS1=10.0, KS2=0.1564, KS3=1.0,
        TW1=2.0, TW2=2.0, TW3=2.0, TW4=0.0,
        T1=0.25, T2=0.03, T3=0.15, T4=0.015,
        T6=0.0, T7=2.0, T8=0.5, T9=0.1,
        VSTMIN=-0.1, VSTMAX=0.1,
        N=0, M=0, IC1=1, IC2=3
    )

    # 恒阻抗负荷 GRIDL
    load = Load(
        bus=4, P0=475.0/500.0, Q0=76.0/500.0,
        model_type=LoadModel.CONSTANT_IMPEDANCE, name="GRIDL"
    )

    return {
        'buses': buses,
        'lines': lines,
        'transformers': [transformer],
        'generator': generator,
        'governor': governor,
        'exciter': exciter,
        'pss': pss,
        'load': load,
        'S_base': 500.0,
        'f_base': 50.0,
    }


# ============================================================
# 网络方程
# ============================================================

def compute_smib_voltages(delta: float, Eq_prime: float, gen: Generator,
                          X_ext: float = 0.0, fault_active: bool = False,
                          V_inf_mag: float = 1.0, is_no_load: bool = False) -> Tuple[float, float, float]:
    """SMIB 网络方程：从发电机内部状态计算端电压和电磁功率

    单机无穷大（SMIB）等效电路：
        Eq'∠δ — jXd' — Vt∠θ — jX_ext — V_inf∠0°

    电流：I = (Eq'∠δ - V_inf∠0°) / j(Xd' + X_ext)
    端电压：Vt = Eq'∠δ - jXd' * I
    电磁功率：Pe = (Eq' * V_inf / X_total) * sin(δ)

    Args:
        delta: 转子角（弧度）
        Eq_prime: 暂态电势（标幺值）
        gen: Generator 对象
        X_ext: 外部等效电抗（标幺值）
        fault_active: 是否发生故障
        V_inf_mag: 无穷大母线电压（标幺值）
        is_no_load: 空载模式

    Returns:
        (Vt_mag, Pe, I_mag): 端电压幅值，电磁功率，电流幅值
    """
    Xd_prime = gen.Xd_prime

    if is_no_load:
        # 空载：无电流，Vt = Eq'
        return abs(Eq_prime), 0.0, 0.0

    X_total = Xd_prime + max(X_ext, 1e-10)

    if fault_active:
        # 故障时：Vt 降为接近 0
        # 实际计算：I = Eq'/jXd_prime (短路到发电机端)
        # Pe = 0 (无功率输出)
        I_fault = Eq_prime / Xd_prime
        return 0.05, 0.0, I_fault

    # 正常工况
    Vt_mag = np.sqrt(
        (V_inf_mag * Xd_prime / X_total * np.cos(delta))**2 +
        (Eq_prime - V_inf_mag * Xd_prime / X_total * np.cos(delta))**2
    )
    Pe = (Eq_prime * V_inf_mag / X_total) * np.sin(delta)

    return Vt_mag, Pe, 0.0


def get_external_reactance(system: Dict, fault_active: bool = False) -> float:
    """计算 SMIB 系统外部等值电抗

    正常：X_ext = X_line1 + X_T1 + X_line2
    故障：X_ext = X_line1 + X_T1 (到 NTHV 故障点)

    Returns:
        外部电抗（标幺值）
    """
    lines = system['lines']
    tx = system['transformers'][0]
    X_normal = lines[0].X + tx.XT + lines[1].X

    if fault_active:
        return lines[0].X + tx.XT
    return X_normal


# ============================================================
# 状态变量
# ============================================================

@dataclass
class SMIBState:
    """SMIB 系统完整状态（发电机3维 + 控制器12维）

    发电机状态（3维）:
        delta: 转子功角（弧度）
        omega: 转速（标幺值）
        Eq_prime: d轴暂态电势（标幺值）

    控制器内部状态:
        exc_state[3]: 励磁系统（SEXS，3阶）
        gov_state[3]: 调速器（TGOV1，3阶）
        pss_state[6]: 稳定器（PSS2A，6阶）

    控制器输出（工作变量）:
        Efd: 励磁电压（标幺值）
        PMECH: 机械功率（标幺值）
    """
    delta: float = 0.0
    omega: float = 1.0
    Eq_prime: float = 1.0
    exc_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gov_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pss_state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    Efd: float = 1.0
    PMECH: float = 0.0


# ============================================================
# 状态导数与 RK4
# ============================================================

def state_derivatives(state: SMIBState, system: Dict,
                       V_ref: float, dt: float,
                       X_ext: float, fault_active: bool,
                       is_no_load: bool = False) -> SMIBState:
    """计算系统状态导数

    发电机方程:
        dδ/dt = (ω - 1.0) * ω_base
        dω/dt = (Pm - Pe - D*Δω) / (2H)
        dEq'/dt = (Efd - Eq') / Td0'

    Args:
        state: 当前状态
        system: 系统参数
        V_ref: 电压参考值
        dt: 时间步长
        X_ext: 外部等值电抗
        fault_active: 是否故障
        is_no_load: 是否空载

    Returns:
        SMIBState: 状态导数（Efd/PMECH 存储控制器输出用于记录）
    """
    gen = system['generator']
    gov = system['governor']
    exc = system['exciter']
    pss_model = system['pss']
    f_base = system['f_base']

    # 计算端电压和电磁功率
    Vt_mag, Pe, _ = compute_smib_voltages(
        state.delta, state.Eq_prime, gen, X_ext,
        fault_active, is_no_load=is_no_load
    )

    delta_omega = state.omega - 1.0

    # 1. PSS
    pss_reset = is_no_load  # 空载/隔离运行关掉PSS
    if pss_reset:
        V_S = 0.0
        new_pss_state = state.pss_state.copy()
    else:
        V_S, new_pss_state = pss_model.compute_stabilizing_signal_rk4(
            delta_omega, state.PMECH, dt, state.pss_state
        )

    # 2. 励磁系统
    # SEXS compute(V_ref, V_measured, V_S, dt, state)
    Efd, new_exc_state = exc.compute_rk4(
        V_ref, Vt_mag, V_S, dt, state.exc_state
    )

    # 3. 调速器
    Pm_ref, new_gov_state = gov.compute_rk4(
        delta_omega, dt, state.gov_state
    )

    # 4. 发电机导数
    omega_base = 2 * np.pi * f_base

    d_delta = delta_omega * omega_base
    d_omega = (Pm_ref - Pe - gen.D * delta_omega) / (2 * gen.H)
    d_Eq_prime = (Efd - state.Eq_prime) / gen.Td0_prime

    # 控制器导数（差分近似）
    d_exc = (new_exc_state - state.exc_state) / dt
    d_gov = (new_gov_state - state.gov_state) / dt
    d_pss = (new_pss_state - state.pss_state) / dt

    deriv = SMIBState()
    deriv.delta = d_delta
    deriv.omega = d_omega
    deriv.Eq_prime = d_Eq_prime
    deriv.exc_state = d_exc
    deriv.gov_state = d_gov
    deriv.pss_state = d_pss
    deriv.Efd = Efd
    deriv.PMECH = Pm_ref

    return deriv


def rk4_integrate(state: SMIBState, system: Dict, dt: float,
                  V_ref: float, X_ext: float, fault_active: bool = False,
                  is_no_load: bool = False) -> SMIBState:
    """RK4积分一步"""
    def f_k(s):
        return state_derivatives(s, system, V_ref, dt, X_ext, fault_active, is_no_load)

    k1 = f_k(state)

    s2 = SMIBState(delta=state.delta + dt/2*k1.delta,
                   omega=state.omega + dt/2*k1.omega,
                   Eq_prime=state.Eq_prime + dt/2*k1.Eq_prime,
                   exc_state=state.exc_state + dt/2*k1.exc_state,
                   gov_state=state.gov_state + dt/2*k1.gov_state,
                   pss_state=state.pss_state + dt/2*k1.pss_state)
    k2 = f_k(s2)

    s3 = SMIBState(delta=state.delta + dt/2*k2.delta,
                   omega=state.omega + dt/2*k2.omega,
                   Eq_prime=state.Eq_prime + dt/2*k2.Eq_prime,
                   exc_state=state.exc_state + dt/2*k2.exc_state,
                   gov_state=state.gov_state + dt/2*k2.gov_state,
                   pss_state=state.pss_state + dt/2*k2.pss_state)
    k3 = f_k(s3)

    s4 = SMIBState(delta=state.delta + dt*k3.delta,
                   omega=state.omega + dt*k3.omega,
                   Eq_prime=state.Eq_prime + dt*k3.Eq_prime,
                   exc_state=state.exc_state + dt*k3.exc_state,
                   gov_state=state.gov_state + dt*k3.gov_state,
                   pss_state=state.pss_state + dt*k3.pss_state)
    k4 = f_k(s4)

    s_new = SMIBState()
    s_new.delta = state.delta + dt/6*(k1.delta+2*k2.delta+2*k3.delta+k4.delta)
    s_new.omega = state.omega + dt/6*(k1.omega+2*k2.omega+2*k3.omega+k4.omega)
    s_new.Eq_prime = state.Eq_prime + dt/6*(k1.Eq_prime+2*k2.Eq_prime+2*k3.Eq_prime+k4.Eq_prime)
    s_new.exc_state = state.exc_state + dt/6*(k1.exc_state+2*k2.exc_state+2*k3.exc_state+k4.exc_state)
    s_new.gov_state = state.gov_state + dt/6*(k1.gov_state+2*k2.gov_state+2*k3.gov_state+k4.gov_state)
    s_new.pss_state = state.pss_state + dt/6*(k1.pss_state+2*k2.pss_state+2*k3.pss_state+k4.pss_state)
    s_new.Efd = 0  # 使用 k4 作为最终值
    s_new.PMECH = 0
    # 用 k4 的输出值
    s_new.Efd = k4.Efd
    s_new.PMECH = k4.PMECH

    return s_new


# ============================================================
# 测试案例 1: 电压参考值阶跃（空载）
# ============================================================
# 配置：S-GEN 断开，发电机孤立运行，无负荷
# 初始：Vt = 1.0, Efd = 1.0（忽略饱和）
# 事件：t=0.1s, V_ref: 1.0 → 1.05
# 仿真时长：2s, 步长 1ms
# 输出：U_NGEN(t), EFD(t)

def run_test_case_1_voltage_step(system: Dict, dt: float = 0.001,
                                    t_end: float = 2.0) -> Dict:
    """测试案例 1：电压参考值阶跃"""
    print("运行测试案例 1: 电压参考值阶跃 +0.05 p.u.")

    n_steps = int(t_end / dt)
    event_idx = int(0.1 / dt)

    # V_ref: 空载 1.0, t=0.1s 阶跃到 1.05
    V_ref_arr = np.ones(n_steps)
    V_ref_arr[event_idx:] = 1.05

    # 初始状态：空载，Efd=1.0, PMECH=0
    state = SMIBState()
    state.Eq_prime = 1.0
    state.Efd = 1.0
    state.PMECH = 0.0

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'V_ref': V_ref_arr.copy(),
        'Vt': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Pe': np.zeros(n_steps),
    }

    for i in range(n_steps):
        # 空载：Vt = Eq'
        Vt_mag, Pe, _ = compute_smib_voltages(
            state.delta, state.Eq_prime, system['generator'],
            0.0, is_no_load=True
        )

        results['Vt'][i] = Vt_mag
        results['Efd'][i] = state.Efd
        results['omega'][i] = state.omega
        results['delta'][i] = np.degrees(state.delta)
        results['Pe'][i] = Pe

        state = rk4_integrate(state, system, dt, V_ref_arr[i],
                               X_ext=0.0, is_no_load=True)

    print(f"  仿真完成: {n_steps} 步")
    return results


# ============================================================
# 测试案例 2: 负荷有功阶跃（隔离运行）
# ============================================================
# 配置：S-GEN 断开，NGEN 节点有附加恒阻抗负荷
# 初始：PL = 0.8 * S_r,G * cosφ_r = 380 MW
# 事件：t=0.1s, PL 增加 +0.05 pu
# PSS 必须关断
# 仿真时长：15s
# 输出：U_NGEN, PG, PMECH, QG, ω

def run_test_case_2_load_step(system: Dict, dt: float = 0.001,
                                 t_end: float = 15.0) -> Dict:
    """测试案例 2：负荷有功阶跃"""
    print("运行测试案例 2: 负荷有功阶跃 +0.05 p.u.")

    n_steps = int(t_end / dt)
    event_idx = int(0.1 / dt)

    # 负荷功率（标幺值）
    P_load_0 = 0.76  # 380/500

    # 初始状态
    # Eq' 需支持负荷：Pe = P_load_0
    # 对于孤立系统，Vt = Eq' (空载基准)
    # 有负荷时：Vt = Eq' - jXd'*I, I = P/Vt
    # 近似：Eq' = sqrt(Vt² + (P*Xd')²) ≈ sqrt(1 + (0.76*0.35)²) ≈ 1.028
    state = SMIBState()
    state.Eq_prime = 1.03
    state.Efd = 1.0
    state.PMECH = P_load_0

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'Vt': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Pm': np.zeros(n_steps),
        'Pe': np.zeros(n_steps),
    }

    # 关掉 PSS
    original_KS1 = system['pss'].KS1
    system['pss'].KS1 = 0.0
    print("  PSS 已关闭 (KS1=0)")

    # 负荷等效 X_ext（近似为纯有功负荷）
    # 恒阻抗：Z_load = V²/P = 1/0.76 = 1.316 pu
    # X_ext 不直接适用，用 no_load=False, X_ext=0 但 V_inf_mag=0

    for i in range(n_steps):
        is_step = (i >= event_idx)
        P_cur = P_load_0 + (0.05 if is_step else 0.0)

        # 计算端电压
        # 简化：孤立系统接恒阻抗负荷
        # I = Pe / Vt  (近似)
        # 用迭代法或直接公式
        Vt_mag, Pe, _ = compute_smib_voltages(
            state.delta, state.Eq_prime, system['generator'],
            0.0, is_no_load=(P_cur < 0.01)  # 近似空载
        )
        # 当有负荷时，修正端电压：
        # Vt 会因负荷电流而下降
        if P_cur > 0.01:
            Xd_prime = system['generator'].Xd_prime
            Z_load = 1.0 / P_cur  # 近似
            load_factor = 1.0 / np.sqrt(1 + ((Xd_prime * P_cur))**2)
            Vt_mag_adjusted = state.Eq_prime * Z_load / np.sqrt(Xd_prime**2 + Z_load**2)
            Vt_mag = min(Vt_mag, Vt_mag_adjusted)

        results['Vt'][i] = Vt_mag
        results['Efd'][i] = state.Efd
        results['omega'][i] = state.omega
        results['delta'][i] = np.degrees(state.delta)
        results['Pm'][i] = state.PMECH
        results['Pe'][i] = Pe

        state = rk4_integrate(state, system, dt, 1.0,
                               X_ext=0.0, is_no_load=(P_cur < 0.01))

    system['pss'].KS1 = original_KS1
    print(f"  仿真完成: {n_steps} 步")
    return results


# ============================================================
# 测试案例 3: 三相短路（全控运行）
# ============================================================
# 配置：基准潮流工况，全部控制器投入
# 事件：t=0.1s, NTHV 三相短路
#       t=0.2s, 故障清除
# 仿真时长：10s
# 输出：U_NGEN, EFD, PG, QG, ω, VOTHSG

def run_test_case_3_three_phase_fault(system: Dict, dt: float = 0.001,
                                          t_end: float = 10.0,
                                          fault_duration: float = 0.1) -> Dict:
    """测试案例 3：NTHV 三相短路"""
    print(f"运行测试案例 3: NTHV 三相短路 ({fault_duration}s)")

    n_steps = int(t_end / dt)
    fault_start = int(0.1 / dt)
    fault_clear = int((0.1 + fault_duration) / dt)

    # 初始状态（基准潮流工况）
    # PG = 475 MW = 0.95 pu
    # V_inf = 1.05 pu, δ 约 30-40°
    state = SMIBState()
    state.delta = np.radians(35.0)
    state.Eq_prime = 1.1
    state.Efd = 1.0
    state.PMECH = 0.95

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'Vt': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Pe': np.zeros(n_steps),
        'fault_active': np.zeros(n_steps, dtype=bool),
    }

    for i in range(n_steps):
        fault_active = (fault_start <= i < fault_clear)
        X_ext = get_external_reactance(system, fault_active)

        results['fault_active'][i] = fault_active

        Vt_mag, Pe, _ = compute_smib_voltages(
            state.delta, state.Eq_prime, system['generator'],
            X_ext, fault_active, V_inf_mag=1.05
        )

        results['Vt'][i] = Vt_mag
        results['Efd'][i] = state.Efd
        results['omega'][i] = state.omega
        results['delta'][i] = np.degrees(state.delta)
        results['Pe'][i] = Pe

        state = rk4_integrate(state, system, dt, 1.05,
                               X_ext, fault_active)

    print(f"  仿真完成: {n_steps} 步")
    return results


# ============================================================
# 绘图
# ============================================================

def plot_test_case_results(case_num: int, results: Dict):
    """绘制测试案例结果"""
    time = results['time']

    if case_num == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Test Case 1: Voltage Reference Step +0.05 pu', fontsize=14)

        ax = axes[0, 0]
        ax.plot(time, results['V_ref'], 'b-', label='V_ref', lw=2)
        ax.plot(time, results['Vt'], 'r--', label='V_NGEN', lw=1.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Voltage (pu)')
        ax.set_title('Terminal Voltage'); ax.legend(); ax.grid(True)

        ax = axes[0, 1]
        ax.plot(time, results['Efd'], 'g-', lw=1.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Efd (pu)')
        ax.set_title('Excitation Voltage'); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(time, results['omega'], 'm-', lw=1.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Speed (pu)')
        ax.set_title('Rotor Speed'); ax.grid(True)

        ax = axes[1, 1]
        ax.plot(time, results['Pe'], 'c-', lw=1.5)
        ax.set_xlabel('Time (s)'); ax.set_ylabel('Pe (pu)')
        ax.set_title('Electrical Power'); ax.grid(True)

        plt.tight_layout(); plt.show()

    elif case_num == 2:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Test Case 2: Load Step +0.05 pu', fontsize=14)

        axes[0, 0].plot(time, results['Vt'], 'r-', label='V_NGEN')
        axes[0, 0].set_title('Terminal Voltage'); axes[0, 0].grid(True)

        axes[0, 1].plot(time, results['Pm'], 'b-', label='P_MECH')
        axes[0, 1].set_title('Mechanical Power'); axes[0, 1].grid(True)

        axes[1, 0].plot(time, results['omega'], 'm-', label='Speed')
        axes[1, 0].set_title('Rotor Speed'); axes[1, 0].grid(True)

        axes[1, 1].plot(time, results['Pe'], 'g-', label='P_ELEC')
        axes[1, 1].set_title('Electrical Power'); axes[1, 1].grid(True)

        axes[2, 0].plot(time, results['Efd'], 'c-', label='Efd')
        axes[2, 0].set_title('Excitation Voltage'); axes[2, 0].grid(True)

        axes[2, 1].plot(time, results['delta'], 'y-', label='Angle')
        axes[2, 1].set_title('Rotor Angle'); axes[2, 1].grid(True)

        plt.tight_layout(); plt.show()

    elif case_num == 3:
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle('Test Case 3: Three-Phase Short Circuit', fontsize=14)

        ax = axes[0, 0]
        ax.plot(time, results['Vt'], 'r-', label='V_NGEN', lw=1.5)
        ax.plot(time, results['fault_active'].astype(float)*0.5, 'k--', alpha=0.3, label='Fault')
        ax.set_title('Terminal Voltage'); ax.legend(); ax.grid(True)

        axes[0, 1].plot(time, results['Efd'], 'c-', label='Efd', lw=1.5)
        axes[0, 1].set_title('Excitation Voltage'); axes[0, 1].grid(True)

        axes[1, 0].plot(time, results['omega'], 'm-', label='Speed', lw=1.5)
        axes[1, 0].set_title('Rotor Speed'); axes[1, 0].grid(True)

        axes[1, 1].plot(time, results['Pe'], 'g-', label='P_ELEC', lw=1.5)
        axes[1, 1].set_title('Electrical Power'); axes[1, 1].grid(True)

        axes[2, 0].plot(time, results['delta'], 'b-', label='Angle', lw=1.5)
        axes[2, 0].set_title('Rotor Angle'); axes[2, 0].grid(True)

        plt.tight_layout(); plt.show()


# ============================================================
# 快速验证
# ============================================================

if __name__ == '__main__':
    system = build_entsoe_smib_system()
    print("System built OK")

    print("\n--- Test Case 1 ---")
    r1 = run_test_case_1_voltage_step(system, dt=0.001, t_end=1.0)
    print(f"  Vt: {r1['Vt'][0]:.4f} -> {r1['Vt'][-1]:.4f} (max={max(r1['Vt']):.4f})")
    print(f"  Efd: {r1['Efd'][0]:.4f} -> {r1['Efd'][-1]:.4f} (max={max(r1['Efd']):.4f})")

    print("\n--- Test Case 3 ---")
    r3 = run_test_case_3_three_phase_fault(system, dt=0.001, t_end=2.0)
    print(f"  Vt min: {min(r3['Vt']):.4f}")
    print(f"  ω range: {min(r3['omega']):.4f} ~ {max(r3['omega']):.4f}")
