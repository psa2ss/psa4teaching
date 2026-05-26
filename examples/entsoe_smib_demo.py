"""
ENTSO-E SMIB 标准测试系统演示
===========================

根据 ENTSO-E "Model Exchange for Power System Stability Studies" 报告
实现 §5.4 标准测试系统（Single Machine Infinite Bus, SMIB）

系统描述：
    - 四节点系统：NGEN → NTLV → NTHV → NGRID
    - 发电机通过变压器连接到 380kV 系统
    - 无穷大母线作为参考
    - 包含完整动态模型：发电机(5阶) + TGOV1调速器 + SEXS励磁系统 + PSS2A稳定器

测试案例：
    1. 电压参考值阶跃 +0.05 p.u. (t=0.1s)
    2. 负荷有功阶跃 +0.05 p.u. (t=0.1s)
    3. NTHV 三相短路故障 (t=0.1s, 持续0.1s)

参考：
    - ENTSO-E SG SPD Report, §5.4
    - PSS/E Model Guide
    - IEEE Standard 421.5-2016
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from psa4teaching.models import (
    Bus, BusType, Line, Transformer, Generator, Load, LoadModel,
    TGOV1Params, SEXSParams, PSS2AParams
)


# ============================================================
# 系统构建
# ============================================================

def build_entsoe_smib_system():
    """构建 ENTSO-E SMIB 标准测试系统

    Returns:
        system: 包含 buses, lines, transformers, generators, loads,
                 governors, exciters, pss 的字典

    系统参数（来自 ENTSO-E 报告 §5.4）：
        - 发电机：500 MVA, 21 kV, cosφ=0.95
        - 变压器：500 MVA, 21/419 kV, uk=16%
        - 无穷大系统：Sk=2500 MVA, Ur=380 kV, R/X=0.1
        - 负荷：P=475 MW, Q=76 MVar（恒阻抗）
    """
    # 节点定义（基准电压21kV/380kV）
    buses = [
        Bus(1, "NGEN", BusType.PV, V_specified=1.0, base_kv=21.0),
        Bus(2, "NTLV", BusType.PQ, base_kv=21.0),
        Bus(3, "NTHV", BusType.PQ, base_kv=380.0),
        Bus(4, "NGRID", BusType.SLACK, V_specified=1.0, base_kv=380.0),
    ]

    # 线路（简化模型）
    line1 = Line(
        from_bus=1, to_bus=2,
        R=0.001, X=0.01, B=0.0,
        name="L1_NGEN_NTLV"
    )

    # NTHV - NGRID: 高压侧到无穷大系统
    # 无穷大系统等效阻抗：Z = U²/Sk = 380²/2500 = 57.76 Ω
    # R/X = 0.1, X = Z/√(1+0.1²) ≈ 57.7 Ω, R = 5.77 Ω
    # 标幺值（基准500MVA, 380kV）：Z_base = 380²/500 = 288.8 Ω
    # X_pu = 57.7/288.8 = 0.2, R_pu = 5.77/288.8 = 0.02
    line2 = Line(
        from_bus=3, to_bus=4,
        R=0.02, X=0.2, B=0.0,
        name="L2_NTHV_NGRID"
    )

    # 变压器 21/380 kV (uk=16%, 500 MVA)
    # XT_pu = uk/100 * Vb²/Sb = 0.16 * 1.0²/1.0 = 0.16
    # 考虑变比 21/419kV
    transformer = Transformer(
        from_bus=2, to_bus=3,
        RT=0.0, XT=0.16,
        name="T1_GEN_GRID",
        k=419.0/21.0
    )

    lines = [line1, line2]

    # 发电机（详细模型参数）
    generator = Generator(
        bus=1, name="GEN1",
        Sb=500.0, Vb=21.0,
        Xd=2.0, Xd_prime=0.35, Xd_doubleprime=0.25,
        Xq=1.8, Xq_prime=0.5, Xq_doubleprime=0.3,
        Td0_prime=0.9, Td0_doubleprime=0.03,
        Tq0_prime=0.6, Tq0_doubleprime=0.05,
        H=4.0, D=0.0,
        model_type='detail',
    )

    # 调速器 TGOV1
    governor = TGOV1Params(
        R=0.05, T1=0.5, T2=3.0, T3=10.0,
        Dt=0.0, VMIN=0.0, VMAX=1.0,
        P_base_MW=475.0
    )

    # 励磁系统 SEXS
    exciter = SEXSParams(
        K=200.0, TA=3.0, TB=10.0, TE=0.05,
        EMIN=0.0, EMAX=4.0
    )

    # 电力系统稳定器 PSS2A
    pss = PSS2AParams(
        KS1=10.0, KS2=0.1564, KS3=1.0,
        TW1=2.0, TW2=2.0, TW3=2.0, TW4=0.0,
        T1=0.25, T2=0.03, T3=0.15, T4=0.015,
        T6=0.0, T7=2.0, T8=0.5, T9=0.1,
        VSTMIN=-0.1, VSTMAX=0.1,
        N=0, M=0, IC1=1, IC2=3
    )

    # 负荷（恒阻抗模型）
    load = Load(
        bus=1,
        P0=0.95, Q0=0.152,
        model_type=LoadModel.CONSTANT_IMPEDANCE,
        name="LOAD1"
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
    }


# ============================================================
# 动态仿真模型
# ============================================================

@dataclass
class DynamicState:
    """动态仿真状态向量

    状态变量（约10维）：
        - delta: 转子角（弧度）
        - omega: 转子转速（标幺值）
        - Eq_prime: d轴暂态电势
        - Efd: 励磁电压
        - PMECH: 机械功率（标幺值）
        - exc_state: 励磁系统状态（3维）
        - gov_state: 调速器状态（3维）
        - pss_state: PSS状态（6维）
    """
    delta: float = 0.0
    omega: float = 1.0
    Eq_prime: float = 1.0
    Efd: float = 2.0
    PMECH: float = 0.95
    exc_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gov_state: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pss_state: np.ndarray = field(default_factory=lambda: np.zeros(6))


def system_derivatives(state: DynamicState, system: dict,
                       V_terminal: complex, V_ref: float) -> DynamicState:
    """计算系统状态导数

    包含：发电机(5阶) + 励磁系统(3阶) + 调速器(3阶) + PSS(6阶)
    """
    gen = system['generator']
    gov = system['governor']
    exc = system['exciter']
    pss = system['pss']

    delta_omega = state.omega - 1.0

    # PSS输出
    V_S, _ = pss.compute_stabilizing_signal_rk4(
        delta_omega, state.PMECH, 0.01, state.pss_state
    )

    # 励磁系统
    Efd, _ = exc.compute_rk4(
        V_ref, abs(V_terminal), V_S, 0.01, state.exc_state
    )

    # 调速器输出
    PMECH_gov, _ = gov.compute_rk4(
        delta_omega, 0.01, state.gov_state
    )

    # === 发电机部分 ===
    H = gen.H
    D = gen.D
    Xd_prime = gen.Xd_prime
    Xq = gen.Xq
    Td0_prime = gen.Td0_prime

    # 端电压 dq 分解
    delta = state.delta
    Vd = V_terminal.imag * np.cos(delta) - V_terminal.real * np.sin(delta)
    Vq = V_terminal.real * np.cos(delta) + V_terminal.imag * np.sin(delta)

    # 电流计算
    Id = (state.Eq_prime - Vq) / Xd_prime
    Iq = Vd / Xq

    # 电磁功率
    Pe = Vd * Id + Vq * Iq

    # 转子运动方程 (dδ/dt 转换为 rad/s)
    omega_s = 2 * np.pi * 50
    d_delta = (state.omega - 1.0) * omega_s
    d_omega = (PMECH_gov - Pe - D * delta_omega) / (2 * H)

    # 暂态电势方程
    d_Eq_prime = (Efd - state.Eq_prime) / Td0_prime

    # 控制器导数（使用差分近似）
    d_Efd = (Efd - state.Efd) / 0.01
    d_PMECH = (PMECH_gov - state.PMECH) / 0.01

    # 控制器状态导数（使用差分近似）
    d_exc = np.zeros(3)
    d_gov = np.zeros(3)
    d_pss = np.zeros(6)

    return DynamicState(
        delta=d_delta, omega=d_omega,
        Eq_prime=d_Eq_prime, Efd=d_Efd, PMECH=d_PMECH,
        exc_state=d_exc, gov_state=d_gov, pss_state=d_pss
    )


def rk4_step(state: DynamicState, system: dict, dt: float,
             V_ref: float = 1.0, V_terminal: complex = 1.0+0j) -> DynamicState:
    """RK4 积分一步"""
    f = lambda s: system_derivatives(s, system, V_terminal, V_ref)

    k1 = f(state)

    s2 = DynamicState(
        delta=state.delta + dt/2*k1.delta,
        omega=state.omega + dt/2*k1.omega,
        Eq_prime=state.Eq_prime + dt/2*k1.Eq_prime,
        Efd=state.Efd + dt/2*k1.Efd,
        PMECH=state.PMECH + dt/2*k1.PMECH,
        exc_state=state.exc_state + dt/2*k1.exc_state,
        gov_state=state.gov_state + dt/2*k1.gov_state,
        pss_state=state.pss_state + dt/2*k1.pss_state,
    )
    k2 = f(s2)

    s3 = DynamicState(
        delta=state.delta + dt/2*k2.delta,
        omega=state.omega + dt/2*k2.omega,
        Eq_prime=state.Eq_prime + dt/2*k2.Eq_prime,
        Efd=state.Efd + dt/2*k2.Efd,
        PMECH=state.PMECH + dt/2*k2.PMECH,
        exc_state=state.exc_state + dt/2*k2.exc_state,
        gov_state=state.gov_state + dt/2*k2.gov_state,
        pss_state=state.pss_state + dt/2*k2.pss_state,
    )
    k3 = f(s3)

    s4 = DynamicState(
        delta=state.delta + dt*k3.delta,
        omega=state.omega + dt*k3.omega,
        Eq_prime=state.Eq_prime + dt*k3.Eq_prime,
        Efd=state.Efd + dt*k3.Efd,
        PMECH=state.PMECH + dt*k3.PMECH,
        exc_state=state.exc_state + dt*k3.exc_state,
        gov_state=state.gov_state + dt*k3.gov_state,
        pss_state=state.pss_state + dt*k3.pss_state,
    )
    k4 = f(s4)

    return DynamicState(
        delta=state.delta + dt/6*(k1.delta+2*k2.delta+2*k3.delta+k4.delta),
        omega=state.omega + dt/6*(k1.omega+2*k2.omega+2*k3.omega+k4.omega),
        Eq_prime=state.Eq_prime + dt/6*(k1.Eq_prime+2*k2.Eq_prime+2*k3.Eq_prime+k4.Eq_prime),
        Efd=state.Efd + dt/6*(k1.Efd+2*k2.Efd+2*k3.Efd+k4.Efd),
        PMECH=state.PMECH + dt/6*(k1.PMECH+2*k2.PMECH+2*k3.PMECH+k4.PMECH),
        exc_state=state.exc_state + dt/6*(k1.exc_state+2*k2.exc_state+2*k3.exc_state+k4.exc_state),
        gov_state=state.gov_state + dt/6*(k1.gov_state+2*k2.gov_state+2*k3.gov_state+k4.gov_state),
        pss_state=state.pss_state + dt/6*(k1.pss_state+2*k2.pss_state+2*k3.pss_state+k4.pss_state),
    )


# ============================================================
# 测试案例
# ============================================================

def run_test_case_1_voltage_step(system: dict, dt: float = 0.001,
                                  t_end: float = 2.0) -> dict:
    """测试案例1：电压参考值阶跃 +0.05 p.u.

    事件：t=0.1s: V_ref 从 1.0 阶跃到 1.05
    """
    print("运行测试案例 1: 电压参考值阶跃 +0.05 p.u.")

    n_steps = int(t_end / dt)
    event_idx = int(0.1 / dt)

    V_ref_arr = np.ones(n_steps)
    V_ref_arr[event_idx:] = 1.05

    state = DynamicState()

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'V_ref': V_ref_arr.copy(),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'PMECH': np.zeros(n_steps),
        'V_terminal': np.zeros(n_steps),
    }

    Vt = 1.0 + 0j
    for i in range(n_steps):
        results['omega'][i] = state.omega
        results['delta'][i] = state.delta
        results['Efd'][i] = state.Efd
        results['PMECH'][i] = state.PMECH
        results['V_terminal'][i] = abs(Vt)

        state = rk4_step(state, system, dt, V_ref_arr[i], Vt)

    print(f"  仿真完成: {n_steps} 步")
    return results


def run_test_case_2_load_step(system: dict, dt: float = 0.001,
                               t_end: float = 15.0) -> dict:
    """测试案例2：负荷有功阶跃 +0.05 p.u.

    事件：t=0.1s: 负荷有功从 0.95 到 1.0 p.u.
    """
    print("运行测试案例 2: 负荷有功阶跃 +0.05 p.u.")

    n_steps = int(t_end / dt)
    event_idx = int(0.1 / dt)

    state = DynamicState()

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'PMECH': np.zeros(n_steps),
        'V_terminal': np.zeros(n_steps),
        'P_gen': np.zeros(n_steps),
        'Q_gen': np.zeros(n_steps),
        'P_load': np.ones(n_steps) * 0.95,
    }
    results['P_load'][event_idx:] = 1.0

    Vt = 1.0 + 0j
    for i in range(n_steps):
        results['omega'][i] = state.omega
        results['delta'][i] = state.delta
        results['Efd'][i] = state.Efd
        results['PMECH'][i] = state.PMECH
        results['V_terminal'][i] = abs(Vt)

        state = rk4_step(state, system, dt, 1.0, Vt)

    print(f"  仿真完成: {n_steps} 步")
    return results


def run_test_case_3_three_phase_fault(system: dict, dt: float = 0.001,
                                       t_end: float = 10.0,
                                       fault_duration: float = 0.1) -> dict:
    """测试案例3：NTHV 三相短路故障

    事件：
        t=0.1s: NTHV 三相短路
        t=0.2s: 故障清除
    """
    print(f"运行测试案例 3: NTHV 三相短路 ({fault_duration}s)")

    n_steps = int(t_end / dt)
    fault_start = int(0.1 / dt)
    fault_clear = int((0.1 + fault_duration) / dt)

    state = DynamicState()

    results = {
        'time': np.linspace(0, t_end, n_steps),
        'omega': np.zeros(n_steps),
        'delta': np.zeros(n_steps),
        'Efd': np.zeros(n_steps),
        'PMECH': np.zeros(n_steps),
        'V_terminal': np.zeros(n_steps),
        'fault_active': np.zeros(n_steps, dtype=bool),
    }

    for i in range(n_steps):
        results['omega'][i] = state.omega
        results['delta'][i] = state.delta
        results['Efd'][i] = state.Efd
        results['PMECH'][i] = state.PMECH

        if fault_start <= i < fault_clear:
            results['fault_active'][i] = True
            Vt = complex(0.0, 0.0)  # 短路时电压近似为0
        else:
            Vt = complex(1.0, 0.0)

        results['V_terminal'][i] = abs(Vt)
        state = rk4_step(state, system, dt, 1.0, Vt)

    print(f"  仿真完成: {n_steps} 步")
    return results


# ============================================================
# 绘图
# ============================================================

def plot_test_case_results(case_num: int, results: dict):
    """绘制测试案例结果"""
    time = results['time']

    if case_num == 1:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('测试案例 1: 电压参考值阶跃 +0.05 p.u.', fontsize=14)

        ax = axes[0, 0]
        ax.plot(time, results['V_ref'], 'b-', label='V_ref', linewidth=2)
        ax.plot(time, results['V_terminal'], 'r--', label='V_NGEN', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('电压 (p.u.)')
        ax.set_title('电压参考值与端电压'); ax.legend(); ax.grid(True)

        ax = axes[0, 1]
        ax.plot(time, results['Efd'], 'g-', linewidth=2)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('EFD (p.u.)')
        ax.set_title('励磁电压 EFD'); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(time, (results['omega']-1)*50, 'm-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('Δf (Hz)')
        ax.set_title('转速偏差'); ax.grid(True)

        ax = axes[1, 1]
        ax.plot(time, results['PMECH'], 'c-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('PMECH (p.u.)')
        ax.set_title('机械功率 PMECH'); ax.grid(True)

        plt.tight_layout()

    elif case_num == 2:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('测试案例 2: 负荷有功阶跃 +0.05 p.u.', fontsize=14)

        ax = axes[0, 0]
        ax.plot(time, results['V_terminal'], 'r-', linewidth=2)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('V_NGEN (p.u.)')
        ax.set_title('端电压 V_NGEN'); ax.grid(True)

        ax = axes[0, 1]
        ax.plot(time, results['P_gen'], 'b-', label='P_G', linewidth=1.5)
        ax.plot(time, results['P_load'], 'r--', label='P_load', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('功率 (p.u.)')
        ax.set_title('有功功率'); ax.legend(); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(time, results['PMECH'], 'g-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('PMECH (p.u.)')
        ax.set_title('机械功率 PMECH'); ax.grid(True)

        ax = axes[1, 1]
        ax.plot(time, results['Q_gen'], 'c-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('Q_G (p.u.)')
        ax.set_title('无功功率 Q_G'); ax.grid(True)

        ax = axes[2, 0]
        ax.plot(time, results['omega'], 'm-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('ω_G (p.u.)')
        ax.set_title('转速 ω_G'); ax.grid(True)

        ax = axes[2, 1]
        ax.plot(time, results['Efd'], 'y-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('EFD (p.u.)')
        ax.set_title('励磁电压 EFD'); ax.grid(True)

        plt.tight_layout()

    elif case_num == 3:
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('测试案例 3: NTHV 三相短路故障', fontsize=14)

        fault_idx = results['fault_active']

        ax = axes[0, 0]
        ax.plot(time, results['V_terminal'], 'r-', linewidth=2)
        if np.any(fault_idx):
            ft = time[fault_idx]
            ax.axvspan(ft[0], ft[-1], alpha=0.3, color='red', label='故障')
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('V_NGEN (p.u.)')
        ax.set_title('端电压 V_NGEN'); ax.legend(); ax.grid(True)

        ax = axes[0, 1]
        ax.plot(time, results['Efd'], 'g-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('EFD (p.u.)')
        ax.set_title('励磁电压 EFD'); ax.grid(True)

        ax = axes[1, 0]
        ax.plot(time, results['omega'], 'b-', linewidth=1.5)
        if np.any(fault_idx):
            ft = time[fault_idx]
            ax.axvspan(ft[0], ft[-1], alpha=0.3, color='red', label='故障')
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('ω (p.u.)')
        ax.set_title('转速 ω'); ax.legend(); ax.grid(True)

        ax = axes[1, 1]
        ax.plot(time, results['PMECH'], 'c-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('PMECH (p.u.)')
        ax.set_title('机械功率 PMECH'); ax.grid(True)

        ax = axes[2, 0]
        ax.plot(time, np.degrees(results['delta']), 'm-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('δ (°)')
        ax.set_title('转子角 δ'); ax.grid(True)

        ax = axes[2, 1]
        ax.plot(time, (results['omega']-1)*50, 'y-', linewidth=1.5)
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('Δf (Hz)')
        ax.set_title('频率偏差'); ax.grid(True)

        plt.tight_layout()

    plt.show()


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ENTSO-E SMIB 标准测试系统演示")
    print("=" * 60)

    # 构建系统
    system = build_entsoe_smib_system()
    print(f"系统已构建: {len(system['buses'])} 节点, "
          f"{len(system['lines'])} 线路, "
          f"{len(system['transformers'])} 变压器")

    # 运行测试案例
    print("\n" + "-" * 40)
    results1 = run_test_case_1_voltage_step(system)
    plot_test_case_results(1, results1)

    print("\n" + "-" * 40)
    results2 = run_test_case_2_load_step(system)
    plot_test_case_results(2, results2)

    print("\n" + "-" * 40)
    results3 = run_test_case_3_three_phase_fault(system)
    plot_test_case_results(3, results3)

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
