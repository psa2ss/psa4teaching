"""
psa4teaching 完整使用示例
=========================

本示例演示如何使用psa4teaching进行潮流计算、短路计算和稳定计算。
"""

import numpy as np
from psa4teaching.models import Bus, BusType, Line, Transformer, Generator, Load, LoadModel
from psa4teaching.network import build_ybus, build_zbus
from psa4teaching.powerflow import run_newton_raphson, run_fast_decoupled, run_dc_powerflow
from psa4teaching.shortcircuit import (
    calculate_three_phase_fault,
    calculate_single_line_to_ground,
    calculate_line_to_line,
    calculate_double_line_to_ground,
    calculate_gb15544,
)
from psa4teaching.stability import (
    simulate_single_machine_infinite_bus_classic,
    simulate_single_machine_infinite_bus_detailed,
    analyze_single_machine_infinite_bus,
)


def example_powerflow():
    """潮流计算示例"""
    print("\n" + "="*60)
    print("潮流计算示例")
    print("="*60)

    # 定义系统
    buses = [
        Bus(1, "Slack", BusType.SLACK, V_specified=1.05),
        Bus(2, "Gen1", BusType.PV, P_specified=0.5, V_specified=1.02),
        Bus(3, "Load1", BusType.PQ, P_specified=-0.8, Q_specified=-0.3),
        Bus(4, "Load2", BusType.PQ, P_specified=-0.6, Q_specified=-0.2),
    ]

    lines = [
        Line(1, 2, R=0.02, X=0.1, B=0.02),
        Line(2, 3, R=0.03, X=0.15, B=0.03),
        Line(2, 4, R=0.025, X=0.12, B=0.025),
        Line(3, 4, R=0.04, X=0.2, B=0.04),
    ]

    transformers = []

    # 构造导纳矩阵
    ybus_result = build_ybus(lines, transformers)
    print(f"\n节点导纳矩阵 Ybus ({ybus_result.n_bus}×{ybus_result.n_bus}):")
    print(ybus_result.Ybus)

    # 牛顿-拉夫逊法
    print("\n--- 牛顿-拉夫逊法 ---")
    nr_result = run_newton_raphson(buses, ybus_result, max_iterations=50, tolerance=1e-8)
    print(f"收敛: {nr_result.converged}")
    print(f"迭代次数: {nr_result.iterations}")
    print(f"最大偏差: {nr_result.max_mismatch:.2e}")
    print(f"电压幅值: {nr_result.V}")
    print(f"电压相角(度): {np.degrees(nr_result.delta)}")
    print(f"网络损耗: {nr_result.losses:.4f} p.u.")

    # P-Q分解法
    print("\n--- P-Q分解法 ---")
    fd_result = run_fast_decoupled(buses, ybus_result, max_iterations=100, tolerance=1e-8)
    print(f"收敛: {fd_result.converged}")
    print(f"迭代次数: {fd_result.iterations}")

    # 直流潮流
    print("\n--- 直流潮流 ---")
    dc_result = run_dc_powerflow(buses, lines, transformers)
    print(f"电压相角(度): {np.degrees(dc_result.theta)}")
    print(f"支路功率流:")
    for (i, j), P in dc_result.branch_flows.items():
        print(f"  {i}-{j}: {P:.4f} p.u.")

    return nr_result


def example_shortcircuit():
    """短路计算示例"""
    print("\n" + "="*60)
    print("短路计算示例")
    print("="*60)

    # 定义系统
    buses = [
        Bus(1, "G1", BusType.SLACK, V_specified=1.05),
        Bus(2, "G2", BusType.PV, P_specified=0.3, V_specified=1.03),
        Bus(3, "Load", BusType.PQ, P_specified=-0.8, Q_specified=-0.3),
    ]

    lines = [
        Line(1, 3, R=0.02, X=0.1, B=0.02),
        Line(2, 3, R=0.025, X=0.12, B=0.025),
        Line(1, 2, R=0.03, X=0.15, B=0.03),
    ]

    generators = [
        Generator(bus=1, name="G1", Xd_doubleprime=0.2, H=6.0),
        Generator(bus=2, name="G2", Xd_doubleprime=0.25, H=5.0),
    ]

    fault_bus = 3  # 短路点设为节点3

    # 三相短路
    print(f"\n--- 三相短路（节点{fault_bus}）---")
    result = calculate_three_phase_fault(
        buses, lines, [], generators, fault_bus=fault_bus,
        S_base=100.0, V_base=10.5
    )
    print(f"短路电流: {result.fault_current_ka:.2f} kA")
    print(f"短路点自阻抗 Zff = {result.Zff:.4f} p.u.")
    print(f"短路后各节点电压:")
    for bus_num, V in result.V_pu.items():
        print(f"  节点{bus_num}: {V:.4f} p.u.")
    print(f"转移阻抗:")
    for bus_num, Z in result.transfer_impedances.items():
        print(f"  发电机{bus_num}→短路点: {Z:.4f} p.u.")

    # 不对称短路
    print(f"\n--- 单相接地短路（节点{fault_bus}，a相）---")
    result_slg = calculate_single_line_to_ground(
        buses, lines, [], generators, fault_bus=fault_bus
    )
    print(f"故障电流: {abs(result_slg.fault_current):.4f} p.u.")
    print(f"序电流: I0={abs(result_slg.sequence_currents[0]):.4f}, "
          f"I1={abs(result_slg.sequence_currents[1]):.4f}, "
          f"I2={abs(result_slg.sequence_currents[2]):.4f} p.u.")
    print(f"三相电流:")
    print(f"  Ia = {result_slg.fault_currents_3phase[0]:.4f} p.u.")
    print(f"  Ib = {result_slg.fault_currents_3phase[1]:.4f} p.u.")
    print(f"  Ic = {result_slg.fault_currents_3phase[2]:.4f} p.u.")

    print(f"\n--- 两相短路（节点{fault_bus}，bc相）---")
    result_ll = calculate_line_to_line(buses, lines, [], generators, fault_bus=fault_bus)
    print(f"三相电流:")
    print(f"  Ia = {result_ll.fault_currents_3phase[0]:.4f} p.u.")
    print(f"  Ib = {result_ll.fault_currents_3phase[1]:.4f} p.u.")
    print(f"  Ic = {result_ll.fault_currents_3phase[2]:.4f} p.u.")

    print(f"\n--- 两相接地短路（节点{fault_bus}，bc相接地）---")
    result_dlg = calculate_double_line_to_ground(
        buses, lines, [], generators, fault_bus=fault_bus
    )
    print(f"故障电流: {abs(result_dlg.fault_current):.4f} p.u.")

    # GB15544
    print(f"\n--- GB/T 15544 短路计算 ---")
    result_gb = calculate_gb15544(
        buses, lines, [], generators, fault_bus=fault_bus,
        V_nominal=10.5, max_current=True, verbose=True
    )

    return result


def example_stability():
    """稳定计算示例"""
    print("\n" + "="*60)
    print("稳定计算示例")
    print("="*60)

    # 系统参数
    E_prime = 1.2      # 暂态电势
    V_inf = 1.0        # 无穷大母线电压
    X_total = 0.5      # 等值电抗
    H = 5.0            # 惯性常数
    D = 2.0            # 阻尼系数
    Pm = 0.8           # 机械功率
    delta_0 = np.radians(30)  # 初始功角30度

    # 小干扰稳定分析
    print("\n--- 小干扰稳定分析（经典模型）---")
    ss_result = analyze_single_machine_infinite_bus(
        E_prime, V_inf, X_total, H, D, delta_0, Pm, verbose=True
    )

    # 暂态稳定仿真 - 经典模型
    print("\n--- 暂态稳定仿真（经典模型）---")
    print("故障清除时间: 0.15秒")
    result_1 = simulate_single_machine_infinite_bus_classic(
        E_prime, V_inf, X_total, H, D, Pm, delta_0,
        fault_time=0.0, fault_clearing_time=0.15,
        t_end=5.0, method="rk4"
    )
    print(f"系统{'稳定' if result_1.stable else '不稳定'}")
    print(f"最大功角: {result_1.max_delta:.2f}°")

    # 暂态稳定仿真 - 详细模型
    print("\n--- 暂态稳定仿真（详细模型）---")
    result_2 = simulate_single_machine_infinite_bus_detailed(
        E_prime_0=E_prime, V_infinity=V_inf, X_total=X_total,
        Xd=1.8, Xd_prime=0.3, Xq=1.7,
        Td0_prime=8.0, H=H, D=D,
        Pm_0=Pm, delta_0=delta_0, Efd_0=2.0,
        fault_time=0.0, fault_clearing_time=0.15,
        t_end=5.0
    )
    print(f"系统{'稳定' if result_2.stable else '不稳定'}")
    print(f"最大功角: {result_2.max_delta:.2f}°")

    return result_1


if __name__ == "__main__":
    # 运行所有示例
    print("="*60)
    print("psa4teaching 使用示例")
    print("="*60)

    example_powerflow()
    example_shortcircuit()
    example_stability()

    print("\n" + "="*60)
    print("示例运行完成")
    print("="*60)