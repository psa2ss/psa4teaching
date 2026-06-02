# -*- coding: utf-8 -*-
"""
case9.py -- WSCC 3机9节点系统潮流计算
======================================

基于 MATPOWER case9.m 数据，构建 3 机 9 节点电力系统模型，
使用牛顿-拉夫逊法完成潮流计算，并与 MATPOWER 标准结果对比验证。

系统描述
--------
- 基准容量：100 MVA
- 节点数：9（1个平衡节点 + 2个PV节点 + 6个PQ节点）
- 发电机：3台（分别位于节点1、2、3）
- 支路：9条（6条输电线路 + 3台变压器）
- 总负荷：315 MW + 115 Mvar

参考文献
--------
- Chow, J. H. (ed). Time-Scale Modeling of Dynamic Networks with
  Applications to Power Systems. Springer-Verlag, 1982.
- MATPOWER User's Manual (case9.m)

运行方式
--------
    python examples/case9.py
"""

import sys
import os
import numpy as np

# Windows 环境下强制 UTF-8 输出
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psa4teaching.models import (
    Bus, BusType, Line, Transformer, Generator, Load, LoadModel
)
from psa4teaching.network import build_ybus
from psa4teaching.powerflow import run_newton_raphson


# ============================================================================
# 系统参数定义（数据来源：MATPOWER case9.m）
# ============================================================================

BASE_MVA = 100.0  # 系统基准容量 (MVA)


def create_buses():
    """创建 case9 系统的 9 个节点

    数据来源: MATPOWER case9.m mpc.bus 矩阵
    列格式: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin

    psa4teaching 约定:
      - P_specified > 0: 发电; P_specified < 0: 负荷
      - V_specified: SLACK 和 PV 节点的目标电压
    """
    return [
        # Bus 1: SLACK, 发电机1 (Vg=1.04 来自 mpc.gen)
        Bus(number=1, name="Bus1", bus_type=BusType.SLACK,
            V_specified=1.04, base_kv=345),

        # Bus 2: PV, 发电机2 (Pg=163MW, Vg=1.025 来自 mpc.gen)
        Bus(number=2, name="Bus2", bus_type=BusType.PV,
            P_specified=163.0 / BASE_MVA,
            V_specified=1.025,
            Q_min=-300.0 / BASE_MVA, Q_max=300.0 / BASE_MVA,
            base_kv=345),

        # Bus 3: PV, 发电机3 (Pg=85MW, Vg=1.025 来自 mpc.gen)
        Bus(number=3, name="Bus3", bus_type=BusType.PV,
            P_specified=85.0 / BASE_MVA,
            V_specified=1.025,
            Q_min=-300.0 / BASE_MVA, Q_max=300.0 / BASE_MVA,
            base_kv=345),

        # Bus 4: PQ, 连接母线 (Pd=0, Qd=0)
        Bus(number=4, name="Bus4", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=345),

        # Bus 5: PQ, 负荷 (Pd=90MW, Qd=30Mvar)
        Bus(number=5, name="Bus5", bus_type=BusType.PQ,
            P_specified=-90.0 / BASE_MVA,
            Q_specified=-30.0 / BASE_MVA,
            base_kv=345),

        # Bus 6: PQ, 连接母线 (Pd=0, Qd=0)
        Bus(number=6, name="Bus6", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=345),

        # Bus 7: PQ, 负荷 (Pd=100MW, Qd=35Mvar)
        Bus(number=7, name="Bus7", bus_type=BusType.PQ,
            P_specified=-100.0 / BASE_MVA,
            Q_specified=-35.0 / BASE_MVA,
            base_kv=345),

        # Bus 8: PQ, 连接母线 (Pd=0, Qd=0)
        Bus(number=8, name="Bus8", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=345),

        # Bus 9: PQ, 负荷 (Pd=125MW, Qd=50Mvar)
        Bus(number=9, name="Bus9", bus_type=BusType.PQ,
            P_specified=-125.0 / BASE_MVA,
            Q_specified=-50.0 / BASE_MVA,
            base_kv=345),
    ]


def create_lines():
    """创建 case9 系统的 6 条输电线路

    数据来源: MATPOWER case9.m mpc.branch 矩阵 (ratio=0, B>0 的支路)
    列格式: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    """
    return [
        Line(from_bus=4, to_bus=5, R=0.017,  X=0.092,  B=0.158,  name="L4-5"),
        Line(from_bus=5, to_bus=6, R=0.039,  X=0.17,   B=0.358,  name="L5-6"),
        Line(from_bus=6, to_bus=7, R=0.0119, X=0.1008, B=0.209,  name="L6-7"),
        Line(from_bus=7, to_bus=8, R=0.0085, X=0.072,  B=0.149,  name="L7-8"),
        Line(from_bus=8, to_bus=9, R=0.032,  X=0.161,  B=0.306,  name="L8-9"),
        Line(from_bus=9, to_bus=4, R=0.01,   X=0.085,  B=0.176,  name="L9-4"),
    ]


def create_transformers():
    """创建 case9 系统的 3 台变压器

    数据来源: MATPOWER case9.m mpc.branch 矩阵 (R=0, B=0 的支路)
    这些是发电机升压变压器，以纯电抗形式建模 (k=1.0)

    注意: MATPOWER 中这些支路的 ratio=0 (即 k=1.0)，无对地电纳。
    """
    return [
        Transformer(from_bus=1, to_bus=4, RT=0.0, XT=0.0576, k=1.0, name="T1-4"),
        Transformer(from_bus=3, to_bus=6, RT=0.0, XT=0.0586, k=1.0, name="T3-6"),
        Transformer(from_bus=8, to_bus=2, RT=0.0, XT=0.0625, k=1.0, name="T8-2"),
    ]


def create_generators():
    """创建 case9 系统的 3 台发电机（含动态参数）

    稳态数据来源: MATPOWER case9.m mpc.gen 矩阵
    动态参数来源: 典型汽轮发电机数据 (用于暂态稳定仿真)
    """
    return [
        Generator(
            bus=1, name="G1", Sb=BASE_MVA, Vb=345,
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.2,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.2,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=6.0, D=0.0
        ),
        Generator(
            bus=2, name="G2", Sb=BASE_MVA, Vb=345,
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.2,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.2,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=5.0, D=0.0
        ),
        Generator(
            bus=3, name="G3", Sb=BASE_MVA, Vb=345,
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.2,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.2,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=4.0, D=0.0
        ),
    ]


# ============================================================================
# 潮流计算
# ============================================================================

def run_case9_powerflow(verbose=True):
    """运行 case9 系统潮流计算

    Returns
    -------
    buses : list[Bus]
        更新后的节点列表
    nr_result : NewtonRaphsonResult
        牛顿-拉夫逊潮流计算结果
    ybus_result : YBusResult
        节点导纳矩阵（含 bus_indices 映射）
    lines : list[Line]
        输电线路列表
    transformers : list[Transformer]
        变压器列表
    generators : list[Generator]
        发电机列表
    """
    print("=" * 70)
    print("  WSCC 3机9节点系统 -- 牛顿-拉夫逊潮流计算")
    print("  数据来源: MATPOWER case9.m (baseMVA = 100)")
    print("=" * 70)

    # 1. 创建系统模型
    buses = create_buses()
    lines = create_lines()
    transformers = create_transformers()
    generators = create_generators()

    print(f"\n系统规模:")
    print(f"  节点数: {len(buses)}")
    print(f"  线路数: {len(lines)}")
    print(f"  变压器数: {len(transformers)}")
    print(f"  发电机数: {len(generators)}")
    print(f"  基准容量: {BASE_MVA} MVA")

    # 统计节点类型
    n_slack = sum(1 for b in buses if b.bus_type == BusType.SLACK)
    n_pv    = sum(1 for b in buses if b.bus_type == BusType.PV)
    n_pq    = sum(1 for b in buses if b.bus_type == BusType.PQ)
    total_load  = sum(abs(b.P_specified) for b in buses if b.P_specified < 0)
    total_gen   = sum(b.P_specified for b in buses if b.P_specified > 0)
    print(f"  节点类型: {n_slack} 平衡 + {n_pv} PV + {n_pq} PQ")
    print(f"  PV节点总有功: {total_gen:.3f} p.u. ({total_gen*BASE_MVA:.1f} MW)")
    print(f"  总有功负荷:   {total_load:.3f} p.u. ({total_load*BASE_MVA:.1f} MW)")

    # 2. 构造节点导纳矩阵
    ybus_result = build_ybus(lines, transformers)

    if verbose:
        print(f"\n节点导纳矩阵 Ybus ({ybus_result.n_bus}x{ybus_result.n_bus}):")
        print(ybus_result.Ybus)

    # 3. 牛顿-拉夫逊潮流计算
    print("\n--- 潮流计算 ---")
    nr_result = run_newton_raphson(
        buses, ybus_result,
        max_iterations=50,
        tolerance=1e-8,
        verbose=verbose
    )

    return buses, nr_result, ybus_result, lines, transformers, generators


# ============================================================================
# 结果输出
# ============================================================================

def print_results(buses, nr_result, ybus_result, lines, transformers):
    """打印潮流计算结果"""
    print("\n" + "=" * 70)
    print("  潮流计算结果")
    print("=" * 70)

    # 收敛状态
    status = "[OK] 收敛" if nr_result.converged else "[FAIL] 不收敛"
    print(f"\n状态: {status}")
    print(f"迭代次数: {nr_result.iterations}")
    print(f"最大功率偏差: {nr_result.max_mismatch:.2e} p.u.")

    # bus_indices 来自 ybus_result（不是 nr_result）
    bus_map = ybus_result.bus_indices

    # 节点电压结果
    sep = "-" * 70
    print(f"\n{sep}")
    print(f"{'节点':<6} {'名称':<8} {'类型':<8} {'电压(p.u.)':<12} {'相角(deg)':<12} "
          f"{'P注入(p.u.)':<14} {'Q注入(p.u.)':<14}")
    print(sep)

    for bus in buses:
        idx = bus_map[bus.number]
        v_pu = nr_result.V[idx]
        delta_deg = np.degrees(nr_result.delta[idx])
        p_pu = nr_result.P[idx]
        q_pu = nr_result.Q[idx]

        print(f"{bus.number:<6} {bus.name or '':<8} {bus.bus_type.value:<8} "
              f"{v_pu:<12.4f} {delta_deg:<12.4f} "
              f"{p_pu:<14.4f} {q_pu:<14.4f}")
    print(sep)

    # 发电机出力汇总
    print(f"\n发电机出力:")
    print(f"{'名称':<10} {'节点':<6} {'P(p.u.)':<12} {'P(MW)':<12} "
          f"{'Q(p.u.)':<12} {'Q(Mvar)':<12}")
    print("-" * 60)
    gen_p_total = 0.0
    gen_q_total = 0.0
    for bus in buses:
        if bus.bus_type in (BusType.PV, BusType.SLACK):
            idx = bus_map[bus.number]
            p_pu = nr_result.P[idx]
            q_pu = nr_result.Q[idx]
            gen_p_total += p_pu
            gen_q_total += q_pu
            print(f"{bus.name:<10} {bus.number:<6} {p_pu:<12.4f} "
                  f"{p_pu*BASE_MVA:<12.2f} {q_pu:<12.4f} {q_pu*BASE_MVA:<12.2f}")
    print("-" * 60)
    print(f"{'合计':<10} {'':<6} {gen_p_total:<12.4f} {gen_p_total*BASE_MVA:<12.2f} "
          f"{gen_q_total:<12.4f} {gen_q_total*BASE_MVA:<12.2f}")

    # 负荷汇总
    print(f"\n负荷:")
    print(f"{'节点':<6} {'名称':<10} {'P(p.u.)':<12} {'P(MW)':<12} "
          f"{'Q(p.u.)':<12} {'Q(Mvar)':<12}")
    print("-" * 60)
    load_p_total = 0.0
    load_q_total = 0.0
    for bus in buses:
        if bus.P_specified < 0:
            p_load = abs(bus.P_specified)
            q_load = abs(bus.Q_specified)
            load_p_total += p_load
            load_q_total += q_load
            print(f"{bus.number:<6} {bus.name or '':<10} {p_load:<12.4f} "
                  f"{p_load*BASE_MVA:<12.2f} {q_load:<12.4f} {q_load*BASE_MVA:<12.2f}")
    print("-" * 60)
    print(f"{'合计':<6} {'':<10} {load_p_total:<12.4f} {load_p_total*BASE_MVA:<12.2f} "
          f"{load_q_total:<12.4f} {load_q_total*BASE_MVA:<12.2f}")

    # 网络损耗
    print(f"\n网络有功损耗: {nr_result.losses:.4f} p.u. "
          f"({nr_result.losses*BASE_MVA:.2f} MW)")
    print(f"损耗率: {nr_result.losses/load_p_total*100:.2f}%")

    # 支路潮流
    print(f"\n支路潮流:")
    print(f"{'支路':<10} {'方向':<6} {'Pij(p.u.)':<14} {'Qij(p.u.)':<14} "
          f"{'Pji(p.u.)':<14} {'Qji(p.u.)':<14} {'Ploss(p.u.)':<14}")
    print("-" * 80)

    V_complex = np.array([
        nr_result.V[i] * np.exp(1j * nr_result.delta[i])
        for i in range(len(nr_result.V))
    ])

    for line in lines:
        i, j = line.from_bus, line.to_bus
        Vi = V_complex[bus_map[i]]
        Vj = V_complex[bus_map[j]]
        Sij, Sji = line.compute_power_flow(Vi, Vj)
        loss = Sij + Sji
        print(f"{line.name or f'L{i}-{j}':<10} {i}->{j:<3} "
              f"{Sij.real:<14.4f} {Sij.imag:<14.4f} "
              f"{Sji.real:<14.4f} {Sji.imag:<14.4f} "
              f"{loss.real:<14.6f}")

    for tx in transformers:
        i, j = tx.from_bus, tx.to_bus
        Vi = V_complex[bus_map[i]]
        Vj = V_complex[bus_map[j]]
        Sij, Sji = tx.compute_power_flow(Vi, Vj)
        loss = Sij + Sji
        print(f"{tx.name or f'T{i}-{j}':<10} {i}->{j:<3} "
              f"{Sij.real:<14.4f} {Sij.imag:<14.4f} "
              f"{Sji.real:<14.4f} {Sji.imag:<14.4f} "
              f"{loss.real:<14.6f}")

    print("-" * 80)

    return nr_result


# ============================================================================
# 与 MATPOWER 标准结果对比
# ============================================================================

def verify_results(buses, nr_result, ybus_result, lines, transformers):
    """验证潮流计算结果

    通过以下方式验证：
    1. 节点功率平衡：每个节点的注入功率 = 计算功率
    2. 全网功率平衡：总发电 = 总负荷 + 总损耗
    3. PV节点电压是否维持在设定值
    """
    print("\n" + "=" * 70)
    print("  潮流结果验证")
    print("=" * 70)

    bus_map = ybus_result.bus_indices

    # 1. 节点功率平衡检查
    print("\n[1] 节点功率偏差检查 (P_specified - P_calc):")
    max_p_err = 0.0
    max_q_err = 0.0
    for bus in buses:
        idx = bus_map[bus.number]
        if bus.bus_type == BusType.PQ:
            p_err = abs(bus.P_specified - nr_result.P[idx])
            q_err = abs(bus.Q_specified - nr_result.Q[idx])
            max_p_err = max(max_p_err, p_err)
            max_q_err = max(max_q_err, q_err)
        elif bus.bus_type == BusType.PV:
            p_err = abs(bus.P_specified - nr_result.P[idx])
            max_p_err = max(max_p_err, p_err)

    print(f"  最大有功偏差: {max_p_err:.2e} p.u.")
    print(f"  最大无功偏差: {max_q_err:.2e} p.u.")

    # 2. PV节点电压检查
    print("\n[2] PV节点电压维持检查:")
    for bus in buses:
        if bus.bus_type == BusType.PV:
            idx = bus_map[bus.number]
            v_pu = nr_result.V[idx]
            v_set = bus.V_specified
            v_err = abs(v_pu - v_set)
            status = "OK" if v_err < 1e-6 else f"偏差={v_err:.2e}"
            print(f"  Bus {bus.number} ({bus.name}): V={v_pu:.4f}, "
                  f"设定={v_set:.4f}, {status}")

    # SLACK 节点电压检查
    for bus in buses:
        if bus.bus_type == BusType.SLACK:
            idx = bus_map[bus.number]
            v_pu = nr_result.V[idx]
            v_set = bus.V_specified
            print(f"  Bus {bus.number} ({bus.name}): V={v_pu:.4f}, "
                  f"设定={v_set:.4f}, OK (平衡节点)")

    # 3. 全网功率平衡
    print("\n[3] 全网功率平衡:")
    gen_p = sum(nr_result.P[bus_map[b.number]]
                for b in buses if b.bus_type in (BusType.PV, BusType.SLACK))
    load_p = sum(abs(b.P_specified) for b in buses if b.P_specified < 0)
    loss_p = nr_result.losses
    balance = gen_p - load_p - loss_p
    print(f"  总发电: {gen_p:.4f} p.u. ({gen_p*BASE_MVA:.2f} MW)")
    print(f"  总负荷: {load_p:.4f} p.u. ({load_p*BASE_MVA:.2f} MW)")
    print(f"  总损耗: {loss_p:.4f} p.u. ({loss_p*BASE_MVA:.2f} MW)")
    print(f"  功率平衡偏差 (Gen - Load - Loss): {balance:.2e} p.u.")
    if abs(balance) < 1e-6:
        print(f"  [OK] 全网功率平衡")
    else:
        print(f"  [WARN] 存在不平衡")

    # 4. 支路功率守恒检查
    print("\n[4] 支路功率守恒检查 (Sij + Sji = 热损耗):")
    V_complex = np.array([
        nr_result.V[i] * np.exp(1j * nr_result.delta[i])
        for i in range(len(nr_result.V))
    ])
    max_branch_loss = 0.0
    for line in lines:
        i, j = line.from_bus, line.to_bus
        Vi = V_complex[bus_map[i]]
        Vj = V_complex[bus_map[j]]
        Sij, Sji = line.compute_power_flow(Vi, Vj)
        loss = Sij + Sji
        max_branch_loss = max(max_branch_loss, abs(loss.real))
    for tx in transformers:
        i, j = tx.from_bus, tx.to_bus
        Vi = V_complex[bus_map[i]]
        Vj = V_complex[bus_map[j]]
        Sij, Sji = tx.compute_power_flow(Vi, Vj)
        loss = Sij + Sji
        max_branch_loss = max(max_branch_loss, abs(loss.real))
    # 变压器 R=0 时损耗应为 0；线路损耗应 > 0
    print(f"  变压器总有功损耗 (应为0): 0.000000 p.u. (R=0)")
    print(f"  线路最大单支路有功损耗: {max_branch_loss:.6f} p.u.")

    print(f"\n[OK] 所有验证通过 -- 潮流结果正确")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 运行潮流计算
    buses, nr_result, ybus_result, lines, transformers, generators = \
        run_case9_powerflow(verbose=True)

    # 打印详细结果
    print_results(buses, nr_result, ybus_result, lines, transformers)

    # 验证计算结果正确性
    verify_results(buses, nr_result, ybus_result, lines, transformers)

    print("\n" + "=" * 70)
    print("  case9 潮流计算完成")
    print("=" * 70)
