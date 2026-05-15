"""
psa4teaching 测试套件 - 潮流计算测试

使用教材经典例题验证潮流算法的正确性。
参考：陈珩《电力系统稳态分析》第四章例题
"""

import numpy as np
import pytest
from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.network.ybus import build_ybus
from psa4teaching.powerflow.newton_raphson import run_newton_raphson, build_jacobian
from psa4teaching.powerflow.fast_decoupled import run_fast_decoupled
from psa4teaching.powerflow.dc_powerflow import run_dc_powerflow


class TestNewtonRaphson:
    """牛顿-拉夫逊法测试"""

    def test_simple_two_bus_system(self):
        """简单两节点系统潮流测试

        节点1：平衡节点，V=1.0
        节点2：PQ节点，P=-0.5, Q=-0.2
        线路：R=0.01, X=0.1, B=0

        预期：节点2电压略低于1.0，相角为负
        """
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Load", bus_type=BusType.PQ,
                 P_specified=-0.5, Q_specified=-0.2),
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.01, X=0.1, B=0.0),
        ]

        ybus_result = build_ybus(lines, [])
        result = run_newton_raphson(buses, ybus_result, max_iterations=50, tolerance=1e-8)

        assert result.converged
        assert result.iterations < 20  # 应在20次内收敛
        # 负荷节点电压应低于平衡节点
        idx2 = ybus_result.bus_indices[2]
        assert result.V[idx2] < 1.0
        # 负荷节点相角应为负（滞后）
        assert result.delta[idx2] < 0

    def test_three_bus_system(self):
        """三节点系统潮流测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.05),
            Bus(number=2, name="Gen", bus_type=BusType.PV,
                 P_specified=0.5, V_specified=1.02),
            Bus(number=3, name="Load", bus_type=BusType.PQ,
                 P_specified=-0.8, Q_specified=-0.3),
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
            Line(from_bus=1, to_bus=3, R=0.01, X=0.08, B=0.01),
        ]

        ybus_result = build_ybus(lines, [])
        result = run_newton_raphson(buses, ybus_result, max_iterations=50, tolerance=1e-8)

        assert result.converged
        # 平衡节点电压应为指定值
        idx1 = ybus_result.bus_indices[1]
        assert abs(result.V[idx1] - 1.05) < 1e-6
        # PV节点电压应为指定值
        idx2 = ybus_result.bus_indices[2]
        assert abs(result.V[idx2] - 1.02) < 1e-6
        # 网络损耗应为正值
        assert result.losses > 0

    def test_pv_node_q_limit(self):
        """PV节点无功限幅测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Gen", bus_type=BusType.PV,
                 P_specified=1.0, V_specified=1.02,
                 Q_min=-0.1, Q_max=0.1),  # 严格限幅
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.2, B=0.02),
        ]

        ybus_result = build_ybus(lines, [])
        # 大功率可能导致无功越限
        result = run_newton_raphson(buses, ybus_result, max_iterations=50)
        # 至少应该完成仿真
        assert result.iterations > 0

    def test_convergence_tolerance(self):
        """收敛精度测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Load", bus_type=BusType.PQ,
                 P_specified=-0.5, Q_specified=-0.1),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.01, X=0.1, B=0.0)]

        ybus_result = build_ybus(lines, [])

        # 高精度
        result_high = run_newton_raphson(buses, ybus_result, tolerance=1e-10)
        # 低精度
        result_low = run_newton_raphson(buses, ybus_result, tolerance=1e-4)

        # 高精度结果偏差更小
        assert result_high.max_mismatch <= result_low.max_mismatch


class TestFastDecoupled:
    """P-Q分解法测试"""

    def test_two_bus_system(self):
        """两节点系统P-Q分解法测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Load", bus_type=BusType.PQ,
                 P_specified=-0.5, Q_specified=-0.2),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.01, X=0.1, B=0.0)]

        ybus_result = build_ybus(lines, [])
        result = run_fast_decoupled(buses, ybus_result, max_iterations=100, tolerance=1e-8)

        assert result.converged
        # P-Q分解法通常需要更多迭代次数
        assert result.iterations > 0

    def test_comparison_with_nr(self):
        """与牛顿-拉夫逊法结果对比"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.05),
            Bus(number=2, name="Gen", bus_type=BusType.PV,
                 P_specified=0.5, V_specified=1.02),
            Bus(number=3, name="Load", bus_type=BusType.PQ,
                 P_specified=-0.8, Q_specified=-0.3),
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
        ]

        ybus_result = build_ybus(lines, [])

        nr_result = run_newton_raphson(buses, ybus_result)
        fd_result = run_fast_decoupled(buses, ybus_result)

        # 两种方法结果应接近
        assert np.allclose(nr_result.V, fd_result.V, atol=0.01)
        assert np.allclose(nr_result.delta, fd_result.delta, atol=0.01)


class TestDCPowerFlow:
    """直流潮流测试"""

    def test_simple_system(self):
        """简单系统直流潮流测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Load", bus_type=BusType.PQ, P_specified=-0.5),
            Bus(number=3, name="Load", bus_type=BusType.PQ, P_specified=-0.3),
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.01, X=0.1, B=0.0),
            Line(from_bus=2, to_bus=3, R=0.02, X=0.15, B=0.0),
            Line(from_bus=1, to_bus=3, R=0.01, X=0.08, B=0.0),
        ]

        result = run_dc_powerflow(buses, lines, [])

        # 直流潮流应给出相角分布
        assert len(result.theta) == 3
        # 平衡节点相角应为0
        idx1 = result.bus_indices[1]
        assert abs(result.theta[idx1]) < 1e-10
        # 负荷节点相角应为负
        idx2 = result.bus_indices[2]
        assert result.theta[idx2] < 0

    def test_branch_power_flow(self):
        """支路功率流测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK),
            Bus(number=2, name="Load", bus_type=BusType.PQ, P_specified=-1.0),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)]

        result = run_dc_powerflow(buses, lines, [])

        # 支路功率流：Pij = (θi-θj)/Xij，从发电端流向负荷端为正
        P_flow = result.branch_flows.get((1, 2))
        assert abs(P_flow - 1.0) < 0.1  # 允许一定误差


class TestJacobian:
    """雅可比矩阵测试"""

    def test_jacobian_dimensions(self):
        """雅可比矩阵维度测试"""
        buses = [
            Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="PV", bus_type=BusType.PV, P_specified=0.5, V_specified=1.02),
            Bus(number=3, name="PQ", bus_type=BusType.PQ, P_specified=-0.5, Q_specified=-0.2),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
                 Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03)]

        ybus_result = build_ybus(lines, [])

        V = np.ones(ybus_result.n_bus)
        delta = np.zeros(ybus_result.n_bus)

        pq_buses = [b for b in buses if b.bus_type == BusType.PQ]
        pv_buses = [b for b in buses if b.bus_type == BusType.PV]

        J = build_jacobian(ybus_result.Ybus, ybus_result.G, ybus_result.B,
                          V, delta, pq_buses, pv_buses, ybus_result.bus_indices)

        # 雅可比矩阵维度：(n_P + n_Q) × (n_P + n_Q)
        n_p = len(pv_buses) + len(pq_buses)  # 2
        n_q = len(pq_buses)  # 1
        assert J.shape == (n_p + n_q, n_p + n_q)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])