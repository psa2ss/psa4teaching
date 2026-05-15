"""
psa4teaching 测试套件 - 元件模型测试
"""

import numpy as np
import pytest
from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator, GeneratorModelType
from psa4teaching.models.load import Load, LoadModel


class TestBus:
    """节点模型测试"""

    def test_slack_bus_init(self):
        bus = Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.05)
        assert bus.V == 1.05
        assert bus.delta == 0.0
        assert bus.bus_type == BusType.SLACK

    def test_pv_bus_init(self):
        bus = Bus(number=2, name="Gen", bus_type=BusType.PV,
                  P_specified=0.5, V_specified=1.02)
        assert bus.V == 1.02
        assert bus.P_specified == 0.5

    def test_pq_bus_init(self):
        bus = Bus(number=3, name="Load", bus_type=BusType.PQ,
                  P_specified=-0.8, Q_specified=-0.3)
        assert bus.V == 1.0  # 默认1.0
        assert bus.P_specified == -0.8
        assert bus.Q_specified == -0.3

    def test_active_power_mismatch(self):
        bus = Bus(number=1, bus_type=BusType.PQ, P_specified=0.5)
        bus.P = 0.3
        assert abs(bus.get_active_power_mismatch() - 0.2) < 1e-10

    def test_reactive_power_mismatch_pq(self):
        bus = Bus(number=1, bus_type=BusType.PQ, Q_specified=0.3)
        bus.Q = 0.1
        assert abs(bus.get_reactive_power_mismatch() - 0.2) < 1e-10

    def test_reactive_power_mismatch_pv(self):
        bus = Bus(number=1, bus_type=BusType.PV)
        assert bus.get_reactive_power_mismatch() == 0.0

    def test_slack_mismatch_zero(self):
        bus = Bus(number=1, bus_type=BusType.SLACK)
        assert bus.get_active_power_mismatch() == 0.0

    def test_q_limit_check(self):
        bus = Bus(number=1, bus_type=BusType.PV, Q_min=-1.0, Q_max=1.0)
        bus.Q = -1.5
        assert bus.check_q_limit() == True
        bus.Q = 0.5
        assert bus.check_q_limit() == False


class TestLine:
    """输电线路模型测试"""

    def test_line_impedance(self):
        line = Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02)
        Z = line.Z
        assert abs(Z.real - 0.02) < 1e-10
        assert abs(Z.imag - 0.1) < 1e-10

    def test_series_admittance(self):
        line = Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)
        Y = line.get_series_admittance()
        # Y = 1/j0.1 = -j10
        assert abs(Y.real) < 1e-10
        assert abs(Y.imag + 10.0) < 1e-10

    def test_shunt_admittance(self):
        line = Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.04)
        Y_shunt = line.get_shunt_admittance()
        assert abs(Y_shunt.imag - 0.02) < 1e-10  # B/2 = 0.02

    def test_ybus_contribution(self):
        line = Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)
        contrib = line.get_ybus_contribution()
        # 应有4个贡献: (1,1), (2,2), (1,2), (2,1)
        assert len(contrib) == 4

    def test_line_validation(self):
        with pytest.raises(ValueError):
            Line(from_bus=1, to_bus=2, R=0.02, X=-0.1, B=0.02)

    def test_power_flow(self):
        """测试线路功率计算"""
        line = Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)
        Vi = complex(1.0, 0)
        Vj = complex(np.cos(np.radians(-5)), np.sin(np.radians(-5)))
        Sij, Sji = line.compute_power_flow(Vi, Vj)
        # 有功应从电压高的一端流向电压低的一端
        assert Sij.real > 0  # 有功从i流向j
        assert Sji.real < 0  # j端吸收


class TestTransformer:
    """变压器模型测试"""

    def test_standard_ratio(self):
        tx = Transformer(from_bus=1, to_bus=2, RT=0.0, XT=0.1, k=1.0)
        contrib = tx.get_ybus_contribution()
        # k=1时，等效为简单串联阻抗
        assert len(contrib) == 4

    def test_nonstandard_ratio(self):
        tx = Transformer(from_bus=1, to_bus=2, RT=0.0, XT=0.1, k=1.05)
        contrib = tx.get_ybus_contribution()
        assert len(contrib) == 4

    def test_transformer_validation(self):
        with pytest.raises(ValueError):
            Transformer(from_bus=1, to_bus=2, RT=0.002, XT=0.1, k=-1.0)


class TestGenerator:
    """发电机模型测试"""

    def test_classic_model(self):
        gen = Generator(bus=1, Xd_prime=0.3, H=6.0, D=1.0,
                       model_type=GeneratorModelType.CLASSIC)
        assert gen.get_transient_reactance() == 0.3
        assert gen.get_inertia_constant() == 6.0

    def test_subtransient_emf(self):
        """测试次暂态电势计算"""
        gen = Generator(bus=1, Xd_doubleprime=0.2)
        V = complex(1.0, 0)
        I = complex(0.5, -0.3)
        E_pp = gen.compute_subtransient_emf(V, I)
        # E" = V + jXd"×I = 1.0 + j0.2×(0.5-j0.3) = 1.0 + j0.1 + 0.06 = 1.06 + j0.1
        assert abs(E_pp.real - 1.06) < 1e-10
        assert abs(E_pp.imag - 0.1) < 1e-10

    def test_transient_emf(self):
        """测试暂态电势计算"""
        gen = Generator(bus=1, Xd_prime=0.3)
        V = complex(1.0, 0)
        I = complex(0.5, -0.3)
        E_p = gen.compute_transient_emf(V, I)
        # E' = V + jXd'×I = 1.0 + j0.3×(0.5-j0.3) = 1.0 + j0.15 + 0.09 = 1.09 + j0.15
        assert abs(E_p.real - 1.09) < 1e-10
        assert abs(E_p.imag - 0.15) < 1e-10


class TestLoad:
    """负荷模型测试"""

    def test_constant_power(self):
        load = Load(bus=1, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_POWER)
        P, Q = load.get_power_at_voltage(0.9)
        assert abs(P - 0.8) < 1e-10  # 恒功率
        assert abs(Q - 0.3) < 1e-10

    def test_constant_impedance(self):
        load = Load(bus=1, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_IMPEDANCE)
        P, Q = load.get_power_at_voltage(0.9)
        assert abs(P - 0.8 * 0.9**2) < 1e-10
        assert abs(Q - 0.3 * 0.9**2) < 1e-10

    def test_constant_current(self):
        load = Load(bus=1, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_CURRENT)
        P, Q = load.get_power_at_voltage(0.9)
        assert abs(P - 0.8 * 0.9) < 1e-10
        assert abs(Q - 0.3 * 0.9) < 1e-10

    def test_zip_model(self):
        load = Load(bus=1, P0=1.0, Q0=0.5, model_type=LoadModel.ZIP,
                    alpha_p=0.3, beta_p=0.3, gamma_p=0.4,
                    alpha_q=0.3, beta_q=0.3, gamma_q=0.4)
        P, Q = load.get_power_at_voltage(1.0)
        assert abs(P - 1.0) < 1e-10  # V=V0时P=P0
        assert abs(Q - 0.5) < 1e-10

    def test_load_admittance(self):
        load = Load(bus=1, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_IMPEDANCE)
        Y = load.get_admittance(V0=1.0)
        # Y = (P - jQ) / V² = (0.8 - j0.3) / 1.0
        assert abs(Y.real - 0.8) < 1e-10
        assert abs(Y.imag + 0.3) < 1e-10

    def test_zip_coefficient_validation(self):
        with pytest.raises(ValueError):
            Load(bus=1, P0=1.0, Q0=0.5, model_type=LoadModel.ZIP,
                 alpha_p=0.3, beta_p=0.3, gamma_p=0.3)  # 总和不为1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])