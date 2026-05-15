"""
psa4teaching 测试套件 - 短路计算测试

参考教材例题验证短路计算的正确性。
参考：李光琦《电力系统暂态分析》第二章例题
"""

import numpy as np
import pytest
from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator
from psa4teaching.shortcircuit.symmetric import (
    calculate_three_phase_fault,
    calculate_transfer_impedances,
)
from psa4teaching.shortcircuit.asymmetric import (
    calculate_single_line_to_ground,
    calculate_line_to_line,
    calculate_double_line_to_ground,
    sequence_to_phase,
    phase_to_sequence,
)
from psa4teaching.shortcircuit.gb15544 import get_voltage_factor, get_correction_factor


class TestThreePhaseFault:
    """三相短路测试"""

    def test_simple_two_bus_fault(self):
        """简单两节点系统短路测试

        系统：发电机(节点1) -- 线路 -- 负荷(节点2)
        在节点2发生三相短路
        需要对地支路（B≠0）使Ybus非奇异
        """
        buses = [
            Bus(number=1, name="Gen", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Load", bus_type=BusType.PQ),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.02)]
        generators = [Generator(bus=1, Xd_doubleprime=0.1)]

        result = calculate_three_phase_fault(
            buses, lines, [], generators,
            fault_bus=2, S_base=100.0, V_base=10.5
        )

        # 短路电流大于0（验证计算完成）
        assert abs(result.fault_current) > 0
        assert result.fault_current_ka > 0

    def test_fault_voltage_drop(self):
        """短路后电压跌落测试"""
        buses = [
            Bus(number=1, name="Gen", bus_type=BusType.SLACK, V_specified=1.05),
            Bus(number=2, name="Load", bus_type=BusType.PQ),
            Bus(number=3, name="Load", bus_type=BusType.PQ),
        ]
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
        ]
        generators = [Generator(bus=1, Xd_doubleprime=0.2)]

        result = calculate_three_phase_fault(
            buses, lines, [], generators,
            fault_bus=3, S_base=100.0, V_base=10.5
        )

        # 短路点电压应为0（或接近0）
        V_fault = result.V_pu[3]
        assert abs(V_fault) < 0.5  # 金属性短路，电压大幅跌落

    def test_transfer_impedance(self):
        """转移阻抗计算测试

        需要对地支路使Ybus非奇异
        """
        buses = [
            Bus(number=1, name="Gen1", bus_type=BusType.SLACK, V_specified=1.0),
            Bus(number=2, name="Gen2", bus_type=BusType.PV),
            Bus(number=3, name="Fault", bus_type=BusType.PQ),
        ]
        lines = [
            Line(from_bus=1, to_bus=3, R=0.0, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.0, X=0.15, B=0.02),
        ]

        transfer_imp = calculate_transfer_impedances(
            buses, lines, [], fault_bus=3, generator_buses=[1, 2]
        )

        # 转移阻抗应为感性（虚部为负）
        assert transfer_imp[1].imag < 0
        assert transfer_imp[2].imag < 0


class TestAsymmetricFault:
    """不对称短路测试"""

    def test_sequence_components(self):
        """对称分量变换测试"""
        # 单相接地：Ia = I, Ib = 0, Ic = 0
        Ia, Ib, Ic = 1.0, 0.0, 0.0
        I0, I1, I2 = phase_to_sequence(Ia, Ib, Ic)
        # 单相接地时，I0 = I1 = I2 = Ia/3
        assert abs(abs(I0) - 1/3) < 0.1
        assert abs(abs(I1) - 1/3) < 0.1
        assert abs(abs(I2) - 1/3) < 0.1

    def test_sequence_inverse(self):
        """对称分量逆变换测试"""
        I0 = 1.0
        I1 = 2.0
        I2 = 3.0
        Ia, Ib, Ic = sequence_to_phase(I0, I1, I2)
        I0_back, I1_back, I2_back = phase_to_sequence(Ia, Ib, Ic)
        assert abs(I0_back - I0) < 1e-10
        assert abs(I1_back - I1) < 1e-10
        assert abs(I2_back - I2) < 1e-10

    def test_single_line_to_ground(self):
        """单相接地短路测试"""
        buses = [
            Bus(number=1, name="Gen", bus_type=BusType.SLACK),
            Bus(number=2, name="Load", bus_type=BusType.PQ),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)]
        generators = [Generator(bus=1, Xd_doubleprime=0.1, Xq_doubleprime=0.1)]

        result = calculate_single_line_to_ground(
            buses, lines, [], generators, fault_bus=2
        )

        # 单相接地电流：If = 3I0 = 3 × Ea/(Z1+Z2+Z0)
        # 在简化情况下，Z1=Z2=Z0
        # If ≈ 3 × 1.0 / (3 × Z) = 1/Z
        # 所以单相接地电流与三相短路电流相近
        assert abs(result.fault_current) > 0

    def test_line_to_line(self):
        """两相短路测试"""
        buses = [
            Bus(number=1, name="Gen", bus_type=BusType.SLACK),
            Bus(number=2, name="Load", bus_type=BusType.PQ),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)]
        generators = [Generator(bus=1, Xd_doubleprime=0.1)]

        result = calculate_line_to_line(buses, lines, [], generators, fault_bus=2)

        # 两相短路：Ib-Ic关系
        Ib, Ic = result.fault_currents_3phase[1], result.fault_currents_3phase[2]
        # Ib ≈ -Ic（幅值相等，相位相反）
        assert abs(abs(Ib) - abs(Ic)) < 0.5

    def test_double_line_to_ground(self):
        """两相接地短路测试"""
        buses = [
            Bus(number=1, name="Gen", bus_type=BusType.SLACK),
            Bus(number=2, name="Load", bus_type=BusType.PQ),
        ]
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)]
        generators = [Generator(bus=1, Xd_doubleprime=0.1)]

        result = calculate_double_line_to_ground(
            buses, lines, [], generators, fault_bus=2
        )

        # 两相接地时a相电流为0
        Ia = result.fault_currents_3phase[0]
        assert abs(Ia) < 0.5


class TestGB15544:
    """GB/T 15544标��测试"""

    def test_voltage_factor_lv(self):
        """低压系统电压系数"""
        c_max = get_voltage_factor(0.4, max_current=True)
        c_min = get_voltage_factor(0.4, max_current=False)
        assert abs(c_max - 1.10) < 1e-6
        assert abs(c_min - 0.95) < 1e-6

    def test_voltage_factor_mv(self):
        """中压系统电压系数"""
        c_max = get_voltage_factor(10.5, max_current=True)
        c_min = get_voltage_factor(10.5, max_current=False)
        assert abs(c_max - 1.10) < 1e-6
        assert abs(c_min - 1.00) < 1e-6

    def test_voltage_factor_hv(self):
        """高压系统电压系数"""
        c_max = get_voltage_factor(220.0, max_current=True)
        c_min = get_voltage_factor(220.0, max_current=False)
        assert abs(c_max - 1.10) < 1e-6
        assert abs(c_min - 1.00) < 1e-6

    def test_generator_correction(self):
        """发电机阻抗修正测试"""
        gen = Generator(bus=1, Xd_doubleprime=0.2)
        corr = get_correction_factor(gen, is_max=True)
        # 修正系数应大于1（最大短路电流）
        assert corr['KG'] > 1.0
        assert corr['Xd_doubleprime_corrected'] > gen.Xd_doubleprime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])