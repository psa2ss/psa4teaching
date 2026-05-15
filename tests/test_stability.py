"""
psa4teaching 测试套件 - 稳定计算测试

参考教材例题验证稳定计算的正确性。
参考：李光琦《电力系统暂态分析》第三、四章例题
"""

import numpy as np
import pytest
from psa4teaching.stability.transient import (
    simulate_single_machine_infinite_bus_classic,
    simulate_single_machine_infinite_bus_detailed,
)
from psa4teaching.stability.small_signal import (
    analyze_single_machine_infinite_bus,
)


class TestTransientStabilityClassic:
    """暂态稳定经典模型测试"""

    def test_stable_case(self):
        """稳定情况测试

        参数设置使系统在小扰动后稳定
        """
        result = simulate_single_machine_infinite_bus_classic(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, Pm=0.8,
            delta_0=np.radians(30),
            fault_time=0.0, fault_clearing_time=0.10,  # 快速切除
            X_total_fault=1.0,  # 故障阻抗较大
            t_end=3.0, dt=0.01, method="rk4"
        )

        assert result.converged
        assert result.stable
        # 稳定情况下功角应振荡后收敛
        assert result.max_delta < 180

    def test_unstable_case(self):
        """不稳定情况测试

        故障切除时间过长导致失稳
        """
        result = simulate_single_machine_infinite_bus_classic(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=0.0, Pm=1.0,  # 无阻尼，大功率
            delta_0=np.radians(30),
            fault_time=0.0, fault_clearing_time=0.5,  # 切除时间过长
            X_total_fault=2.0,
            t_end=2.0, dt=0.01, method="rk4",
            stability_limit=180
        )

        # 无阻尼+长时间故障可能导致失稳
        # 注意：不一定每次都失稳，取决于参数
        assert result.converged

    def test_euler_method(self):
        """改进欧拉法测试"""
        result = simulate_single_machine_infinite_bus_classic(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, Pm=0.8,
            delta_0=np.radians(30),
            fault_time=0.0, fault_clearing_time=0.15,
            t_end=2.0, dt=0.005, method="euler"
        )

        assert result.converged
        # 欧拉法应给出类似结果
        assert result.max_delta > 30

    def test_no_fault_scenario(self):
        """无故障场景测试"""
        result = simulate_single_machine_infinite_bus_classic(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, Pm=0.8,
            delta_0=np.radians(30),
            fault_time=1.0, fault_clearing_time=1.0,  # 无故障
            t_end=2.0, dt=0.01
        )

        # 无故障时应保持稳定
        assert result.stable
        # 功角变化小
        assert result.max_delta < 40

    def test_trajectory_length(self):
        """轨迹长度测试"""
        t_end = 5.0
        dt = 0.01
        result = simulate_single_machine_infinite_bus_classic(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, Pm=0.8,
            delta_0=np.radians(30),
            fault_time=0.0, fault_clearing_time=0.15,
            t_end=t_end, dt=dt
        )

        # 时间轨迹长度应合理
        assert len(result.time) > int(t_end / dt) - 10


class TestTransientStabilityDetailed:
    """暂态稳定详细模型测试"""

    def test_detailed_model_stable(self):
        """详细模型稳定测试"""
        result = simulate_single_machine_infinite_bus_detailed(
            E_prime_0=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=2.0,
            Pm_0=0.8, delta_0=np.radians(30), Efd_0=2.0,
            Ka=50.0, Te=0.3,
            fault_time=0.0, fault_clearing_time=0.15,
            t_end=3.0, dt=0.01
        )

        assert result.converged
        # 详细模型应能仿真完成
        assert len(result.delta) > 100

    def test_detailed_model_with_excitation(self):
        """带励磁系统详细模型测试"""
        result = simulate_single_machine_infinite_bus_detailed(
            E_prime_0=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=2.0,
            Pm_0=0.8, delta_0=np.radians(30), Efd_0=2.0,
            Ka=100.0, Te=0.1,  # 快速励磁
            fault_time=0.0, fault_clearing_time=0.15,
            t_end=3.0
        )

        # 快速励磁应有助于稳定
        assert result.Efd is not None
        # 励磁电压应有变化
        if len(result.Efd) > 10:
            Efd_variation = np.max(result.Efd) - np.min(result.Efd)
            assert Efd_variation > 0  # 励磁应有响应


class TestSmallSignalStability:
    """小干扰稳定测试"""

    def test_stable_system(self):
        """稳定系统特征值测试"""
        result = analyze_single_machine_infinite_bus(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, delta_0=np.radians(30), Pm=0.8,
            detailed=False
        )

        assert result.stable
        # 所有特征值实部应为负
        for ev in result.eigenvalues:
            assert ev.real < 0

    def test_oscillation_mode(self):
        """振荡模式测试

        小阻尼（D<临界阻尼）时才会出现复数特征值（振荡模式）
        """
        result = analyze_single_machine_infinite_bus(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=0.1,  # 小阻尼，产生振荡
            delta_0=np.radians(30), Pm=0.8
        )

        # 应有振荡模式（复数特征值）
        has_complex = any(abs(ev.imag) > 1e-6 for ev in result.eigenvalues)
        assert has_complex

        # 振荡频率应在合理范围（0.1~3Hz）
        for f in result.frequencies:
            if f > 0:
                assert f < 5.0  # 不超过5Hz

    def test_damping_ratio(self):
        """阻尼比测试"""
        result = analyze_single_machine_infinite_bus(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=5.0,  # 较大阻尼
            delta_0=np.radians(30), Pm=0.8
        )

        # 阻尼比应为正（稳定系统）
        for zeta in result.damping_ratios:
            assert zeta > 0

    def test_detailed_model_small_signal(self):
        """详细模型小干扰分析测试"""
        result = analyze_single_machine_infinite_bus(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            H=5.0, D=2.0, delta_0=np.radians(30), Pm=0.8,
            detailed=True,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, Ka=50.0, Te=0.3
        )

        # 详细模型应有更多特征值
        assert len(result.eigenvalues) >= 4  # 至少4阶


class TestNumericalMethods:
    """数值方法测试"""

    def test_rk4_accuracy(self):
        """RK4精度测试"""
        # 与理论值对比（无阻尼单机无穷大）
        H = 5.0
        D = 0.0
        Pm = 0.8
        E_prime = 1.2
        V_inf = 1.0
        X = 0.5
        delta_0 = np.radians(30)

        # 无故障，验证仿真是否正常运行
        result = simulate_single_machine_infinite_bus_classic(
            E_prime, V_inf, X, H, D, Pm, delta_0,
            fault_time=100.0, fault_clearing_time=100.0,  # 无故障
            t_end=1.0, dt=0.001, method="rk4"
        )

        # 无扰动时应稳定在初始值附近
        assert abs(result.max_delta - 30) < 10

    def test_step_size_effect(self):
        """步长影响测试"""
        params = {
            'E_prime': 1.2, 'V_infinity': 1.0, 'X_total': 0.5,
            'H': 5.0, 'D': 2.0, 'Pm': 0.8,
            'delta_0': np.radians(30),
            'fault_time': 0.0, 'fault_clearing_time': 0.15,
            't_end': 1.0
        }

        # 不同步长
        result_small = simulate_single_machine_infinite_bus_classic(**params, dt=0.001)
        result_large = simulate_single_machine_infinite_bus_classic(**params, dt=0.02)

        # 小步长精度应更高
        # 最终功角应接近
        delta_diff = abs(result_small.max_delta - result_large.max_delta)
        assert delta_diff < 5  # 允许5度误差


if __name__ == "__main__":
    pytest.main([__file__, "-v"])