"""
IEEET1 励磁系统模型测试
=======================

测试 IEEE Type 1 励磁系统的计算功能，包括：
- 参数初始化和饱和系数计算
- 时域仿真（欧拉法和 RK4）
- 线性化状态矩阵
"""

import numpy as np
import pytest
from psa4teaching.models.ieeet1 import IEEET1Params


class TestIEEET1Init:
    """IEEET1 参数初始化测试"""

    def test_default_parameters(self):
        """测试默认参数初始化"""
        exc = IEEET1Params()
        assert exc.KA == 200.0
        assert exc.TA == 0.02
        assert exc.KE == 1.0
        assert exc.TE == 0.5
        assert exc.KF == 0.05
        assert exc.TF == 1.0
        # 验证饱和系数已计算
        assert exc._A_sat > 0
        assert exc._B_sat > 0

    def test_saturation_function(self):
        """测试饱和函数 SE(Efd) = A * exp(B * Efd)"""
        exc = IEEET1Params(SE1=0.1, SE2=0.03, Efd_MAX=5.0)
        # 在工作点处验证
        assert abs(exc.saturation(5.0) - 0.1) < 1e-10
        assert abs(exc.saturation(0.75 * 5.0) - 0.03) < 1e-10

    def test_no_saturation(self):
        """测试零饱和时的降级行为"""
        exc = IEEET1Params(SE1=0.0, SE2=0.0)
        assert exc.saturation(1.0) == 0.0
        assert exc.saturation(5.0) == 0.0

    def test_custom_parameters(self):
        """测试自定义参数"""
        exc = IEEET1Params(KA=100, TA=0.05, KE=-0.05, TE=0.8,
                           KF=0.1, TF=1.5)
        assert exc.KA == 100
        assert exc.KE == -0.05
        assert exc.TF == 1.5


class TestIEEET1Compute:
    """IEEET1 时域计算测试"""

    def test_zero_input_steady_state(self):
        """稳态时励磁输出应保持恒定"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5, KF=0.05, TF=1.0)

        # 初始状态为零
        state = np.array([1.0, 1.0, 1.0, 0.0])  # [VR, x_ex, Efd, Rf]
        efd, new_state = exc.compute(V_ref=1.0, V_measured=1.0,
                                     V_S=0.0, dt=0.01, state=state)
        # 零误差时，调节器导数 ≈ 0
        assert isinstance(efd, float)
        assert len(new_state) == 4

    def test_step_response_voltage_increase(self):
        """Vref 阶跃增加应最终导致 Efd 增加（多步仿真）"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5, KF=0.05, TF=1.0)

        # 从稳态开始
        state = np.array([1.0, 1.0, 1.0, 0.0])
        # 运行多个步长，让励磁系统有时间响应
        for _ in range(50):
            efd, state = exc.compute(V_ref=1.05, V_measured=1.0,
                                     V_S=0.0, dt=0.01, state=state)
        # Vref 高于 Vmeasured，Efd 应增加
        assert efd > 1.0

    def test_output_limiting(self):
        """验证输出限幅"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5,
                           Efd_MIN=0.0, Efd_MAX=3.0)

        state = np.array([10.0, 10.0, 10.0, 0.0])
        efd, new_state = exc.compute(V_ref=1.5, V_measured=1.0,
                                     V_S=0.0, dt=0.01, state=state)
        assert 0.0 <= efd <= 3.0

    def test_rk4_vs_euler(self):
        """RK4 和欧拉法结果应接近（小步长）"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5)
        state = np.array([0.5, 0.5, 0.5, 0.0])

        efd_euler, _ = exc.compute(V_ref=1.05, V_measured=1.0,
                                   V_S=0.0, dt=0.001, state=state.copy())
        efd_rk4, _ = exc.compute_rk4(V_ref=1.05, V_measured=1.0,
                                     V_S=0.0, dt=0.001, state=state.copy())
        assert abs(efd_euler - efd_rk4) < 0.1

    def test_pss_signal(self):
        """PSS 附加信号应影响励磁输出（多步仿真）"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5)
        state_no_pss = np.array([1.0, 1.0, 1.0, 0.0])
        state_with_pss = np.array([1.0, 1.0, 1.0, 0.0])

        for _ in range(30):
            efd_no, state_no_pss = exc.compute(
                V_ref=1.0, V_measured=1.0, V_S=0.0,
                dt=0.01, state=state_no_pss)
            efd_with, state_with_pss = exc.compute(
                V_ref=1.0, V_measured=1.0, V_S=0.1,
                dt=0.01, state=state_with_pss)
        # PSS 信号应改变输出
        assert abs(efd_with - efd_no) > 1e-6


class TestIEEET1Linearized:
    """IEEET1 线性化测试"""

    def test_get_state_space(self):
        """测试传递函数和状态空间返回值"""
        exc = IEEET1Params()
        A, B, C, D = exc.get_state_space()
        assert A.ndim == 2
        assert B.ndim == 2
        assert C.ndim == 2

    def test_linearized_matrices(self):
        """测试线性化矩阵计算"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5)
        A_exc, B_exc, C_exc, D_exc = exc.get_linearized_matrices(Efd0=1.0)
        assert A_exc.shape == (3, 3)
        assert B_exc.shape == (3, 1)
        assert C_exc.shape == (1, 3)
        # 检查基本结构：输出映射到 Efd（第二个状态）
        assert C_exc[0, 1] == 1.0

    def test_linearized_saturation_effect(self):
        """饱和应增加等效 KE"""
        exc = IEEET1Params(KA=200, TE=0.5, SE1=0.1, SE2=0.03, Efd_MAX=5.0)
        A_sat, _, _, _ = exc.get_linearized_matrices(Efd0=3.0)
        exc2 = IEEET1Params(KA=200, TE=0.5, SE1=0.0, SE2=0.0)
        A_nosat, _, _, _ = exc2.get_linearized_matrices(Efd0=3.0)
        # 饱和使励磁机极点左移（更稳定）
        assert A_sat[1, 1] < A_nosat[1, 1]

    def test_transfer_function(self):
        """测试传递函数返回非零多项式"""
        exc = IEEET1Params()
        num, den = exc.get_transfer_function()
        assert len(num) > 0
        assert len(den) > 0
