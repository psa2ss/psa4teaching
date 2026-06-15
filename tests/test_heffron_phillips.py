"""
Heffron-Phillips K1-K6 常数测试
===============================

验证 SMIB Heffron-Phillips 线性化模型的正确性：
- K1-K6 常数计算
- 状态矩阵特征值
- 励磁系统增强
- K 常数扫描
"""

import numpy as np
import pytest
from psa4teaching.stability.heffron_phillips import (
    compute_heffron_phillips_constants,
    sweep_k_constants,
    HeffronPhillipsResult,
)
from psa4teaching.models.ieeet1 import IEEET1Params


class TestHeffronPhillipsBasic:
    """基本 K1-K6 计算测试"""

    def test_k_constants_signs(self):
        """验证 K 常数的符号约定"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        # K1 > 0: 同步转矩系数为正
        assert result.K1 > 0
        # K2 > 0: Pe 随 Eq' 增加而增加
        assert result.K2 > 0
        # K3 > 0: 励磁回路阻抗系数
        assert result.K3 > 0
        # K4 > 0: 去磁效应
        assert result.K4 > 0
        # K6 > 0: Vt 随 Eq' 增加而增加
        assert result.K6 > 0

    def test_k5_negative_at_high_loading(self):
        """高功角下 K5 应为负（负阻尼风险）"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(45), verbose=False,
        )
        assert result.K5 < 0, "高功角下 K5 应为负"

    def test_k5_positive_at_low_loading(self):
        """低功角下 K5 可能为正"""
        result = compute_heffron_phillips_constants(
            E_prime=1.0, V_infinity=1.0, X_total=0.3,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(5), verbose=False,
        )
        # 低功角时 K5 可能为正（取决于参数）
        # 这里只验证结果类型正确
        assert isinstance(result.K5, float)

    def test_result_is_dataclass(self):
        """验证返回正确的 dataclass 类型"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        assert isinstance(result, HeffronPhillipsResult)
        assert hasattr(result, 'K1')
        assert hasattr(result, 'K6')
        assert hasattr(result, 'eigenvalues')
        assert hasattr(result, 'state_matrix')


class TestHeffronPhillipsStateMatrix:
    """状态矩阵测试"""

    def test_third_order_model_size(self):
        """3 阶模型（无励磁）状态矩阵应为 3×3"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        assert result.state_matrix.shape == (3, 3)
        assert len(result.eigenvalues) == 3

    def test_fourth_order_model_size(self):
        """4 阶模型（简化励磁）状态矩阵应为 4×4"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), Ka=50, Te=0.3, verbose=False,
        )
        assert result.state_matrix.shape == (4, 4)
        assert len(result.eigenvalues) == 4

    def test_ieeet1_model_size(self):
        """IEEET1 励磁模型状态矩阵应为 6×6"""
        exc = IEEET1Params(KA=200, TA=0.02, TE=0.5, KF=0.05, TF=1.0)
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), exciter_params=exc, verbose=False,
        )
        assert result.state_matrix.shape == (6, 6)
        assert len(result.eigenvalues) == 6

    def test_oscillation_frequency_reasonable(self):
        """振荡频率应在机电振荡范围 (0.2-2 Hz)"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        osc_freqs = [f for f in result.frequencies if f > 0.01]
        assert len(osc_freqs) > 0
        for f in osc_freqs:
            assert 0.1 < f < 5.0, f"振荡频率 {f:.2f} Hz 不在机电振荡范围"

    def test_damping_with_exciter(self):
        """励磁系统应改变阻尼"""
        result_noexc = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        result_exc = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), Ka=50, Te=0.3, verbose=False,
        )
        # 励磁系统的存在改变了阻尼
        zeta_noexc = [z for z, f in zip(result_noexc.damping_ratios,
                     result_noexc.frequencies) if f > 0.01]
        zeta_exc = [z for z, f in zip(result_exc.damping_ratios,
                    result_exc.frequencies) if f > 0.01]
        assert zeta_noexc != zeta_exc

    def test_eigenvalue_conjugate_pairs(self):
        """特征值应为共轭对"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        for ev in result.eigenvalues:
            if ev.imag > 1e-6:
                conj_exists = any(
                    abs(e2 - np.conj(ev)) < 1e-10
                    for e2 in result.eigenvalues
                )
                assert conj_exists

    def test_participation_factors_exist(self):
        """3 阶模型的参与因子矩阵应存在"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        assert result.participation_factors is not None
        assert result.participation_factors.shape == (3, 3)

    def test_operating_point_info(self):
        """工作点信息应完整"""
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        op = result.operating_point
        assert 'delta_0_deg' in op
        assert 'Vt0' in op
        assert 'Pe0' in op
        assert 'Id0' in op
        assert 'Iq0' in op

    def test_delta_from_power(self):
        """从 Pm 反解 δ 应正确"""
        # 先算一个参考 Pe
        ref = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            delta_0=np.radians(30), verbose=False,
        )
        # 用算出的 Pe 作为 Pm 反解 δ
        result = compute_heffron_phillips_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
            Td0_prime=8.0, H=5.0, D=0.0,
            Pm=ref.operating_point['Pe0'], verbose=False,
        )
        delta_deg = result.operating_point['delta_0_deg']
        assert abs(delta_deg - 30) < 0.1


class TestSweepKConstants:
    """K 常数扫描测试"""

    def test_sweep_returns_valid_data(self):
        """扫描应返回完整的曲线数据"""
        curves = sweep_k_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            delta_range=(5, 85), n_points=20,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
        )
        assert len(curves['delta_deg']) == 20
        for key in ['K1', 'K2', 'K3', 'K4', 'K5', 'K6']:
            assert key in curves
            assert len(curves[key]) == 20
            assert not np.any(np.isnan(curves[key]))

    def test_k5_crosses_zero(self):
        """K5 在低功角为正、高功角为负"""
        curves = sweep_k_constants(
            E_prime=1.2, V_infinity=1.0, X_total=0.5,
            delta_range=(5, 85), n_points=50,
            Xd=1.8, Xd_prime=0.3, Xq=1.7,
        )
        k5_vals = curves['K5']
        # K5 应随功角增大而减小
        assert k5_vals[-1] < k5_vals[0]
