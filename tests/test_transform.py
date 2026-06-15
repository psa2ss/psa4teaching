"""
坐标变换模块测试
=================

测试 Park/Clarke 变换的正确性，包括：
- 正变换和反变换的一致性（round-trip 测试）
- 与已知解析解的对比
- 向量化变换的正确性
"""

import numpy as np
import pytest
from psa4teaching.utils.transform import (
    park_transform,
    inv_park_transform,
    park_transform_vectorized,
    inv_park_transform_vectorized,
    build_park_matrix,
    build_inv_park_matrix,
    clarke_transform,
)


class TestParkTransform:
    """Park 变换测试"""

    def test_round_trip_single_point(self):
        """正变换+反变换应恢复原始值"""
        theta = np.pi / 6  # 30°
        ia, ib, ic = 1.2, -0.3, -0.9

        id_val, iq_val, i0 = park_transform(ia, ib, ic, theta)
        ia_r, ib_r, ic_r = inv_park_transform(id_val, iq_val, i0, theta)

        assert abs(ia_r - ia) < 1e-10
        assert abs(ib_r - ib) < 1e-10
        assert abs(ic_r - ic) < 1e-10

    def test_balanced_sinusoidal(self):
        """平衡三相正弦信号 → dq 应为直流量"""
        theta = np.pi / 3  # 60°
        # 平衡三相: ia = cos(θ), ib = cos(θ-120°), ic = cos(θ+120°)
        ia = np.cos(theta)
        ib = np.cos(theta - 2 * np.pi / 3)
        ic = np.cos(theta + 2 * np.pi / 3)

        id_val, iq_val, i0 = park_transform(ia, ib, ic, theta)

        # 平衡三相: id = 1.0, iq = 0.0, i0 = 0.0 (幅值不变形式)
        assert abs(id_val - 1.0) < 1e-10
        assert abs(iq_val - 0.0) < 1e-10
        assert abs(i0 - 0.0) < 1e-10

    def test_zero_sequence(self):
        """同相位信号 → 零序分量"""
        theta = np.pi / 4
        # 三相相等 → 只有零序分量
        ia = ib = ic = 1.0

        id_val, iq_val, i0 = park_transform(ia, ib, ic, theta)

        assert abs(i0 - 1.0) < 1e-10
        assert abs(id_val) < 1e-10
        assert abs(iq_val) < 1e-10

    def test_dc_to_ac(self):
        """dq 直流 → abc 应为正弦"""
        thetas = np.linspace(0, 2 * np.pi, 361)  # 0°, 1°, ..., 360°
        id_dc = 1.0
        iq_dc = 0.0
        i0 = 0.0

        ia_vals = []
        for theta in thetas:
            ia, ib, ic = inv_park_transform(id_dc, iq_dc, i0, theta)
            ia_vals.append(ia)

        # ia 应为 cos 波: max=1, min=-1
        assert abs(np.max(ia_vals) - 1.0) < 1e-10
        assert abs(np.min(ia_vals) + 1.0) < 1e-10


class TestVectorizedTransform:
    """向量化变换测试"""

    def test_vectorized_round_trip(self):
        """向量化变换的 round-trip 测试"""
        t = np.linspace(0, 0.02, 100)  # 1 个周期 (50 Hz)
        theta = 2 * np.pi * 50 * t

        ia = np.cos(theta)
        ib = np.cos(theta - 2 * np.pi / 3)
        ic = np.cos(theta + 2 * np.pi / 3)

        id_val, iq_val, i0 = park_transform_vectorized(ia, ib, ic, theta)
        ia_r, ib_r, ic_r = inv_park_transform_vectorized(id_val, iq_val, i0, theta)

        assert np.allclose(ia, ia_r, atol=1e-10)
        assert np.allclose(ib, ib_r, atol=1e-10)
        assert np.allclose(ic, ic_r, atol=1e-10)

    def test_vectorized_vs_scalar(self):
        """向量化应与逐点标量计算一致"""
        t = np.linspace(0, 0.02, 50)
        theta = 2 * np.pi * 50 * t
        ia = np.cos(theta)
        ib = np.cos(theta - 2 * np.pi / 3)
        ic = np.cos(theta + 2 * np.pi / 3)

        id_vec, iq_vec, i0_vec = park_transform_vectorized(ia, ib, ic, theta)

        for k in range(len(t)):
            id_s, iq_s, i0_s = park_transform(ia[k], ib[k], ic[k], theta[k])
            assert abs(id_vec[k] - id_s) < 1e-10
            assert abs(iq_vec[k] - iq_s) < 1e-10


class TestParkMatrix:
    """Park 变换矩阵测试"""

    def test_matrix_vs_function(self):
        """矩阵变换应与函数变换一致"""
        theta = np.pi / 6
        P = build_park_matrix(theta)
        P_inv = build_inv_park_matrix(theta)

        abc = np.array([1.0, -0.5, -0.5])
        dq0_matrix = P @ abc
        dq0_func = park_transform(abc[0], abc[1], abc[2], theta)
        dq0_func_arr = np.array(dq0_func)

        assert np.allclose(dq0_matrix, dq0_func_arr, atol=1e-10)

        # 反变换验证
        abc_matrix = P_inv @ dq0_matrix
        assert np.allclose(abc_matrix, abc, atol=1e-10)

    def test_matrix_orthogonality(self):
        """P * P_inv ≈ I"""
        theta = np.pi / 4
        P = build_park_matrix(theta)
        P_inv = build_inv_park_matrix(theta)

        product = P @ P_inv
        assert np.allclose(product, np.eye(3), atol=1e-10)


class TestClarkeTransform:
    """Clarke 变换测试"""

    def test_balanced_sinusoidal(self):
        """平衡三相 → Clarke αβ0"""
        ia = 1.0
        ib = -0.5
        ic = -0.5

        alpha, beta, zero = clarke_transform(ia, ib, ic)
        assert abs(alpha - 1.0) < 1e-10
        assert abs(beta - 0.0) < 1e-10
        assert abs(zero) < 1e-10

    def test_symmetrical_component(self):
        """对称三相 → Clarke 零序为零"""
        theta = np.pi / 3
        ia = np.cos(theta)
        ib = np.cos(theta - 2 * np.pi / 3)
        ic = np.cos(theta + 2 * np.pi / 3)

        alpha, beta, zero = clarke_transform(ia, ib, ic)
        assert abs(zero) < 1e-10
