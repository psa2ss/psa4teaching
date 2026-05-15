"""
psa4teaching 测试套件 - 网络矩阵测试

使用教材经典例题验证Ybus和Zbus的正确性。
参考：陈珩《电力系统稳态分析》第二章例题
"""

import numpy as np
import pytest
from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.network.ybus import build_ybus, compute_injection_power
from psa4teaching.network.zbus import build_zbus, compute_thevenin_impedance


class TestYbus:
    """节点导纳矩阵测试"""

    def test_two_bus_system(self):
        """两节点系统Ybus测试

        系统：1 ---(R=0, X=0.1, B=0)--- 2
        Ybus应为：
            [10j  -10j]
            [-10j  10j]
        """
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.0)]
        result = build_ybus(lines, [])

        assert result.n_bus == 2
        # Y12 = Y21 = -1/j0.1 = -(-10j) = 10j → 但实际 1/(j0.1) = -10j
        # Y12 = -Y_series = -(-10j) = 10j? No.
        # Y_series = 1/(j0.1) = -10j
        # Y12 = -Y_series = 10j... wait
        # Actually: Ybus[i,j] = -y_ij for i≠j where y_ij = 1/Z
        # y_12 = 1/(j0.1) = -10j
        # Ybus[0,1] = -(-10j) = 10j... but the code uses get_ybus_contribution
        # which gives (i,j,-Y_series)
        # Y_series = 1/(0+j0.1) = -10j
        # contribution: (1,2,-Y_series) = (1,2,-(-10j)) = (1,2,10j)
        # Hmm, that's wrong. For a line, the off-diagonal should be -Y_series = -(-10j) = 10j
        # But conventionally Ybus[i,j] = -y_series for i≠j
        # So Ybus[0,1] = -y_series = -(-10j) = 10j? No!
        # y_series = 1/Z = 1/(j0.1) = -10j
        # Ybus[0,1] = -y_series = -(-10j) = 10j
        # That doesn't seem right. Let me reconsider.

        # Actually in the Line model:
        # get_ybus_contribution returns (i,j,-Y_series) for off-diagonal
        # Y_series = 1/Z = 1/(j0.1) = -10j
        # So off-diagonal contribution = -Y_series = -(-10j) = 10j
        # But the standard formula is Ybus[i,j] = -y_ij
        # where y_ij is the series admittance of the branch
        # So Ybus[0,1] = -(-10j) = 10j???

        # Wait, let me recalculate properly:
        # Z = j0.1
        # Y_series = 1/Z = 1/(j0.1) = -j10
        # Standard Ybus:
        #   Ybus[i,i] = Y_shunt + Y_series = 0 + (-j10) = -j10
        #   Ybus[i,j] = -Y_series = -(-j10) = j10
        # Hmm, but that gives Ybus = [[-10j, 10j],[10j, -10j]]
        # which has negative diagonal and positive off-diagonal.

        # Actually the convention is:
        # For a simple series impedance Z between i and j:
        #   Yii = y_ij = 1/Z
        #   Yij = -y_ij = -1/Z
        # So with Z = j0.1:
        #   y = 1/(j0.1) = -10j
        #   Yii = -10j, Yij = 10j
        # This gives Ybus = [[-10j, 10j],[10j, -10j]]

        # Let's check the actual values
        Ybus = result.Ybus
        # Diagonal elements
        assert abs(Ybus[0, 0].imag + 10.0) < 1e-8  # -10j
        assert abs(Ybus[1, 1].imag + 10.0) < 1e-8  # -10j
        # Off-diagonal elements
        assert abs(Ybus[0, 1].imag - 10.0) < 1e-8  # 10j
        assert abs(Ybus[1, 0].imag - 10.0) < 1e-8  # 10j

    def test_three_bus_system(self):
        """三节点系统Ybus测试

        节点1-2: R=0.02, X=0.1, B=0.02
        节点2-3: R=0.03, X=0.15, B=0.03
        节点1-3: R=0.01, X=0.08, B=0.01
        """
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
            Line(from_bus=1, to_bus=3, R=0.01, X=0.08, B=0.01),
        ]
        result = build_ybus(lines, [])

        assert result.n_bus == 3

        # Ybus应为对称矩阵（Yij = Yji）
        Ybus = result.Ybus
        assert np.allclose(Ybus, Ybus.T)

        # 对角元应为负虚部主导（感性网络）
        for i in range(3):
            assert Ybus[i, i].imag < 0  # 感性网络对角元虚部为负

    def test_transformer_ybus(self):
        """变压器Ybus测试（标准变比）"""
        tx = Transformer(from_bus=1, to_bus=2, RT=0.0, XT=0.1, k=1.0)
        lines = []
        result = build_ybus(lines, [tx])

        Ybus = result.Ybus
        # 标准变比变压器等效为串联阻抗
        assert abs(Ybus[0, 0].imag + 10.0) < 1e-6
        assert abs(Ybus[1, 1].imag + 10.0) < 1e-6

    def test_line_with_shunt(self):
        """带对地电容的线路Ybus测试"""
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.04)]
        result = build_ybus(lines, [])

        Ybus = result.Ybus
        # 对角元 = Y_series + Y_shunt = -10j + j0.02
        assert abs(Ybus[0, 0].imag + 9.98) < 1e-8
        assert abs(Ybus[1, 1].imag + 9.98) < 1e-8

    def test_ybus_symmetry(self):
        """Ybus对称性测试"""
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
        ]
        txs = [Transformer(from_bus=1, to_bus=3, RT=0.002, XT=0.08, k=1.05)]

        result = build_ybus(lines, txs)
        Ybus = result.Ybus
        # 纯电感网络Ybus是对称矩阵(Yij=Yji)，不是Hermitian矩阵
        assert np.allclose(Ybus, Ybus.T)


class TestZbus:
    """节点阻抗矩阵测试"""

    def test_zbus_inverse_of_ybus(self):
        """Zbus = Ybus⁻¹ 验证"""
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
        ]
        ybus_result = build_ybus(lines, [])
        zbus_result = build_zbus(lines, [])

        # Zbus × Ybus ≈ I
        product = zbus_result.Zbus @ ybus_result.Ybus
        identity = np.eye(ybus_result.n_bus)
        assert np.allclose(product, identity, atol=1e-8)

    def test_self_impedance(self):
        """自阻抗测试

        注意：纯电感两节点系统Ybus奇异（行和为零），
        需要增加对地支路（如对地电容）才能求逆
        """
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.02)]
        result = build_zbus(lines, [])

        Z11 = result.get_self_impedance(1)
        Z22 = result.get_self_impedance(2)
        # 对称系统，|Z11| ≈ |Z22|
        assert abs(abs(Z11) - abs(Z22)) < 1e-6

    def test_transfer_impedance(self):
        """转移阻抗测试

        增加对地电容使Ybus非奇异
        """
        lines = [Line(from_bus=1, to_bus=2, R=0.0, X=0.1, B=0.02)]
        result = build_zbus(lines, [])

        Z12 = result.get_transfer_impedance(1, 2)
        Z21 = result.get_transfer_impedance(2, 1)
        # Z12 = Z21（对称性）
        assert abs(Z12 - Z21) < 1e-10

    def test_thevenin_impedance(self):
        """戴维南等效阻抗测试"""
        lines = [
            Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02),
            Line(from_bus=2, to_bus=3, R=0.03, X=0.15, B=0.03),
        ]
        result = build_zbus(lines, [])

        Zth = compute_thevenin_impedance(result.Zbus, result.bus_indices, 3)
        # 应等于Zbus[2,2]
        idx = result.bus_indices[3]
        assert abs(Zth - result.Zbus[idx, idx]) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])