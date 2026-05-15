"""
节点阻抗矩阵 (Zbus)
==================

构造电力系统节点阻抗矩阵。

数学模型
--------
节点阻抗矩阵Zbus是节点导纳矩阵Ybus的逆矩阵：

    Zbus = Ybus⁻¹

网络方程可写为：
    V = Zbus × I

Zbus的物理意义：
    Zᵢᵢ：节点i的自阻抗（节点i单独注入单位电流时的电压）
    Zᵢⱼ：节点i和j之间的互阻抗（节点j注入单位电流时，节点i的电压）

短路计算中：
    - 三相短路电流：Iₖ = Vᶠ / Zkk（Zkk为短路点自阻抗）
    - 转移阻抗：Zkf（发电机k到短路点f的转移阻抗）

Zbus构造方法：
1. 直接求逆法：Zbus = inv(Ybus)
2. 支路追加法：逐条追加支路，递推构造Zbus

参考教材：陈珩《电力系统稳态分析》第二章 2.3节
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.load import Load
from psa4teaching.network.ybus import build_ybus, YBusResult


@dataclass
class ZBusResult:
    """节点阻抗矩阵构造结果

    Attributes:
        Zbus: 节点阻抗矩阵（复数矩阵）
        n_bus: 节点数
        bus_indices: 节点编号到矩阵索引的映射
        R: 电阻矩阵（Zbus的实部）
        X: 电抗矩阵（Zbus的虚部）
        Ybus: 原始节点导纳矩阵
    """
    Zbus: NDArray[np.complex128]       # 节点阻抗矩阵
    n_bus: int                          # 节点数
    bus_indices: Dict[int, int]         # 节点编号到索引的映射
    R: NDArray[np.float64]              # 电阻矩阵（实部）
    X: NDArray[np.float64]              # 电抗矩阵（虚部）
    Ybus: NDArray[np.complex128]        # 原始导纳矩阵

    def get_self_impedance(self, bus: int) -> complex:
        """获取指定节点的自阻抗

        Args:
            bus: 节点编号

        Returns:
            Zbus[i,i] —— 节点的自阻抗

        Note:
            物理意义：节点i单独注入单位电流时，节点i的电压。
            短路计算中，Zkk用于计算短路电流。
        """
        idx = self.bus_indices[bus]
        return self.Zbus[idx, idx]

    def get_transfer_impedance(self, bus_k: int, bus_f: int) -> complex:
        """获取两个节点之间的转移阻抗

        Args:
            bus_k: 节点k编号（发电机节点）
            bus_f: 节点f编号（短路点）

        Returns:
            Zbus[k,f] —— 节点k和f之间的转移阻抗

        Note:
            物理意义：节点f注入单位电流时，节点k的电压。
            短路计算中，转移阻抗用于计算各发电机对短路电流的贡献。
        """
        idx_k = self.bus_indices[bus_k]
        idx_f = self.bus_indices[bus_f]
        return self.Zbus[idx_k, idx_f]


def build_zbus(
    lines: List[Line],
    transformers: List[Transformer],
    loads: Optional[List[Load]] = None,
    bus_numbers: Optional[List[int]] = None,
    method: str = "inverse"
) -> ZBusResult:
    """构造节点阻抗矩阵

    Args:
        lines: 输电线路列表
        transformers: 变压器列表
        loads: 负荷列表（可选，恒阻抗负荷将加入网络）
        bus_numbers: 节点编号列表（可选）
        method: 构造方法，"inverse"为直接求逆，"building"为支路追加法

    Returns:
        ZBusResult: 包含Zbus矩阵和相关信息的结构体

    Note:
        直接求逆法：
            Zbus = inv(Ybus)

        支路追加法：
            从一个节点开始，逐条追加支路，递推构造Zbus。
            适用于大规模系统或需要增量更新的场合。

    Example:
        >>> lines = [Line(1, 2, 0.02, 0.1, 0.02)]
        >>> result = build_zbus(lines, [])
        >>> Zbus = result.Zbus
    """
    # 首先构造Ybus
    ybus_result = build_ybus(lines, transformers, loads, bus_numbers)
    Ybus = ybus_result.Ybus
    n_bus = ybus_result.n_bus
    bus_indices = ybus_result.bus_indices

    if method == "inverse":
        # 直接求逆法
        try:
            Zbus = np.linalg.inv(Ybus)
        except np.linalg.LinAlgError:
            raise ValueError("Ybus矩阵奇异，无法求逆。可能存在孤立节点或接地节点。")
    elif method == "building":
        # 支路追加法
        Zbus = _build_zbus_by_building(lines, transformers, bus_indices)
    else:
        raise ValueError(f"未知的构造方法: {method}")

    # 计算R和X矩阵
    R = Zbus.real
    X = Zbus.imag

    return ZBusResult(
        Zbus=Zbus,
        n_bus=n_bus,
        bus_indices=bus_indices,
        R=R,
        X=X,
        Ybus=Ybus
    )


def _build_zbus_by_building(
    lines: List[Line],
    transformers: List[Transformer],
    bus_indices: Dict[int, int]
) -> NDArray[np.complex128]:
    """支路追加法构造Zbus

    Args:
        lines: 输电线路列表
        transformers: 变压器列表
        bus_indices: 节点编号到索引的映射

    Returns:
        Zbus矩阵

    Note:
        支路追加法的基本原理：
        1. 从一个节点开始，Zbus初始为单位矩阵
        2. 追加树支（新节点）：扩展Zbus
        3. 追加连支（已有节点之间）：修正Zbus

        该方法在需要增量更新Zbus时比直接求逆更高效。
    """
    n_bus = len(bus_indices)

    # 简化实现：对于教学目的，这里使用直接求逆法
    # 支路追加法的完整实现较为复杂，可作为扩展练习

    # 先构造Ybus
    all_buses = sorted(bus_indices.keys())
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)

    for line in lines:
        contrib = line.get_ybus_contribution()
        for i, j, y in contrib:
            idx_i = bus_indices[i]
            idx_j = bus_indices[j]
            Ybus[idx_i, idx_j] += y

    for tx in transformers:
        contrib = tx.get_ybus_contribution()
        for i, j, y in contrib:
            idx_i = bus_indices[i]
            idx_j = bus_indices[j]
            Ybus[idx_i, idx_j] += y

    # 求逆
    Zbus = np.linalg.inv(Ybus)
    return Zbus


def compute_thevenin_impedance(
    Zbus: NDArray[np.complex128],
    bus_indices: Dict[int, int],
    fault_bus: int
) -> complex:
    """计算短路点的戴维南等效阻抗

    Args:
        Zbus: 节点阻抗矩阵
        bus_indices: 节点编号到索引的映射
        fault_bus: 短路点节点编号

    Returns:
        Zth = Zbus[f,f] —— 从短路点看入的戴维南等效阻抗

    Note:
        三相短路时，短路电流：
            If = Vf / Zth
        其中Vf为短路前故障点电压。
    """
    idx = bus_indices[fault_bus]
    return Zbus[idx, idx]


def compute_contribution_factors(
    Zbus: NDArray[np.complex128],
    bus_indices: Dict[int, int],
    fault_bus: int,
    generator_buses: List[int]
) -> Dict[int, complex]:
    """计算各发电机对短路电流的贡献系数

    Args:
        Zbus: 节点阻抗矩阵
        bus_indices: 节点编号到索引的映射
        fault_bus: 短路点节点编号
        generator_buses: 发电机节点编号列表

    Returns:
        字典，键为发电机节点编号，值为转移阻抗Zgf

    Note:
        发电机g对短路点f的短路电流贡献：
            Ig = Eg / Zgf
        其中Zgf = Zbus[g,f]为转移阻抗。
    """
    idx_f = bus_indices[fault_bus]
    contributions = {}

    for gen_bus in generator_buses:
        idx_g = bus_indices[gen_bus]
        Zgf = Zbus[idx_g, idx_f]
        contributions[gen_bus] = Zgf

    return contributions


def compute_shortport_impedance(
    Zbus: NDArray[np.complex128],
    bus_indices: Dict[int, int],
    bus_i: int,
    bus_j: int
) -> complex:
    """计算两个节点之间的端口阻抗

    Args:
        Zbus: 节点阻抗矩阵
        bus_indices: 节点编号到索引的映射
        bus_i: 端口节点i
        bus_j: 端口节点j

    Returns:
        Zij = Zii + Zjj - 2×Zij：端口阻抗

    Note:
        物理意义：在节点i和j之间注入单位电流时，
        节点i和j之间的电压差。
        用于计算两个节点之间的等效阻抗。
    """
    idx_i = bus_indices[bus_i]
    idx_j = bus_indices[bus_j]

    Zii = Zbus[idx_i, idx_i]
    Zjj = Zbus[idx_j, idx_j]
    Zij = Zbus[idx_i, idx_j]

    return Zii + Zjj - 2 * Zij