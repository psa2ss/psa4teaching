"""
节点导纳矩阵 (Ybus)
==================

构造电力系统节点导纳矩阵。

数学模型
--------
节点导纳矩阵Ybus是电力系统网络方程的核心：

    I = Ybus × V

其中：
    I = [I₁, I₂, ..., Iₙ]ᵀ —— 节点注入电流向量
    V = [V₁, V₂, ..., Vₙ]ᵀ —— 节点电压向量
    Ybus —— 节点导纳矩阵

Ybus的构造规则：
1. 对角元 Yii = Σyᵢⱼ（与节点i相连的所有支路导纳之和）
2. 非对角元 Yij = -yᵢⱼ（节点i与j之间支路导纳的负值）

参考教材：陈珩《电力系统稳态分析》第二章 2.1-2.2节
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.load import Load


@dataclass
class YBusResult:
    """节点导纳矩阵构造结果

    Attributes:
        Ybus: 节点导纳矩阵（复数矩阵）
        n_bus: 节点数
        bus_indices: 节点编号到矩阵索引的映射
        G: 电导矩阵（Ybus的实部）
        B: 电纳矩阵（Ybus的虚部）
    """
    Ybus: NDArray[np.complex128]       # 节点导纳矩阵
    n_bus: int                          # 节点数
    bus_indices: Dict[int, int]         # 节点编号到索引的映射
    G: NDArray[np.float64]              # 电导矩阵（实部）
    B: NDArray[np.float64]              # 电纳矩阵（虚部）

    def get_yii(self, bus: int) -> complex:
        """获取指定节点的自导纳

        Args:
            bus: 节点编号

        Returns:
            Ybus[i,i] —— 节点的自导纳
        """
        idx = self.bus_indices[bus]
        return self.Ybus[idx, idx]

    def get_yij(self, bus_i: int, bus_j: int) -> complex:
        """获取两个节点之间的互导纳

        Args:
            bus_i: 节点i编号
            bus_j: 节点j编号

        Returns:
            Ybus[i,j] —— 节点i和j之间的互导纳
        """
        idx_i = self.bus_indices[bus_i]
        idx_j = self.bus_indices[bus_j]
        return self.Ybus[idx_i, idx_j]


def build_ybus(
    lines: List[Line],
    transformers: List[Transformer],
    loads: Optional[List[Load]] = None,
    bus_numbers: Optional[List[int]] = None
) -> YBusResult:
    """构造节点导纳矩阵

    Args:
        lines: 输电线路列表
        transformers: 变压器列表
        loads: 负荷列表（可选，恒阻抗负荷将加入Ybus）
        bus_numbers: 节点编号列表（可选，若提供则按此顺序排列）

    Returns:
        YBusResult: 包含Ybus矩阵和相关信息的结构体

    Note:
        构造步骤：
        1. 确定所有节点编号
        2. 建立节点编号到矩阵索引的映射
        3. 初始化零矩阵
        4. 遍历所有支路，累加导纳贡献
        5. （可选）累加恒阻抗负荷的导纳

    Example:
        >>> lines = [Line(1, 2, 0.02, 0.1, 0.02)]
        >>> transformers = [Transformer(2, 3, 0.002, 0.1, k=1.05)]
        >>> result = build_ybus(lines, transformers)
        >>> Ybus = result.Ybus
        >>> n_bus = result.n_bus
    """
    # Step 1: 收集所有节点编号
    bus_set = set()
    for line in lines:
        bus_set.add(line.from_bus)
        bus_set.add(line.to_bus)
    for tx in transformers:
        bus_set.add(tx.from_bus)
        bus_set.add(tx.to_bus)

    if loads:
        for load in loads:
            if load.model_type.name == "CONSTANT_IMPEDANCE":
                bus_set.add(load.bus)

    # 确定节点顺序
    if bus_numbers:
        # 使用用户指定的节点顺序
        all_buses = bus_numbers
        for bus in bus_set:
            if bus not in all_buses:
                all_buses.append(bus)
    else:
        all_buses = sorted(list(bus_set))

    n_bus = len(all_buses)
    bus_indices = {bus: idx for idx, bus in enumerate(all_buses)}

    # Step 2: 初始化导纳矩阵
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)

    # Step 3: 添加线路贡献
    for line in lines:
        contrib = line.get_ybus_contribution()
        for i, j, y in contrib:
            idx_i = bus_indices[i]
            idx_j = bus_indices[j]
            Ybus[idx_i, idx_j] += y

    # Step 4: 添加变压器贡献
    for tx in transformers:
        contrib = tx.get_ybus_contribution()
        for i, j, y in contrib:
            idx_i = bus_indices[i]
            idx_j = bus_indices[j]
            Ybus[idx_i, idx_j] += y

    # Step 5: 添加恒阻抗负荷
    if loads:
        for load in loads:
            if load.model_type.name == "CONSTANT_IMPEDANCE":
                idx = bus_indices[load.bus]
                Y_load = load.get_admittance()
                Ybus[idx, idx] += Y_load

    # 计算G和B矩阵
    G = Ybus.real
    B = Ybus.imag

    return YBusResult(
        Ybus=Ybus,
        n_bus=n_bus,
        bus_indices=bus_indices,
        G=G,
        B=B
    )


def compute_injection_currents(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """计算节点注入电流

    Args:
        Ybus: 节点导纳矩阵
        V: 节点电压向量

    Returns:
        I = Ybus × V：节点注入电流向量

    Note:
        发电机节点注入电流为正，负荷节点为负。
    """
    return Ybus @ V


def compute_injection_power(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """计算节点注入功率

    Args:
        Ybus: 节点导纳矩阵
        V: 节点电压向量

    Returns:
        S = V ⊙ conj(I) = V ⊙ conj(Ybus × V)：节点注入复功率向量

    Note:
        Sᵢ = Vᵢ × conj(Iᵢ) = Vᵢ × conj(Σⱼ YᵢⱼVⱼ)
    """
    I = Ybus @ V
    S = V * np.conj(I)
    return S


def compute_branch_flows(
    Ybus: NDArray[np.complex128],
    V: NDArray[np.complex128],
    bus_indices: Dict[int, int],
    branches: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], complex]:
    """计算支路功率分布

    Args:
        Ybus: 节点导纳矩阵
        V: 节点电压向量
        bus_indices: 节点编号到索引的映射
        branches: 支路列表，每个元素为(from_bus, to_bus)

    Returns:
        字典，键为(from_bus, to_bus)，值为从from端流向to端的复功率

    Note:
        Sᵢⱼ = Vᵢ × conj((Vᵢ - Vⱼ) × Yᵢⱼ)
    """
    flows = {}
    for i, j in branches:
        idx_i = bus_indices[i]
        idx_j = bus_indices[j]
        Vi = V[idx_i]
        Vj = V[idx_j]
        Yij = Ybus[idx_i, idx_j]

        # 支路电流（从i流向j）
        Iij = (Vi - Vj) * (-Yij)  # 注意：Yij是负的支路导纳

        # 支路功率
        Sij = Vi * np.conj(Iij)
        flows[(i, j)] = Sij

    return flows