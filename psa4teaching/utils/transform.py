"""
坐标变换模块 (Coordinate Transformations)
======================================

提供电力系统分析中常用的坐标变换函数，包括 Park 变换 (abc→dq0)
和反 Park 变换 (dq0→abc)。

Park 变换将三相静止坐标系 (abc) 变换到两相旋转坐标系 (dq0)，
是同步发电机建模和暂态分析的基础工具。

变换公式
--------

Park 变换 (幅值不变形式):
    id = (2/3) * [ia*cos(θ-0°) + ib*cos(θ-120°) + ic*cos(θ+120°)]
    iq = -(2/3) * [ia*sin(θ-0°) + ib*sin(θ-120°) + ic*sin(θ+120°)]
    i0 = (1/3) * (ia + ib + ic)

反 Park 变换:
    ia = id*cos(θ) - iq*sin(θ) + i0
    ib = id*cos(θ-120°) - iq*sin(θ-120°) + i0
    ic = id*cos(θ+120°) - iq*sin(θ+120°) + i0

参考：
    - Kundur P. Power System Stability and Control, Ch.3
    - Park R.H. "Two-Reaction Theory of Synchronous Machines" (1929)
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def park_transform(
    ia: float, ib: float, ic: float, theta: float
) -> Tuple[float, float, float]:
    """Park 变换: abc → dq0（幅值不变形式）

    将三相静止坐标系下的电流/电压变换到转子 dq0 旋转坐标系。

    Args:
        ia: a 相瞬时值
        ib: b 相瞬时值
        ic: c 相瞬时值
        theta: 转子角度（弧度），θ = ωt + θ₀

    Returns:
        (id, iq, i0): d 轴、q 轴和零序分量

    Example:
        >>> import numpy as np
        >>> theta = np.pi / 4  # 45°
        >>> ia, ib, ic = 1.0, -0.5, -0.5
        >>> id_val, iq_val, i0 = park_transform(ia, ib, ic, theta)
    """
    c = np.cos(theta)
    s = np.sin(theta)
    c120 = np.cos(theta - 2 * np.pi / 3)
    s120 = np.sin(theta - 2 * np.pi / 3)
    c240 = np.cos(theta + 2 * np.pi / 3)
    s240 = np.sin(theta + 2 * np.pi / 3)

    id_val = (2.0 / 3.0) * (ia * c + ib * c120 + ic * c240)
    iq_val = -(2.0 / 3.0) * (ia * s + ib * s120 + ic * s240)
    i0 = (ia + ib + ic) / 3.0

    return id_val, iq_val, i0


def inv_park_transform(
    id_val: float, iq_val: float, i0: float, theta: float
) -> Tuple[float, float, float]:
    """反 Park 变换: dq0 → abc（幅值不变形式）

    将转子 dq0 坐标系下的量变换回三相静止坐标系。

    Args:
        id_val: d 轴分量
        iq_val: q 轴分量
        i0: 零序分量
        theta: 转子角度（弧度）

    Returns:
        (ia, ib, ic): 三相瞬时值

    Example:
        >>> ia, ib, ic = inv_park_transform(1.2, 0.3, 0.0, np.pi/6)
    """
    ia = id_val * np.cos(theta) - iq_val * np.sin(theta) + i0
    ib = id_val * np.cos(theta - 2 * np.pi / 3) - iq_val * np.sin(theta - 2 * np.pi / 3) + i0
    ic = id_val * np.cos(theta + 2 * np.pi / 3) - iq_val * np.sin(theta + 2 * np.pi / 3) + i0

    return ia, ib, ic


def park_transform_vectorized(
    ia: NDArray[np.float64], ib: NDArray[np.float64], ic: NDArray[np.float64],
    theta: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """向量化 Park 变换: abc → dq0

    同时变换整个时间序列。

    Args:
        ia, ib, ic: 三相瞬时值数组
        theta: 转子角度数组（弧度），长度与 ia 相同

    Returns:
        (id, iq, i0): dq0 分量数组

    Example:
        >>> t = np.linspace(0, 1, 1000)
        >>> theta = 100*np.pi*t  # 50Hz
        >>> ia = np.cos(theta)
        >>> ib = np.cos(theta - 2*np.pi/3)
        >>> ic = np.cos(theta + 2*np.pi/3)
        >>> id_val, iq_val, i0 = park_transform_vectorized(ia, ib, ic, theta)
    """
    c = np.cos(theta)
    s = np.sin(theta)
    c120 = np.cos(theta - 2 * np.pi / 3)
    s120 = np.sin(theta - 2 * np.pi / 3)
    c240 = np.cos(theta + 2 * np.pi / 3)
    s240 = np.sin(theta + 2 * np.pi / 3)

    id_val = (2.0 / 3.0) * (ia * c + ib * c120 + ic * c240)
    iq_val = -(2.0 / 3.0) * (ia * s + ib * s120 + ic * s240)
    i0 = (ia + ib + ic) / 3.0

    return id_val, iq_val, i0


def inv_park_transform_vectorized(
    id_val: NDArray[np.float64], iq_val: NDArray[np.float64],
    i0: NDArray[np.float64], theta: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """向量化反 Park 变换: dq0 → abc"""
    ia = id_val * np.cos(theta) - iq_val * np.sin(theta) + i0
    ib = id_val * np.cos(theta - 2 * np.pi / 3) - iq_val * np.sin(theta - 2 * np.pi / 3) + i0
    ic = id_val * np.cos(theta + 2 * np.pi / 3) - iq_val * np.sin(theta + 2 * np.pi / 3) + i0

    return ia, ib, ic


def build_park_matrix(theta: float) -> NDArray[np.float64]:
    """构建 Park 变换矩阵 P(θ)（幅值不变形式）

    Args:
        theta: 转子角度（弧度）

    Returns:
        P: 3×3 Park 变换矩阵，使得 [id, iq, i0]^T = P * [ia, ib, ic]^T

    Note:
        P(θ) = (2/3) * [[ cos(θ),  cos(θ-120°),  cos(θ+120°)],
                         [-sin(θ), -sin(θ-120°), -sin(θ+120°)],
                         [ 1/2,     1/2,           1/2        ]]

    Example:
        >>> P = build_park_matrix(np.pi/6)
        >>> dq0 = P @ np.array([1.0, -0.5, -0.5])
    """
    c = np.cos(theta)
    s = np.sin(theta)
    c120 = np.cos(theta - 2 * np.pi / 3)
    s120 = np.sin(theta - 2 * np.pi / 3)
    c240 = np.cos(theta + 2 * np.pi / 3)
    s240 = np.sin(theta + 2 * np.pi / 3)

    P = np.array([
        [c, c120, c240],
        [-s, -s120, -s240],
        [0.5, 0.5, 0.5]
    ])
    P *= 2.0 / 3.0

    return P


def build_inv_park_matrix(theta: float) -> NDArray[np.float64]:
    """构建反 Park 变换矩阵 P⁻¹(θ)

    Args:
        theta: 转子角度（弧度）

    Returns:
        P_inv: 3×3 反 Park 变换矩阵

    Note:
        P⁻¹(θ) = [[cos(θ), -sin(θ), 1],
                   [cos(θ-120°), -sin(θ-120°), 1],
                   [cos(θ+120°), -sin(θ+120°), 1]]
    """
    P_inv = np.array([
        [np.cos(theta), -np.sin(theta), 1.0],
        [np.cos(theta - 2 * np.pi / 3), -np.sin(theta - 2 * np.pi / 3), 1.0],
        [np.cos(theta + 2 * np.pi / 3), -np.sin(theta + 2 * np.pi / 3), 1.0],
    ])

    return P_inv


def clarke_transform(
    ia: float, ib: float, ic: float
) -> Tuple[float, float, float]:
    """Clarke 变换: abc → αβ0（幅值不变形式）

    将三相系统变换到两相静止坐标系（α轴对齐 a 相）。

    Args:
        ia, ib, ic: 三相瞬时值

    Returns:
        (i_alpha, i_beta, i0): αβ0 分量

    Note:
        与 Park 变换不同的是，Clarke 变换不旋转坐标系（θ=0），
        α 轴固定于 a 相绕组轴线。
    """
    i_alpha = (2.0 / 3.0) * (ia - 0.5 * ib - 0.5 * ic)
    i_beta = (2.0 / 3.0) * (np.sqrt(3) / 2 * (ib - ic))
    i0 = (ia + ib + ic) / 3.0

    return i_alpha, i_beta, i0

__all__ = [
    "park_transform",
    "inv_park_transform",
    "park_transform_vectorized",
    "inv_park_transform_vectorized",
    "build_park_matrix",
    "build_inv_park_matrix",
    "clarke_transform",
]
