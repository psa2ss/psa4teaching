"""
通用工具函数
============
"""

import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray


def per_unit(value: float, base_value: float) -> float:
    """转换为标幺值

    Args:
        value: 实际值
        base_value: 基准值

    Returns:
        标幺值
    """
    return value / base_value


def from_per_unit(pu_value: float, base_value: float) -> float:
    """从标幺值转换为实际值

    Args:
        pu_value: 标幺值
        base_value: 基准值

    Returns:
        实际值
    """
    return pu_value * base_value


def compute_S_base(V_base_kV: float, I_base_kA: float) -> float:
    """计算基准容量

    Args:
        V_base_kV: 基准电压（kV）
        I_base_kA: 基准电流（kA）

    Returns:
        S_base：基准容量（MVA）
    """
    return np.sqrt(3) * V_base_kV * I_base_kA


def compute_Z_base(V_base_kV: float, S_base_MVA: float) -> float:
    """计算基准阻抗

    Args:
        V_base_kV: 基准电压（kV）
        S_base_MVA: 基准容量（MVA）

    Returns:
        Z_base：基准阻抗（Ω）
    """
    return V_base_kV**2 / S_base_MVA


def compute_I_base(V_base_kV: float, S_base_MVA: float) -> float:
    """计算基准电流

    Args:
        V_base_kV: 基准电压（kV）
        S_base_MVA: 基准容量（MVA）

    Returns:
        I_base：基准电流（kA）
    """
    return S_base_MVA / (np.sqrt(3) * V_base_kV)


def deg_to_rad(degrees: float) -> float:
    """角度转弧度"""
    return np.radians(degrees)


def rad_to_deg(radians: float) -> float:
    """弧度转角度"""
    return np.degrees(radians)


def polar_to_rect(magnitude: float, angle_deg: float) -> complex:
    """极坐标转直角坐标

    Args:
        magnitude: 幅值
        angle_deg: 角度（度）

    Returns:
        复数
    """
    angle_rad = np.radians(angle_deg)
    return magnitude * (np.cos(angle_rad) + 1j * np.sin(angle_rad))


def rect_to_polar(z: complex) -> Tuple[float, float]:
    """直角坐标转极坐标

    Args:
        z: 复数

    Returns:
        (magnitude, angle_deg): 幅值和角度（度）
    """
    return abs(z), np.degrees(np.angle(z))


def print_matrix(matrix, name="M", precision=4):
    """格式化打印矩阵"""
    rows, cols = matrix.shape
    print(f"\n{name} ({rows}×{cols}):")
    for i in range(rows):
        row_str = "  ".join(f"{matrix[i,j]:.{precision}f}" for j in range(cols))
        print(f"  [{row_str}]")
