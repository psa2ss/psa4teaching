"""
短路计算模块
============

本模块提供电力系统短路电流计算功能，包括：
1. 转移阻抗计算
2. 对称短路电流计算（基于Zbus）
3. 不对称短路计算（单相接地、两相短路、两相接地）
4. 序网模型构造
5. GB15544标准支持

参考教材：
    - 陈珩《电力系统稳态分析》第六章
    - 李光琦《电力系统暂态分析》第二章
"""

from psa4teaching.shortcircuit.symmetric import (
    SymmetricFaultResult,
    calculate_three_phase_fault,
    calculate_transfer_impedances,
)
from psa4teaching.shortcircuit.asymmetric import (
    AsymmetricFaultResult,
    calculate_single_line_to_ground,
    calculate_line_to_line,
    calculate_double_line_to_ground,
)
from psa4teaching.shortcircuit.sequence_network import (
    SequenceNetworks,
    build_positive_sequence_network,
    build_negative_sequence_network,
    build_zero_sequence_network,
)
from psa4teaching.shortcircuit.gb15544 import (
    GB15544Result,
    calculate_gb15544,
    get_correction_factor,
)

__all__ = [
    # 对称短路
    "SymmetricFaultResult", "calculate_three_phase_fault", "calculate_transfer_impedances",
    # 不对称短路
    "AsymmetricFaultResult", "calculate_single_line_to_ground",
    "calculate_line_to_line", "calculate_double_line_to_ground",
    # 序网
    "SequenceNetworks", "build_positive_sequence_network",
    "build_negative_sequence_network", "build_zero_sequence_network",
    # GB15544
    "GB15544Result", "calculate_gb15544", "get_correction_factor",
]