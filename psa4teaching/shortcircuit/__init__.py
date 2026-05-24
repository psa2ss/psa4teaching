"""
短路计算模块
============

本模块提供电力系统短路电流计算功能，包括：
1. 转移阻抗计算
2. 对称短路电流计算（基于Zbus）
3. 不对称短路计算（单相接地、两相短路、两相接地）
4. 序网模型构造
5. GB15544标准支持
6. 同步发电机机端三相短路计算（三种方法）

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
    phase_to_sequence,
    sequence_to_phase,
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
from psa4teaching.shortcircuit.terminal_shortcircuit import (
    GeneratorSCParams,
    TerminalSCResult,
    calculate_terminal_shortcircuit_mathematical,
    calculate_terminal_shortcircuit_experimental,
    calculate_terminal_shortcircuit_simplified,
    plot_comparison,
    plot_components_comparison,
)

__all__ = [
    # 对称短路
    "SymmetricFaultResult", "calculate_three_phase_fault", "calculate_transfer_impedances",
    # 不对称短路
    "AsymmetricFaultResult", "calculate_single_line_to_ground",
    "calculate_line_to_line", "calculate_double_line_to_ground",
    "phase_to_sequence", "sequence_to_phase",
    # 序网
    "SequenceNetworks", "build_positive_sequence_network",
    "build_negative_sequence_network", "build_zero_sequence_network",
    # GB15544
    "GB15544Result", "calculate_gb15544", "get_correction_factor",
    # 机端三相短路
    "GeneratorSCParams", "TerminalSCResult",
    "calculate_terminal_shortcircuit_mathematical",
    "calculate_terminal_shortcircuit_experimental",
    "calculate_terminal_shortcircuit_simplified",
    "plot_comparison", "plot_components_comparison",
]
