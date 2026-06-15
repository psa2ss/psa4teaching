"""
电力系统数据集模块
==================

提供标准测试系统的参数定义，用于教学示例和算法验证。

支持的系统：
    - Kundur 两区域四机系统 (Kundur Two-Area System)
"""

from psa4teaching.data.kundur_two_area import (
    create_kundur_two_area_system,
    get_area_generators,
    get_load_buses,
    get_tie_line_buses,
    get_generator_parameters_table,
    S_BASE,
    V_BASE,
    F_BASE,
    GEN_MVA,
    GEN_KV,
)

__all__ = [
    "create_kundur_two_area_system",
    "get_area_generators",
    "get_load_buses",
    "get_tie_line_buses",
    "get_generator_parameters_table",
    "S_BASE",
    "V_BASE",
    "F_BASE",
    "GEN_MVA",
    "GEN_KV",
]
