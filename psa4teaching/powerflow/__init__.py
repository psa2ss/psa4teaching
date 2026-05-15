"""
潮流计算模块
============

本模块提供电力系统潮流计算的多种算法，包括：
1. 牛顿-拉夫逊法（Newton-Raphson Method）
2. P-Q分解法（Fast Decoupled Power Flow）
3. 直流潮流（DC Power Flow）

参考教材：陈珩《电力系统稳态分析》第三至四章
"""

from psa4teaching.powerflow.newton_raphson import (
    NewtonRaphsonResult,
    run_newton_raphson,
    build_jacobian,
)
from psa4teaching.powerflow.fast_decoupled import (
    FastDecoupledResult,
    run_fast_decoupled,
    build_b_matrices,
)
from psa4teaching.powerflow.dc_powerflow import (
    DCPowerFlowResult,
    run_dc_powerflow,
)

__all__ = [
    # 牛顿-拉夫逊法
    "NewtonRaphsonResult", "run_newton_raphson", "build_jacobian",
    # P-Q分解法
    "FastDecoupledResult", "run_fast_decoupled", "build_b_matrices",
    # 直流潮流
    "DCPowerFlowResult", "run_dc_powerflow",
]