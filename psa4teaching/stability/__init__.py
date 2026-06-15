"""
稳定计算模块
============

本模块提供电力系统稳定性计算功能，包括：
1. 暂态稳定时域仿真
2. 小干扰稳定分析（特征值法）
3. 等面积准则 + 摇摆曲线交互展示

支持模型：
- 单机无穷大系统：经典模型 + 详细模型
- 多机系统：经典模型

参考教材：李光琦《电力系统暂态分析》第三、四章
"""

from psa4teaching.stability.transient import (
    TransientStabilityResult,
    simulate_single_machine_infinite_bus_classic,
    simulate_single_machine_infinite_bus_detailed,
    simulate_multi_machine_classic,
)
from psa4teaching.stability.small_signal import (
    SmallSignalResult,
    analyze_single_machine_infinite_bus,
    analyze_multi_machine,
    analyze_multi_machine_detailed,
)
from psa4teaching.stability.equal_area_interactive import (
    EqualAreaParams,
    EqualAreaInteractive,
    show_equal_area_interactive,
)
from psa4teaching.stability.heffron_phillips import (
    HeffronPhillipsResult,
    compute_heffron_phillips_constants,
    sweep_k_constants,
)
from psa4teaching.stability.voltage_stability import (
    PVCurveResult,
    QVCurveResult,
    compute_pv_curve,
    compute_qv_curve,
)

__all__ = [
    # 暂态稳定
    "TransientStabilityResult",
    "simulate_single_machine_infinite_bus_classic",
    "simulate_single_machine_infinite_bus_detailed",
    "simulate_multi_machine_classic",
    # 小干扰稳定
    "SmallSignalResult",
    "analyze_single_machine_infinite_bus",
    "analyze_multi_machine",
    "analyze_multi_machine_detailed",
    # 等面积准则交互工具
    "EqualAreaParams",
    "EqualAreaInteractive",
    "show_equal_area_interactive",
    # Heffron-Phillips K1-K6 常数
    "HeffronPhillipsResult",
    "compute_heffron_phillips_constants",
    "sweep_k_constants",
    # 电压稳定性分析
    "PVCurveResult",
    "QVCurveResult",
    "compute_pv_curve",
    "compute_qv_curve",
]
