"""
PSA4Teaching - 电力系统分析教学Python包
========================================

面向本科教学的电力系统分析工具包，覆盖潮流计算、短路计算和稳定计算三大模块。

参考教材：
    - 陈珩《电力系统稳态分析》（第三版）
    - 李光琦《电力系统暂态分析》（第三版）
"""

__version__ = "1.0.0"
__author__ = "PSA Teaching Team"

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator
from psa4teaching.models.load import Load, LoadModel
from psa4teaching.network.ybus import build_ybus
from psa4teaching.network.zbus import build_zbus

__all__ = [
    "Bus", "BusType", "Line", "Transformer", "Generator", "Load", "LoadModel",
    "build_ybus", "build_zbus",
]
