"""
psa4teaching - 电力系统分析教学Python包
=======================================

面向本科教学的电力系统分析工具包，覆盖潮流计算、短路计算和稳定计算三大模块。

功能特点
--------
1. **元件模型模块** (models/)
   - 节点（Bus）：PQ/PV/平衡节点
   - 输电线路（Line）：π型等值电路
   - 变压器（Transformer）：非标准变比支持
   - 发电机（Generator）：暂态/次暂态参数
   - 负荷（Load）：恒功率/恒阻抗/恒电流/ZIP模型

2. **网络矩阵模块** (network/)
   - Ybus：节点导纳矩阵构造
   - Zbus：节点阻抗矩阵构造

3. **潮流计算模块** (powerflow/)
   - 牛顿-拉夫逊法
   - P-Q分解法（快速解耦）
   - 直流潮流

4. **短路计算模块** (shortcircuit/)
   - 三相短路电流计算
   - 转移阻抗计算
   - 不对称短路：单相接地、两相短路、两相接地
   - 序网模型
   - GB/T 15544 标准支持

5. **稳定计算模块** (stability/)
   - 暂态稳定时域仿真（单机无穷大+多机）
   - 小干扰稳定分析（特征值法）

参考教材
--------
- 陈珩《电力系统稳态分析》（第三版）
- 李光琦《电力系统暂态分析》（第三版）
- Kundur P. *Power System Stability and Control*. McGraw-Hill, 1994.

安装
----
pip install psa4teaching

或从源码安装：
git clone https://github.com/psa2ss/psa4teaching.git
cd psa4teaching
pip install -e .

快速示例
--------
>>> from psa4teaching import Bus, BusType, Line, Transformer
>>> from psa4teaching.network import build_ybus, build_zbus
>>> from psa4teaching.powerflow import run_newton_raphson
>>>
>>> # 创建节点
>>> buses = [
>>>     Bus(1, "Slack", BusType.SLACK, V_specified=1.05),
>>>     Bus(2, "Gen1", BusType.PV, P_specified=0.5, V_specified=1.02),
>>>     Bus(3, "Load1", BusType.PQ, P_specified=-0.8, Q_specified=-0.3)
>>> ]
>>>
>>> # 创建线路
>>> lines = [
>>>     Line(1, 2, R=0.02, X=0.1, B=0.02),
>>>     Line(2, 3, R=0.03, X=0.15, B=0.03)
>>> ]
>>>
>>> # 构造导纳矩阵
>>> ybus = build_ybus(lines, [])
>>>
>>> # 潮流计算
>>> result = run_newton_raphson(buses, ybus)

许可证
------
MIT License

作者
----
PSA Teaching Team
"""

__version__ = "1.0.0"