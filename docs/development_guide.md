# psa4teaching 开发文档

## 目录
1. [概述](#概述)
2. [项目结构](#项目结构)
3. [元件模型](#元件模型)
4. [网络矩阵](#网络矩阵)
5. [潮流计算](#潮流计算)
6. [短路计算](#短路计算)
7. [稳定计算](#稳定计算)
8. [使用示例](#使用示例)
9. [参考教材](#参考教材)

---

## 概述

psa4teaching（Power System Analysis for Teaching）是一个面向本科教学的电力系统分析Python包，覆盖电力系统分析的三大部分：潮流计算、短路计算和稳定计算。

### 设计原则

- **模型与算法分离**：模型类只存数据，算法函数接受模型对象
- **可读性优先**：代码注释充分，变量命名与教材一致
- **教学友好**：中间结果可查，计算过程透明
- **扩展性强**：易于添加新模型和算法

### 依赖

- Python >= 3.8
- NumPy >= 1.20

---

## 项目结构

```
psa4teaching/
├── psa4teaching/
│   ├── __init__.py          # 包入口
│   ├── models/              # 元件模型
│   │   ├── bus.py           # 节点模型
│   │   ├── line.py          # 输电线路模型
│   │   ├── transformer.py   # 变压器模型
│   │   ├── generator.py     # 发电机模型
│   │   └── load.py          # 负荷模型
│   ├── network/             # 网络矩阵
│   │   ├── ybus.py          # 节点导纳矩阵
│   │   └── zbus.py          # 节点阻抗矩阵
│   ├── powerflow/           # 潮流计算
│   │   ├── newton_raphson.py # 牛顿-拉夫逊法
│   │   ├── fast_decoupled.py # P-Q分解法
│   │   └── dc_powerflow.py  # 直流潮流
│   ├── shortcircuit/        # 短路计算
│   │   ├── symmetric.py     # 对称短路
│   │   ├── asymmetric.py    # 不对称短路
│   │   ├── sequence_network.py # 序网模型
│   │   └── gb15544.py       # GB15544标准
│   ├── stability/           # 稳定计算
│   │   ├── transient.py     # 暂态稳定
│   │   └── small_signal.py  # 小干扰稳定
│   └── utils/               # 工具函数
│       └── helpers.py
├── docs/                    # 文档
├── examples/                # 示例
├── tests/                   # 测试
├── setup.py
├── requirements.txt
└── README.md
```

---

## 元件模型

### 节点 (Bus)

节点是电力系统的基本连接点，按已知条件分为三类：

| 类型 | 已知量 | 未知量 |
|------|--------|--------|
| PQ节点 | P, Q | V, δ |
| PV节点 | P, V | Q, δ |
| 平衡节点 | V, δ | P, Q |

```python
from psa4teaching.models.bus import Bus, BusType

# 平衡节点
slack = Bus(number=1, name="Slack", bus_type=BusType.SLACK, V_specified=1.05)

# PV节点（发电机）
gen = Bus(number=2, name="Gen1", bus_type=BusType.PV,
          P_specified=0.5, V_specified=1.02)

# PQ节点（负荷）
load = Bus(number=3, name="Load1", bus_type=BusType.PQ,
           P_specified=-0.8, Q_specified=-0.3)
```

### 输电线路 (Line)

采用π型等值电路，参数包括串联阻抗 R+jX 和对地电纳 jB/2。

```
I₁    ┌── Z=R+jX ──┐    I₂
○──────┤              ├──────○
        └── jB/2 jB/2 ──┘
```

```python
from psa4teaching.models.line import Line

line = Line(from_bus=1, to_bus=2, R=0.02, X=0.1, B=0.02)
```

### 变压器 (Transformer)

支持非标准变比k的π型等值电路模型。

```python
from psa4teaching.models.transformer import Transformer

# 标准变比变压器
tx1 = Transformer(from_bus=1, to_bus=2, RT=0.002, XT=0.1, k=1.0)

# 非标准变比（变比1.05）
tx2 = Transformer(from_bus=1, to_bus=2, RT=0.002, XT=0.1, k=1.05)
```

### 发电机 (Generator)

支持暂态和次暂态参数，用于潮流、短路和稳定计算。

```python
from psa4teaching.models.generator import Generator, GeneratorModelType

gen = Generator(
    bus=1, name="G1",
    Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.2,
    H=6.0, D=1.0,
    model_type=GeneratorModelType.CLASSIC
)
```

### 负荷 (Load)

支持恒功率、恒阻抗、恒电流和ZIP四种模型。

```python
from psa4teaching.models.load import Load, LoadModel

# 恒功率负荷
load1 = Load(bus=3, P0=0.8, Q0=0.3, model_type=LoadModel.CONSTANT_POWER)

# ZIP模型负荷
load2 = Load(bus=4, P0=1.0, Q0=0.4, model_type=LoadModel.ZIP,
             alpha_p=0.3, beta_p=0.3, gamma_p=0.4)
```

---

## 网络矩阵

### 节点导纳矩阵 (Ybus)

Ybus的构造规则：
- 对角元 Yii = Σ（与i相连的支路导纳之和）
- 非对角元 Yij = -yij（i、j之间支路导纳的负值）

```python
from psa4teaching.network import build_ybus

ybus_result = build_ybus(lines, transformers)
print(ybus_result.Ybus)     # 复数矩阵
print(ybus_result.G)         # 电导矩阵
print(ybus_result.B)         # 电纳矩阵
```

### 节点阻抗矩阵 (Zbus)

Zbus = Ybus⁻¹

```python
from psa4teaching.network import build_zbus

zbus_result = build_zbus(lines, transformers)
print(zbus_result.Zbus)      # 复数矩阵
print(zbus_result.R)          # 电阻矩阵
print(zbus_result.X)          # 电抗矩阵
```

---

## 潮流计算

### 牛顿-拉夫逊法

极坐标形式的潮流方程：

$$P_i = V_i \sum_j V_j (G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})$$

$$Q_i = V_i \sum_j V_j (G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})$$

雅可比矩阵：

$$\begin{bmatrix} \Delta P \\ \Delta Q \end{bmatrix} = \begin{bmatrix} H & N \\ J & L \end{bmatrix} \begin{bmatrix} \Delta\theta \\ \Delta V \end{bmatrix}$$

```python
from psa4teaching.powerflow import run_newton_raphson

result = run_newton_raphson(buses, ybus_result, max_iterations=50, tolerance=1e-8)
print(f"收敛: {result.converged}, 迭代: {result.iterations}次")
print(f"电压: {result.V}")
print(f"相角: {np.degrees(result.delta)}")
print(f"损耗: {result.losses}")
```

### P-Q分解法

将雅可比矩阵解耦：

$$\begin{bmatrix} \Delta P/V \\ \Delta Q/V \end{bmatrix} = \begin{bmatrix} B' & 0 \\ 0 & B'' \end{bmatrix} \begin{bmatrix} \Delta\theta \\ \Delta V \end{bmatrix}$$

```python
from psa4teaching.powerflow import run_fast_decoupled

result = run_fast_decoupled(buses, ybus_result)
```

### 直流潮流

线性化模型：P = B × θ

```python
from psa4teaching.powerflow import run_dc_powerflow

result = run_dc_powerflow(buses, lines, transformers)
print(f"相角: {result.theta}")
print(f"支路功率流: {result.branch_flows}")
```

---

## 短路计算

### 三相短路

基于Zbus的短路电流计算：

$$I_f = V_f^{(0)} / Z_{ff}$$

```python
from psa4teaching.shortcircuit import calculate_three_phase_fault

result = calculate_three_phase_fault(
    buses, lines, transformers, generators,
    fault_bus=3, S_base=100, V_base=10.5
)
print(f"短路电流: {result.fault_current_ka:.2f} kA")
print(f"各节点电压: {result.V_pu}")
```

### 不对称短路

基于对称分量法，将三相系统分解为正序、负序、零序三个序网。

```python
from psa4teaching.shortcircuit import (
    calculate_single_line_to_ground,    # 单相接地
    calculate_line_to_line,             # 两相短路
    calculate_double_line_to_ground,    # 两相接地
)

# 单相接地短路
result = calculate_single_line_to_ground(
    buses, lines, transformers, generators, fault_bus=3
)
print(f"故障电流: {result.fault_current}")
print(f"三相电流: {result.fault_currents_3phase}")
print(f"序电流: {result.sequence_currents}")
```

### GB/T 15544 标准

```python
from psa4teaching.shortcircuit import calculate_gb15544

result = calculate_gb15544(
    buses, lines, transformers, generators,
    fault_bus=3, V_nominal=10.5, max_current=True
)
print(f"初始短路电流 I\"k = {result.Ik_initial:.2f} kA")
print(f"峰值短路电流 ip = {result.ip_peak:.2f} kA")
print(f"最小短路电流 = {result.Ik_min:.2f} kA")
```

---

## 稳定计算

### 暂态稳定 - 单机无穷大（经典模型）

二阶运动方程：

$$2H \frac{d^2\delta}{dt^2} = P_m - P_e - D\frac{d\delta}{dt}$$

$$P_e = \frac{E' V_\infty}{X_\Sigma} \sin\delta$$

```python
from psa4teaching.stability import simulate_single_machine_infinite_bus_classic

result = simulate_single_machine_infinite_bus_classic(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=5.0, D=0.0, Pm=0.8, delta_0=np.radians(30),
    fault_time=0.0, fault_clearing_time=0.15,
    t_end=5.0, dt=0.005, method="rk4"
)
print(f"系统{'稳定' if result.stable else '不稳定'}")
print(f"最大功角: {result.max_delta:.2f}°")
```

### 暂态稳定 - 单机无穷大（详细模型）

四阶模型，包含励磁系统和调速系统。

```python
from psa4teaching.stability import simulate_single_machine_infinite_bus_detailed

result = simulate_single_machine_infinite_bus_detailed(
    E_prime_0=1.2, V_infinity=1.0, X_total=0.5,
    Xd=1.8, Xd_prime=0.3, Xq=1.7,
    Td0_prime=8.0, H=5.0, D=0.0,
    Pm_0=0.8, delta_0=np.radians(30), Efd_0=1.5,
    fault_time=0.0, fault_clearing_time=0.15
)
```

### 小干扰稳定分析

通过线性化和特征值分析判断系统稳定性。

```python
from psa4teaching.stability import analyze_single_machine_infinite_bus

result = analyze_single_machine_infinite_bus(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=5.0, D=0.0, delta_0=np.radians(30), Pm=0.8
)
print(f"系统{'稳定' if result.stable else '不稳定'}")
print(f"特征值: {result.eigenvalues}")
print(f"阻尼比: {result.damping_ratios}")
print(f"振荡频率: {result.frequencies} Hz")
```

---

## 使用示例

### 完整的潮流计算示例

```python
import numpy as np
from psa4teaching import Bus, BusType, Line, Transformer
from psa4teaching.network import build_ybus
from psa4teaching.powerflow import run_newton_raphson, run_fast_decoupled

# 1. 定义节点
buses = [
    Bus(1, "Slack", BusType.SLACK, V_specified=1.05),
    Bus(2, "Gen1", BusType.PV, P_specified=0.5, V_specified=1.02),
    Bus(3, "Load1", BusType.PQ, P_specified=-0.8, Q_specified=-0.3),
    Bus(4, "Load2", BusType.PQ, P_specified=-0.6, Q_specified=-0.2),
]

# 2. 定义线路
lines = [
    Line(1, 2, R=0.02, X=0.1, B=0.02),
    Line(2, 3, R=0.03, X=0.15, B=0.03),
    Line(2, 4, R=0.025, X=0.12, B=0.025),
]

# 3. 定义变压器
transformers = [
    Transformer(1, 2, RT=0.002, XT=0.08, k=1.05),
]

# 4. 构造导纳矩阵
ybus_result = build_ybus(lines, transformers)

# 5. 牛顿-拉夫逊法潮流计算
nr_result = run_newton_raphson(buses, ybus_result, verbose=True)

# 6. P-Q分解法潮流计算
fd_result = run_fast_decoupled(buses, ybus_result, verbose=True)

# 7. 结果对比
print(f"\nNR法: 收敛={nr_result.converged}, 迭代={nr_result.iterations}次")
print(f"PQ法: 收敛={fd_result.converged}, 迭代={fd_result.iterations}次")
```

### 完整的短路计算示例

```python
from psa4teaching import Bus, BusType, Line
from psa4teaching.models.generator import Generator
from psa4teaching.shortcircuit import (
    calculate_three_phase_fault,
    calculate_single_line_to_ground,
    calculate_line_to_line,
    calculate_double_line_to_ground,
    calculate_gb15544,
)

buses = [
    Bus(1, "G1", BusType.PV, P_specified=0.5, V_specified=1.05),
    Bus(2, "G2", BusType.PV, P_specified=0.3, V_specified=1.03),
    Bus(3, "Load", BusType.PQ, P_specified=-0.8, Q_specified=-0.3),
]

lines = [
    Line(1, 3, R=0.02, X=0.1, B=0.02),
    Line(2, 3, R=0.025, X=0.12, B=0.025),
]

generators = [
    Generator(bus=1, Xd_doubleprime=0.2, H=6.0),
    Generator(bus=2, Xd_doubleprime=0.25, H=5.0),
]

# 三相短路
result = calculate_three_phase_fault(buses, lines, [], generators, fault_bus=3)
print(f"三相短路电流: {result.fault_current_ka:.2f} kA")

# 单相接地
result = calculate_single_line_to_ground(buses, lines, [], generators, fault_bus=3)
print(f"单相接地故障电流: {result.fault_current:.4f} p.u.")

# GB15544标准
result = calculate_gb15544(buses, lines, [], generators, fault_bus=3,
                           V_nominal=10.5, verbose=True)
```

### 完整的暂态稳定仿真示例

```python
import numpy as np
from psa4teaching.stability import (
    simulate_single_machine_infinite_bus_classic,
    simulate_single_machine_infinite_bus_detailed,
    analyze_single_machine_infinite_bus,
)

# 经典模型仿真
result = simulate_single_machine_infinite_bus_classic(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=5.0, D=2.0, Pm=0.8, delta_0=np.radians(30),
    fault_time=0.0, fault_clearing_time=0.2,
    t_end=5.0, method="rk4"
)
print(f"经典模型: {'稳定' if result.stable else '不稳定'}")

# 详细模型仿真
result = simulate_single_machine_infinite_bus_detailed(
    E_prime_0=1.2, V_infinity=1.0, X_total=0.5,
    Xd=1.8, Xd_prime=0.3, Xq=1.7,
    Td0_prime=8.0, H=5.0, D=2.0,
    Pm_0=0.8, delta_0=np.radians(30), Efd_0=1.5,
    fault_time=0.0, fault_clearing_time=0.2
)
print(f"详细模型: {'稳定' if result.stable else '不稳定'}")

# 小干扰稳定分析
ss_result = analyze_single_machine_infinite_bus(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=5.0, D=2.0, delta_0=np.radians(30), Pm=0.8,
    verbose=True
)
```

---

## 参考教材

1. 陈珩. 电力系统稳态分析（第三版）. 中国电力出版社.
2. 李光琦. 电力系统暂态分析（第三版）. 中国电力出版社.
3. Kundur P. *Power System Stability and Control*. McGraw-Hill, 1994.
4. GB/T 15544-1995 三相交流系统短路电流计算.
5. IEC 60909 Short-circuit currents in three-phase a.c. systems.
