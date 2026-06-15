# psa4teaching 零基础入门教程

> 用 Python 解决电力系统分析课本计算题  
> 适用于：电力系统分析课程初学者（无需 Python 基础）  
> 参考教材：陈珩《电力系统稳态分析》、李光琦《电力系统暂态分析》、Kundur《Power System Stability and Control》

---

## 目录

- [第一课：环境准备——5分钟上手](#第一课环境准备5分钟上手)
- [第二课：Python 速成（只学需要的）](#第二课python-速成只学需要的)
- [第三课：建立你的第一个电力系统](#第三课建立你的第一个电力系统)
- [第四课：潮流计算](#第四课潮流计算)
- [第五课：短路计算](#第五课短路计算)
- [第六课：稳定计算](#第六课稳定计算)
- [第七课：励磁与控制系统](#第七课励磁与控制系统)
- [第八课：电压稳定性分析](#第八课电压稳定性分析)
- [附录A：常见错误与排错](#附录a常见错误与排错)
- [附录B：课本例题对照表](#附录b课本例题对照表)
- [附录C：常用公式速查](#附录c常用公式速查)
- [附录D：完整代码模板](#附录d完整代码模板)
- [附录E：Kundur 教材 Notebook 索引](#附录e-kundur-教材-notebook-索引)

---

## 第一课：环境准备——5分钟上手

### 1.1 安装 Python

**Windows：**
1. 打开浏览器，访问 https://www.python.org/downloads/
2. 点击 "Download Python 3.x.x"
3. 运行下载的安装程序
4. ⚠️ **勾选 "Add Python to PATH"**（非常重要！）
5. 点击 "Install Now"

**验证安装：**
打开"命令提示符"（按 Win+R，输入 `cmd`，回车），输入：
```
python --version
```
如果显示 `Python 3.x.x`，说明安装成功。

### 1.2 安装 psa4teaching

在命令提示符中输入：
```
pip install numpy
pip install psa4teaching
```

> **说明：** `numpy` 是科学计算库，psa4teaching 依赖它。

### 1.3 你的第一个 Python 程序

打开命令提示符，输入 `python` 回车，进入 Python 交互环境（看到 `>>>` 提示符）：

```python
>>> print("Hello, 电力系统！")
Hello, 电力系统！
>>> 2 * 3 + 1
7
>>> import math
>>> math.sqrt(3)
1.7320508075688772
```

输入 `exit()` 退出。

### 1.4 推荐使用 Jupyter Notebook（可选但强烈推荐）

Jupyter Notebook 是一个"笔记本式"的编程环境，可以逐行运行代码、查看结果，非常适合学习和实验。

```
pip install jupyter
jupyter notebook
```

浏览器会自动打开一个网页界面。点击 "New" → "Python 3" 创建新笔记本。

> **本教程后面所有代码都可以在 Jupyter Notebook 中逐行运行。**

---

## 第二课：Python 速成（只学需要的）

> 这一课只教你电力系统计算用得上的 Python 知识，10分钟足够。

### 2.1 变量与赋值

Python 用 `=` 给变量赋值：

```python
V = 1.05          # 电压标幺值
R = 0.02          # 电阻
X = 0.10          # 电抗
P = -0.8          # 有功功率（负号表示消耗）
Q = -0.3          # 无功功率

# 可以做计算
Z = R + X * 1j    # 复数阻抗，1j 表示虚数单位 j
print(Z)          # (0.02+0.1j)
print(abs(Z))     # 阻抗模 0.102
```

> **电力系统小贴士：** Python 中用 `1j` 表示虚数单位，所以 `R + X*1j` 就是 $R + jX$。

### 2.2 列表（方括号）

列表用于存放一组数据：

```python
# 一组节点
buses = [1, 2, 3, 4, 5]

# 取出第1个元素（注意：Python 从 0 开始编号！）
print(buses[0])     # 1
print(buses[1])     # 2

# 列表长度
print(len(buses))   # 5
```

### 2.3 导入模块

要使用 psa4teaching，需要先"导入"它：

```python
# 导入需要的类和函数（核心模块）
from psa4teaching.models import Bus, BusType, Line, Transformer, Generator, Load
from psa4teaching.network import build_ybus, build_zbus
from psa4teaching.powerflow import run_newton_raphson

# 扩展模块（进阶使用）
from psa4teaching.models import GeneratorModelType, LoadModel         # 模型类型枚举
from psa4teaching.models import IEEET1Params, SEXSParams, TGOV1Params  # 控制器模型
from psa4teaching.utils.transform import park_transform                 # 坐标变换
from psa4teaching.data import create_kundur_two_area_system             # 标准测试系统
```

这就像从工具箱里拿出需要的工具。每次运行程序时都要先导入。

### 2.4 调用函数

```python
# 调用函数：名称(参数1, 参数2, ...)
result = run_newton_raphson(buses, ybus_result, max_iterations=50)

# 查看结果
print(result.converged)    # True（收敛）或 False（不收敛）
print(result.V)            # 电压幅值数组
print(result.iterations)   # 迭代次数
```

### 2.5 import numpy as np

电力系统计算大量使用矩阵运算，我们需要 `numpy` 库：

```python
import numpy as np

# 数组（类似数学中的向量）
V = np.array([1.05, 1.02, 0.98, 0.96])

# 常用数学函数
np.sqrt(3)          # √3 = 1.732
np.degrees(0.5236)  # 弧度转角度 = 30.0°
np.radians(30)      # 角度转弧度 = 0.5236
np.pi               # 圆周率 π = 3.14159
```

---

## 第三课：建立你的第一个电力系统

> 在这节课中，我们用代码描述一个简单的 3 节点电力系统。

### 3.1 系统图

```
        线路1          线路2
 G1 ----[R=0.02,jX=0.1]---- 节点2 ----[R=0.03,jX=0.15]---- 节点3
节点1                                              |
(平衡节点)                                         负荷
```

### 3.2 定义节点

电力系统中，节点分为三类：

| 类型 | 英文名 | 含义 | 已知量 | 未知量 |
|------|--------|------|--------|--------|
| 平衡节点 | SLACK | 参考节点，提供功率平衡 | V, δ | P, Q |
| PV 节点 | PV | 发电机节点 | P, V | Q, δ |
| PQ 节点 | PQ | 负荷节点 | P, Q | V, δ |

```python
from psa4teaching.models import Bus, BusType

# 参数说明：Bus(编号, 名称, 类型, ...)
buses = [
    Bus(1, "G1",    BusType.SLACK, V_specified=1.05),        # 节点1：平衡节点，电压1.05
    Bus(2, "Gen1",  BusType.PV,    P_specified=0.5, V_specified=1.02),  # 节点2：PV节点
    Bus(3, "Load1", BusType.PQ,    P_specified=-0.8, Q_specified=-0.3), # 节点3：PQ节点
]
```

> **关键理解：** 
> - `V_specified=1.05` 表示电压幅值为 1.05 p.u.（标幺值）
> - `P_specified=-0.8`：负号表示这是**消耗**的有功（负荷），0.8 p.u.
> - `Q_specified=-0.3`：负号表示**消耗**的无功

### 3.3 定义线路

线路用 π 型等值电路表示，参数为 R（电阻）、X（电抗）、B（电纳）：

```python
from psa4teaching.models import Line

# 参数说明：Line(起始节点, 终止节点, R, X, B)
lines = [
    Line(1, 2, R=0.02, X=0.10, B=0.02),   # 线路1：节点1 → 节点2
    Line(2, 3, R=0.03, X=0.15, B=0.03),   # 线路2：节点2 → 节点3
]
```

> **参数含义：**
> - `R`：线路电阻（标幺值），反映有功损耗
> - `X`：线路电抗（标幺值），反映电压降落
> - `B`：线路总电纳（标幺值），反映线路充电电容效应

### 3.4 定义变压器（含非标准变比）

```python
from psa4teaching.models import Transformer

# 参数说明：Transformer(高压侧节点, 低压侧节点, R, X, k)
# k = 非标准变比，k=1.0 为标准变比
transformers = [
    Transformer(1, 2, RT=0.002, XT=0.10, k=1.05),
]
```

### 3.5 定义发电机（用于短路和稳定计算）

```python
from psa4teaching.models import Generator

# 参数说明：Generator(bus=所在节点, name=名称, ...)
generators = [
    Generator(bus=1, name="G1", Xd_doubleprime=0.20, Xd_prime=0.30, H=6.0, D=1.0),
    Generator(bus=2, name="G2", Xd_doubleprime=0.25, Xd_prime=0.35, H=5.0, D=1.0),
]
```

> **参数含义：**
> - `Xd_doubleprime`：次暂态电抗 $X_d''$，用于**短路计算**
> - `Xd_prime`：暂态电抗 $X_d'$，用于**稳定计算**
> - `H`：惯性时间常数（秒），典型值 3~10 秒
> - `D`：阻尼系数

**控制器模型（进阶）：**

```python
from psa4teaching.models import IEEET1Params, TGOV1Params, PSS2AParams

# IEEE Type 1 励磁系统
exciter = IEEET1Params(KA=200, TA=0.02, TE=0.5, KF=0.05, TF=1.0)

# TGOV1 汽轮机调速器
governor = TGOV1Params(R=0.05, T1=0.5, T2=3.0, T3=10.0)

# PSS2A 电力系统稳定器
pss = PSS2AParams(KS1=10.0, TW1=2.0, T1=0.25, T2=0.03)
```

> - IEEET1 是 Kundur 教材中最常用的励磁模型
> - 控制器模型提供 `compute()` 和 `compute_rk4()` 两种积分方法
> - 详细使用见第七课"励磁与控制系统"

### 3.6 完整示例：打印导纳矩阵

把以上所有部分组合起来：

```python
from psa4teaching.models import Bus, BusType, Line
from psa4teaching.network import build_ybus

# 1. 定义节点
buses = [
    Bus(1, "G1",    BusType.SLACK, V_specified=1.05),
    Bus(2, "Gen1",  BusType.PV,    P_specified=0.5, V_specified=1.02),
    Bus(3, "Load1", BusType.PQ,    P_specified=-0.8, Q_specified=-0.3),
]

# 2. 定义线路
lines = [
    Line(1, 2, R=0.02, X=0.10, B=0.02),
    Line(2, 3, R=0.03, X=0.15, B=0.03),
]

# 3. 构造导纳矩阵
import numpy as np
ybus_result = build_ybus(lines, [])

# 4. 打印结果
print("节点导纳矩阵 Ybus：")
print(np.round(ybus_result.Ybus, 4))
print(f"节点数：{ybus_result.n_bus}")
```

运行输出（示意）：
```
节点导纳矩阵 Ybus：
[[-1.9416+9.7078j  1.9416-9.7078j  0.0000+0.0000j]
 [ 1.9416-9.7078j -3.6413+17.7165j  1.6997-8.0087j]
 [ 0.0000+0.0000j  1.6997-8.0087j -1.6997+8.0087j]]
节点数：3
```

> **如何读懂结果：** `Ybus[0,0]` 是节点1的自导纳，`Ybus[0,1]` 是节点1和2之间的互导纳（负号表示连接关系）。

---

## 第四课：潮流计算

> 潮流计算是电力系统分析的核心计算。给定网络结构和负荷/发电数据，求出各节点的电压和功率分布。

### 4.1 牛顿-拉夫逊法（最常用）

```python
from psa4teaching.models import Bus, BusType, Line
from psa4teaching.network import build_ybus
from psa4teaching.powerflow import run_newton_raphson
import numpy as np

# === 定义系统 ===
buses = [
    Bus(1, "Slack", BusType.SLACK, V_specified=1.05),
    Bus(2, "Gen1",  BusType.PV,    P_specified=0.5,  V_specified=1.02),
    Bus(3, "Load1", BusType.PQ,    P_specified=-0.8, Q_specified=-0.3),
    Bus(4, "Load2", BusType.PQ,    P_specified=-0.6, Q_specified=-0.2),
]

lines = [
    Line(1, 2, R=0.02, X=0.10, B=0.02),
    Line(2, 3, R=0.03, X=0.15, B=0.03),
    Line(2, 4, R=0.025, X=0.12, B=0.025),
    Line(3, 4, R=0.04, X=0.20, B=0.04),
]

# === 构造导纳矩阵 ===
ybus_result = build_ybus(lines, [])

# === 执行潮流计算 ===
result = run_newton_raphson(
    buses, ybus_result,
    max_iterations=50,      # 最大迭代次数
    tolerance=1e-8,          # 收敛精度
    verbose=True             # 打印迭代过程
)

# === 查看结果 ===
print(f"\n是否收敛：{result.converged}")
print(f"迭代次数：{result.iterations}")

print("\n各节点电压：")
print(f"{'节点':>4s}  {'名称':>6s}  {'电压幅值':>8s}  {'电压相角(°)':>10s}")
for bus in buses:
    idx = list(ybus_result.bus_indices.values())[buses.index(bus)]
    V = result.V[idx]
    delta = np.degrees(result.delta[idx])
    print(f"{bus.number:>4d}  {bus.name:>6s}  {V:>8.4f}  {delta:>10.2f}")

print(f"\n网络损耗：{result.losses:.4f} p.u. = {result.losses * 100:.2f} MW（100MVA基准）")
```

**运行结果（示意）：**
```
迭代 1: 最大偏差 = 8.00e-01
迭代 2: 最大偏差 = 1.23e-01
迭代 3: 最大偏差 = 5.67e-04
迭代 4: 最大偏差 = 1.02e-08

是否收敛：True
迭代次数：4

各节点电压：
 节点    名称    电压幅值  电压相角(°)
   1    Slack    1.0500       0.00
   2     Gen1    1.0200      -2.34
   3    Load1    0.9854      -4.67
   4    Load2    0.9712      -5.12

网络损耗：0.0253 p.u. = 2.53 MW（100MVA基准）
```

> **课本联系：** 这就是陈珩《电力系统稳态分析》第四章的牛顿-拉夫逊潮流计算。

### 4.2 P-Q 分解法（更快但精度略低）

```python
from psa4teaching.powerflow import run_fast_decoupled

result_fd = run_fast_decoupled(buses, ybus_result, max_iterations=100, tolerance=1e-8)
print(f"P-Q分解法：收敛={result_fd.converged}，迭代={result_fd.iterations}次")
```

> **对比：** 牛顿-拉夫逊法通常 4~6 次迭代收敛，P-Q 分解法需要 10~30 次，但每次计算量更小。

### 4.3 直流潮流（最简单，只算有功）

直流潮流忽略电阻和无功，只关心有功和相角的关系：

$$P_{ij} \approx \frac{\theta_i - \theta_j}{X_{ij}}$$

```python
from psa4teaching.powerflow import run_dc_powerflow

result_dc = run_dc_powerflow(buses, lines, [])

print("各节点电压相角：")
for bus in buses:
    idx = list(result_dc.bus_indices.values())[buses.index(bus)]
    theta_deg = np.degrees(result_dc.theta[idx])
    print(f"  节点{bus.number}: {theta_deg:.2f}°")

print("\n各支路有功功率：")
for (i, j), P in result_dc.branch_flows.items():
    print(f"  {i}→{j}: {P:.4f} p.u. = {P*100:.2f} MW")
```

> **什么时候用哪个？**
> - **精确计算** → 牛顿-拉夫逊法
> - **快速估算** → P-Q 分解法或直流潮流
> - **规划方案比选** → 直流潮流（可以一次算很多方案）

### 4.4 实战：课本例题手算验证

假设课本有一道潮流计算题：
- 3 节点系统
- 节点1为平衡节点，$V_1 = 1.05 \angle 0°$
- 节点2为 PV 节点，$P_2 = 0.5$，$V_2 = 1.02$
- 节点3为 PQ 节点，$P_3 = -0.8$，$Q_3 = -0.3$
- 线路参数：1-2 (R=0.02, X=0.1, B=0.02)，2-3 (R=0.03, X=0.15, B=0.03)

```python
# 课本例题验证代码
buses = [
    Bus(1, "平衡", BusType.SLACK, V_specified=1.05),
    Bus(2, "PV",   BusType.PV,    P_specified=0.5, V_specified=1.02),
    Bus(3, "PQ",   BusType.PQ,    P_specified=-0.8, Q_specified=-0.3),
]
lines = [
    Line(1, 2, R=0.02, X=0.10, B=0.02),
    Line(2, 3, R=0.03, X=0.15, B=0.03),
]

ybus = build_ybus(lines, [])
result = run_newton_raphson(buses, ybus, verbose=True)

# 打印结果，和课本答案对比
for bus in buses:
    idx = list(ybus.bus_indices.keys()).index(bus.number)
    print(f"V_{bus.number} = {result.V[idx]:.4f} ∠ {np.degrees(result.delta[idx]):.2f}°")
```

> **学习方法：** 把你课本上的题目参数改一改，运行程序，对比手算结果。如果一致，说明你理解了！

---

## 第五课：短路计算

> 短路计算用于确定短路电流大小，是选择电气设备和整定保护装置的依据。

### 5.1 三相短路（最基本）

三相短路是最严重的短路类型，电流最大。

```python
from psa4teaching.models import Bus, BusType, Line, Generator
from psa4teaching.network import build_ybus
from psa4teaching.shortcircuit import calculate_three_phase_fault

# === 定义系统 ===
buses = [
    Bus(1, "G1",   BusType.SLACK, V_specified=1.05),
    Bus(2, "G2",   BusType.PV,    P_specified=0.3, V_specified=1.03),
    Bus(3, "Load", BusType.PQ,    P_specified=-0.8, Q_specified=-0.3),
]

lines = [
    Line(1, 3, R=0.02, X=0.10, B=0.02),
    Line(2, 3, R=0.025, X=0.12, B=0.025),
    Line(1, 2, R=0.03, X=0.15, B=0.03),
]

generators = [
    Generator(bus=1, name="G1", Xd_doubleprime=0.20, H=6.0),
    Generator(bus=2, name="G2", Xd_doubleprime=0.25, H=5.0),
]

# === 在节点3发生三相短路 ===
fault_bus = 3

result = calculate_three_phase_fault(
    buses, lines, [],           # 无变压器
    generators,                  # 发电机列表
    fault_bus=fault_bus,         # 短路点
    S_base=100.0,                # 系统基准容量 100 MVA
    V_base=10.5,                 # 短路点基准电压 10.5 kV
    verbose=True
)

# === 查看结果 ===
print(f"\n{'='*40}")
print(f"短路点：节点{result.fault_bus}")
print(f"短路电流：{result.fault_current_ka:.2f} kA")
print(f"短路电流：{abs(result.fault_current):.4f} p.u.")
print(f"短路点自阻抗：{result.Zff:.4f} p.u.")
print(f"\n短路后各节点残压：")
for bus_num, V in result.V_pu.items():
    print(f"  节点{bus_num}：{V:.4f} p.u.")
print(f"\n发电机对短路电流的贡献（转移阻抗）：")
for bus_num, Z in result.transfer_impedances.items():
    print(f"  发电机{bus_num} → 短路点：Z = {Z:.4f} p.u.")
```

> **课本联系：** 对应李光琦《电力系统暂态分析》第二章的三相短路计算。  
> **核心公式：** $I_f = V_f / Z_{ff}$，即短路电流 = 短路前电压 / 短路点自阻抗。

### 5.2 不对称短路（单相接地、两相短路、两相接地）

不对称短路需要用**对称分量法**，程序内部已自动处理。

```python
from psa4teaching.shortcircuit import (
    calculate_single_line_to_ground,    # 单相接地短路
    calculate_line_to_line,             # 两相短路
    calculate_double_line_to_ground,    # 两相接地短路
)

fault_bus = 3

# ① 单相接地短路（a相接地）
print("=== 单相接地短路（a相）===")
result_slg = calculate_single_line_to_ground(
    buses, lines, [], generators, fault_bus=fault_bus
)
print(f"短路电流 Ia = {abs(result_slg.fault_current):.4f} p.u.")
print(f"序电流：I₁={abs(result_slg.sequence_currents[1]):.4f}，"
      f"I₂={abs(result_slg.sequence_currents[2]):.4f}，"
      f"I₀={abs(result_slg.sequence_currents[0]):.4f}")
print(f"三相电流：Ia={abs(result_slg.fault_currents_3phase[0]):.4f}，"
      f"Ib={abs(result_slg.fault_currents_3phase[1]):.4f}，"
      f"Ic={abs(result_slg.fault_currents_3phase[2]):.4f}")

# ② 两相短路（b、c相短路）
print("\n=== 两相短路（b、c相）===")
result_ll = calculate_line_to_line(buses, lines, [], generators, fault_bus=fault_bus)
print(f"三相电流：Ia={abs(result_ll.fault_currents_3phase[0]):.4f}，"
      f"Ib={abs(result_ll.fault_currents_3phase[1]):.4f}，"
      f"Ic={abs(result_ll.fault_currents_3phase[2]):.4f}")

# ③ 两相接地短路（b、c相接地）
print("\n=== 两相接地短路（b、c相）===")
result_dlg = calculate_double_line_to_ground(buses, lines, [], generators, fault_bus=fault_bus)
print(f"短路电流 Ia = {abs(result_dlg.fault_current):.4f} p.u.")
print(f"三相电流：Ia={abs(result_dlg.fault_currents_3phase[0]):.4f}，"
      f"Ib={abs(result_dlg.fault_currents_3phase[1]):.4f}，"
      f"Ic={abs(result_dlg.fault_currents_3phase[2]):.4f}")
```

> **对称分量法速记：**
> | 故障类型 | 序网连接 |
> |----------|----------|
> | 单相接地 | 正序-负序-零序 **串联** |
> | 两相短路 | 正序-负序 **并联** |
> | 两相接地 | 正序与(负序‖零序) **并联** |

### 5.3 带 GB/T 15544 标准的计算

```python
from psa4teaching.shortcircuit import calculate_gb15544

result_gb = calculate_gb15544(
    buses, lines, [], generators,
    fault_bus=fault_bus,
    V_nominal=10.5,    # 额定电压 kV
    max_current=True,  # 计算最大短路电流
    verbose=True
)
```

---

## 第六课：稳定计算

> 稳定计算分析电力系统在受到扰动后能否恢复稳定运行。

### 6.1 小干扰稳定分析（特征值法）

小干扰稳定分析回答的问题是：系统在正常运行点附近受到小扰动后，能否恢复稳定？

```python
from psa4teaching.stability import analyze_single_machine_infinite_bus
import numpy as np

# 单机无穷大系统参数
E_prime = 1.2      # 暂态电势 E'
V_inf   = 1.0      # 无穷大母线电压
X_total = 0.5      # 等值电抗
H       = 5.0      # 惯性时间常数（秒）
D       = 2.0      # 阻尼系数
delta_0 = np.radians(30)  # 初始功角 30°
Pm      = 0.8      # 机械功率

# 经典模型分析
result = analyze_single_machine_infinite_bus(
    E_prime, V_inf, X_total, H, D, delta_0, Pm,
    detailed=False,    # False=经典模型，True=详细模型
    verbose=True
)

print(f"\n系统{'稳定 ✓' if result.stable else '不稳定 ✗'}")
print(f"特征值：{result.eigenvalues}")
print(f"阻尼比：{result.damping_ratios}")
print(f"振荡频率：{result.frequencies} Hz")
```

> **如何判断稳定？**
> - 所有特征值实部 < 0 → **稳定**
> - 阻尼比 ζ > 0 → 稳定（工程上要求 ζ > 0.03~0.05）
> - 如果 ζ < 0，系统振荡会越来越大，最终失稳

### 6.2 暂态稳定仿真（单机无穷大系统）

暂态稳定分析回答的问题是：系统发生严重故障（如三相短路）后，能否保持同步运行？

```python
from psa4teaching.stability import simulate_single_machine_infinite_bus_classic
import numpy as np

# 系统参数
E_prime = 1.2
V_inf   = 1.0
X_total = 0.5
H       = 5.0
D       = 2.0
Pm      = 0.8
delta_0 = np.radians(30)

# === 仿真：故障在0秒发生，0.15秒清除 ===
result = simulate_single_machine_infinite_bus_classic(
    E_prime=E_prime,
    V_infinity=V_inf,
    X_total=X_total,
    H=H, D=D, Pm=Pm, delta_0=delta_0,
    fault_time=0.0,                 # 故障发生时刻（秒）
    fault_clearing_time=0.15,        # 故障清除时刻（秒）
    t_end=5.0,                       # 仿真时长（秒）
    dt=0.005,                        # 仿真步长（秒）
    method="rk4",                    # 积分方法："rk4"（龙格-库塔）或 "euler"
    stability_limit=150.0            # 功角超过150°认为失稳
)

print(f"系统{'稳定 ✓' if result.stable else '不稳定 ✗'}")
print(f"最大功角：{result.max_delta:.2f}°")
print(f"仿真时长：{len(result.time)} 步，共 {result.time[-1]:.1f} 秒")

# 打印关键时间点的功角
print(f"\n功角变化：")
for i in [0, 10, 50, 100, 200, 500]:
    if i < len(result.time):
        print(f"  t={result.time[i]:.3f}s → δ={result.delta[i]:.2f}°，"
              f"ω={result.omega[i]:.4f}，Pe={result.Pe[i]:.4f}")
```

> **课本联系：** 对应李光琦《电力系统暂态分析》第四章。  
> **关键概念：** 故障清除时间越短，系统越容易保持稳定。

### 6.3 寻找临界清除时间

临界清除时间是系统能保持稳定的**最大**故障清除时间。

```python
# 用二分法搜索临界清除时间
import numpy as np

def find_critical_clearing_time(
    E_prime, V_inf, X_total, H, D, Pm, delta_0,
    t_low=0.05, t_high=1.0, tol=0.005
):
    """二分法搜索临界清除时间"""
    while t_high - t_low > tol:
        t_mid = (t_low + t_high) / 2
        result = simulate_single_machine_infinite_bus_classic(
            E_prime, V_inf, X_total, H, D, Pm, delta_0,
            fault_time=0.0, fault_clearing_time=t_mid,
            t_end=5.0, dt=0.005, stability_limit=150.0
        )
        if result.stable:
            t_low = t_mid
        else:
            t_high = t_mid
    return (t_low + t_high) / 2

cct = find_critical_clearing_time(E_prime, V_inf, X_total, H, D, Pm, delta_0)
print(f"临界清除时间：{cct:.3f} 秒")
```

> **工程意义：** 实际的继电保护必须在临界清除时间内切除故障，否则系统将失稳。

### 6.4 详细模型仿真（含励磁系统）

```python
from psa4teaching.stability import simulate_single_machine_infinite_bus_detailed
import numpy as np

result = simulate_single_machine_infinite_bus_detailed(
    E_prime_0=1.2, V_infinity=1.0, X_total=0.5,
    Xd=1.8, Xd_prime=0.3, Xq=1.7,     # 同步电抗参数
    Td0_prime=8.0,                       # 暂态时间常数
    H=5.0, D=2.0,                        # 机械参数
    Pm_0=0.8, delta_0=np.radians(30),
    Efd_0=2.0,                           # 初始励磁电压
    Ka=50.0, Ta=0.05, Te=0.3,           # 励磁系统参数
    Tg=0.5,                              # 调速器时间常数
    fault_time=0.0, fault_clearing_time=0.15,
    t_end=5.0
)

print(f"系统{'稳定 ✓' if result.stable else '不稳定 ✗'}")
print(f"最大功角：{result.max_delta:.2f}°")
```

> **经典模型 vs 详细模型：**
> | 特点 | 经典模型 | 详细模型 |
> |------|----------|----------|
> | 状态变量 | δ, ω（2个） | δ, ω, Eq', Efd, Pm（5个） |
> | 励磁系统 | ❌ | ✅ |
> | 调速系统 | ❌ | ✅ |
> | 计算速度 | 快 | 慢 |
> | 适用场景 | 初步分析 | 精确分析 |

### 6.5 Heffron-Phillips K1-K6 分析（Kundur Ch.12）

K1-K6 常数是分析励磁系统对电力系统小干扰稳定影响的经典工具：

```python
from psa4teaching.stability import compute_heffron_phillips_constants, sweep_k_constants
import numpy as np

# SMIB 系统 Heffron-Phillips 线性化
result = compute_heffron_phillips_constants(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    Xd=1.8, Xd_prime=0.3, Xq=1.7,
    Td0_prime=8.0, H=5.0, D=0.0,
    delta_0=np.radians(30), Ka=50, Te=0.3,
    verbose=True
)

print(f"K1 = {result.K1:+.4f}  (同步转矩系数)")
print(f"K5 = {result.K5:+.4f}  (dVt/ddelta)")
print(f"K5 < 0 → AVR 引入负阻尼风险！")
print(f"机电模式: f = {result.frequencies[0]:.2f} Hz")
```

### 6.6 多机系统小干扰稳定

```python
from psa4teaching.stability import analyze_multi_machine, analyze_multi_machine_detailed
from psa4teaching.data import create_kundur_two_area_system
from psa4teaching.network import build_ybus
import numpy as np

# 加载 Kundur 两区域系统
sys = create_kundur_two_area_system()
ybus = build_ybus(sys['lines'], sys['transformers'])
gen_idx = [ybus.bus_indices[b] for b in [1, 2, 3, 4]]
Y_gen = ybus.Ybus[np.ix_(gen_idx, gen_idx)]

# 经典模型多机特征值分析
result = analyze_multi_machine(
    E_primes=[1.05, 1.03, 1.03, 1.01],
    H_list=[6.5, 6.5, 6.175, 6.175],
    D_list=[0.0, 0.0, 0.0, 0.0],
    delta_0_list=[np.radians(15), np.radians(10),
                   np.radians(-5), np.radians(-10)],
    Ybus_reduced=Y_gen, verbose=True
)

# 参与因子分析
pf = result.participation_factors
print(f"区域间模式参与因子: {pf[:, 0]}")
```

### 6.7 多机系统暂态稳定

```python
from psa4teaching.stability import simulate_multi_machine_classic

# 两区域系统故障仿真
result = simulate_multi_machine_classic(
    E_primes=[1.05, 1.03, 1.03, 1.01],
    H_list=[6.5, 6.5, 6.175, 6.175],
    D_list=[0.0, 0.0, 0.0, 0.0],
    Pm_list=[7.0, 7.0, 7.19, 7.0],
    delta_0_list=list(np.radians([20, 15, 0, -5])),
    Ybus_reduced=Y_gen,
    fault_time=0.1, fault_clearing_time=0.2,
    t_end=10.0, dt=0.01
)

# result.delta 形状为 [n_steps, n_gen]
# result.delta[:, 0] = G1 功角轨迹
print(f"系统{'稳定' if result.stable else '不稳定'}")
print(f"最大功角: {result.max_delta:.2f}°")
```

### 6.8 PV/QV 电压稳定性曲线

```python
from psa4teaching.stability import compute_pv_curve, compute_qv_curve
import copy

# PV 曲线 — 追踪负荷增长时的电压变化
pv = compute_pv_curve(
    copy.deepcopy(buses), lines, transformers, generators, loads,
    target_bus=3, lambda_max=3.0, n_points=50, verbose=True
)
print(f"临界负荷因子: {pv.critical_lambda:.2f}")
print(f"鼻点电压: {pv.V_curve[pv.nose_point_index, :].min():.4f} pu")

# QV 曲线 — 评估无功裕度
qv = compute_qv_curve(
    copy.deepcopy(buses), lines, transformers, generators, loads,
    target_bus=3, V_range=(0.4, 1.2), n_points=40
)
print(f"最小无功需求: {qv.Q_min:.4f} pu at V={qv.V_at_Qmin:.4f} pu")
```

---

## 第七课：励磁与控制系统

> 励磁系统、调速器和电力系统稳定器 (PSS) 是发电机控制的核心。

### 7.1 IEEET1 励磁系统

```python
from psa4teaching.models import IEEET1Params
import numpy as np

# 创建 IEEET1 励磁系统（晶闸管型快速励磁）
exc = IEEET1Params(
    KA=200, TA=0.02,      # 电压调节器
    KE=1.0, TE=0.5,       # 励磁机
    KF=0.05, TF=1.0,      # 稳定反馈
    VR_MIN=-1.0, VR_MAX=5.0,
    Efd_MIN=0.0, Efd_MAX=5.0
)

# 时域仿真
state = None
for t in np.linspace(0, 2, 2000):
    V_ref = 1.05 if t >= 0.05 else 1.0  # 阶跃
    efd, state = exc.compute(V_ref=V_ref, V_measured=1.0,
                              V_S=0.0, dt=0.001, state=state)
    # efd 为励磁电压输出
```

### 7.2 TGOV1 调速器

```python
from psa4teaching.models import TGOV1Params

gov = TGOV1Params(R=0.05, T1=0.5, T2=3.0, T3=10.0)
P_mech, state = gov.compute(delta_omega=0.01, dt=0.01, state=None)
```

### 7.3 PSS2A 电力系统稳定器

```python
from psa4teaching.models import PSS2AParams

pss = PSS2AParams(KS1=10.0, TW1=2.0, T1=0.25, T2=0.03)
V_S, state = pss.compute_stabilizing_signal(
    delta_omega=0.01, P_gen=0.8, dt=0.01, state=None
)
```

### 7.4 坐标变换 (Park 变换)

```python
from psa4teaching.utils.transform import park_transform, inv_park_transform
import numpy as np

theta = np.pi / 6  # 转子角度 30°
id_val, iq_val, i0 = park_transform(1.0, -0.5, -0.5, theta)
# 平衡三相 → id≈1.0, iq≈0.0, i0≈0.0
```

---

## 第八课：电压稳定性分析

> 电压稳定性是重负荷、长距离输电系统的关键问题。

### 8.1 PV 曲线（鼻点检测）

```python
from psa4teaching.stability import compute_pv_curve
import copy

pv = compute_pv_curve(
    copy.deepcopy(buses), lines, transformers, generators, loads,
    target_bus=3, lambda_max=3.0, n_points=50, verbose=True
)

# 绘图
import matplotlib.pyplot as plt
plt.plot(pv.P_total, pv.V_curve[:, 2])  # 节点3的PV曲线
plt.scatter([pv.P_total[pv.nose_point_index]],
           [pv.V_curve[pv.nose_point_index, 2]],
           color='red', s=100, label='鼻点 (临界点)')
plt.xlabel('Total Load Power (pu)'); plt.ylabel('Voltage (pu)')
plt.title(f'PV Curve - Critical lambda = {pv.critical_lambda:.2f}')
plt.legend(); plt.grid(True, alpha=0.3); plt.show()
```

### 8.2 QV 曲线（无功裕度）

```python
from psa4teaching.stability import compute_qv_curve

qv = compute_qv_curve(
    copy.deepcopy(buses), lines, transformers, generators, loads,
    target_bus=3, V_range=(0.3, 1.2), n_points=40
)
print(f"最小无功需求 Q_min = {qv.Q_min:.4f} pu")
print(f"对应电压 V = {qv.V_at_Qmin:.4f} pu")
print(f"无功裕度 = {qv.reactive_margin:.4f} pu")
```

> **关键理解：**
> - PV 曲线的上半支是稳定运行区，下半支不稳定
> - QV 曲线最低点对应最大无功传输能力
> - 恒功率负荷比恒阻抗负荷更不利于电压稳定

## 附录A：常见错误与排错

### A.1 安装问题

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `'pip' 不是内部或外部命令` | Python 未加入 PATH | 重新安装，勾选"Add to PATH" |
| `No module named 'psa4teaching'` | 未安装程序包 | `pip install psa4teaching` |
| `No module named 'numpy'` | 未安装 numpy | `pip install numpy` |

### A.2 运行问题

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `NameError: name 'Bus' is not defined` | 忘记导入 | 加 `from psa4teaching.models import Bus, BusType` |
| `LinAlgError: Singular matrix` | 雅可比矩阵奇异 | 检查系统是否有孤立节点或参数错误 |
| `ValueError: 电抗X必须为正` | X 值设成了 0 或负数 | 线路电抗不能为 0 |
| 潮流不收敛 | 参数不合理或初值不好 | 减小步长、检查节点类型设置 |

### A.3 调试技巧

```python
# 1. 打开 verbose 模式查看迭代过程
result = run_newton_raphson(buses, ybus, verbose=True)

# 2. 检查导纳矩阵是否正确
print(np.round(ybus_result.Ybus, 4))

# 3. 检查节点编号是否连续
bus_numbers = [bus.number for bus in buses]
print(f"节点编号：{bus_numbers}")
```

---

## 附录B：课本例题对照表

| 功能 | 参考教材 | 典型例题位置 |
|------|----------|-------------|
| 导纳矩阵构造 | 陈珩《稳态分析》第二章 | 例2-1, 例2-2 |
| 牛顿-拉夫逊潮流 | 陈珩《稳态分析》第四章 | 例4-1, 例4-2, 例4-3 |
| P-Q 分解潮流 | 陈珩《稳态分析》第四章 | 例4-4 |
| 直流潮流 | 陈珩《稳态分析》第四章 | 例4-5 |
| 三相短路 | 李光琦《暂态分析》第二章 | 例2-1, 例2-2 |
| 不对称短路 | 李光琦《暂态分析》第二章 | 例2-3, 例2-4, 例2-5 |
| 小干扰稳定 | 李光琦《暂态分析》第四章 | 例4-1 |
| 暂态稳定 | 李光琦《暂态分析》第三章 | 例3-1, 例3-2 |
| Park 变换 / dq0 | Kundur Ch.3 | Ex 3.1-3.8 |
| 同步电机模型 | Kundur Ch.4 | Ex 4.1-4.3 |
| IEEET1 励磁系统 | Kundur Ch.8 | Sec 8.6 |
| Heffron-Phillips K1-K6 | Kundur Ch.12 | Ex 12.1-12.4 |
| 多机特征值分析 | Kundur Ch.12 | Ex 12.5-12.6 |
| 等面积定则 / CCT | Kundur Ch.13 | Ex 13.1-13.4 |
| PV/QV 电压稳定性 | Kundur Ch.14 | Ex 14.1 |
| 次同步谐振 SSR | Kundur Ch.15 | Sec 15.2-15.4 |

---

## 附录C：常用公式速查

### C.1 标幺值换算

$$Z_{pu} = Z_{actual} \cdot \frac{S_B}{V_B^2}$$

### C.2 潮流方程

$$P_i = V_i \sum_j V_j(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})$$

$$Q_i = V_i \sum_j V_j(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})$$

### C.3 短路电流

三相短路：$I_f = V_f / Z_{ff}$

单相接地：$I_f = 3 I_{a0} = \frac{3E_a}{Z_1 + Z_2 + Z_0}$

两相短路：$I_f = \sqrt{3} I_{a1} = \frac{\sqrt{3} E_a}{Z_1 + Z_2}$

### C.4 转子运动方程

$$\frac{d\delta}{dt} = \omega - \omega_s$$

$$\frac{d\omega}{dt} = \frac{P_m - P_e - D(\omega - \omega_s)}{2H}$$

### C.5 特征值与阻尼

特征值：$\lambda = \sigma \pm j\omega_d$

阻尼比：$\zeta = -\sigma / \sqrt{\sigma^2 + \omega_d^2}$

振荡频率：$f = \omega_d / (2\pi)$

稳定条件：$\zeta > 0$（工程要求 $\zeta > 0.03 \sim 0.05$）

### C.6 Heffron-Phillips K1-K6 常数

$$K_1 = \frac{\partial P_e}{\partial \delta} = \frac{E'V_\infty}{X_{d\Sigma}}\cos\delta_0 + V_\infty^2\frac{X_d'-X_q}{X_{d\Sigma}X_{q\Sigma}}\cos 2\delta_0$$

$$K_2 = \frac{\partial P_e}{\partial E_q'} = \frac{V_\infty}{X_{d\Sigma}}\sin\delta_0$$

$$K_3 = \frac{X_{d\Sigma}}{X_d + X_e}$$

$$K_4 = \frac{\partial E_q}{\partial \delta} = V_\infty\frac{X_d-X_d'}{X_{d\Sigma}}\sin\delta_0$$

$$K_5 = \frac{\partial V_t}{\partial \delta} = \frac{V_{d0}X_q\cos\delta_0}{V_{t0}X_{q\Sigma}} - \frac{V_{q0}X_d'\sin\delta_0}{V_{t0}X_{d\Sigma}}$$

$$K_6 = \frac{\partial V_t}{\partial E_q'} = \frac{V_{q0}X_e}{V_{t0}X_{d\Sigma}}$$

> 当 $K_5 < 0$ 时，AVR 通过励磁通道引入负阻尼 → 需要 PSS。

### C.7 电压稳定 PV 曲线

负荷裕度：$\lambda_{crit} = \frac{P_{max} - P_0}{P_0}$

鼻点条件：$\frac{dV}{dP} \to \infty$（潮流雅可比矩阵奇异）

### C.8 SSR 电气谐振频率

$$f_{er} = f_0 \sqrt{\frac{X_C}{X_d'' + X_T + X_L}} = f_0 \sqrt{K}$$

其中 $K = X_C/X_L$ 为串联补偿度。

---

## 附录D：完整代码模板

将以下代码保存为 `my_power_system.py`，修改参数后直接运行：

```python
"""
psa4teaching 快速使用模板
========================
复制此文件，修改系统参数后运行。
"""

# ========== 导入 ==========
from psa4teaching.models import Bus, BusType, Line, Transformer, Generator, Load
from psa4teaching.models import IEEET1Params, TGOV1Params  # 控制器模型
from psa4teaching.network import build_ybus, build_zbus
from psa4teaching.powerflow import run_newton_raphson
from psa4teaching.shortcircuit import (
    calculate_three_phase_fault,
    calculate_single_line_to_ground,
    calculate_line_to_line,
    calculate_double_line_to_ground,
)
from psa4teaching.stability import (
    analyze_single_machine_infinite_bus,
    simulate_single_machine_infinite_bus_classic,
    simulate_single_machine_infinite_bus_detailed,
    simulate_multi_machine_classic,
    compute_heffron_phillips_constants,
    compute_pv_curve, compute_qv_curve,
)
import numpy as np
import copy

# ========== 在这里定义你的系统 ==========

# 节点
buses = [
    # 编号  名称        类型           电压设定  有功设定   无功设定
    Bus(1, "Slack",  BusType.SLACK, V_specified=1.05),
    Bus(2, "Gen",    BusType.PV,    P_specified=0.5, V_specified=1.02),
    Bus(3, "Load",   BusType.PQ,    P_specified=-0.8, Q_specified=-0.3),
]

# 线路
lines = [
    # 起始  终止  R      X      B
    Line(1,   2,   0.02,  0.10,  0.02),
    Line(2,   3,   0.03,  0.15,  0.03),
]

# 变压器（如果没有可以留空）
transformers = []

# 发电机（短路和稳定计算用）
generators = [
    Generator(bus=1, name="G1", Xd_doubleprime=0.20, Xd_prime=0.30, H=6.0, D=1.0),
]

# ========== 潮流计算 ==========
print("=" * 50)
print("潮流计算")
print("=" * 50)
ybus = build_ybus(lines, transformers)
pf_result = run_newton_raphson(buses, ybus, verbose=True)
print(f"收敛：{pf_result.converged}，迭代：{pf_result.iterations} 次")
print(f"网络损耗：{pf_result.losses:.4f} p.u.")

# ========== 短路计算 ==========
print("\n" + "=" * 50)
print("三相短路计算（节点3）")
print("=" * 50)
fault_bus = 3
sc_result = calculate_three_phase_fault(
    buses, lines, transformers, generators,
    fault_bus=fault_bus, S_base=100.0, V_base=10.5
)
print(f"短路电流：{sc_result.fault_current_ka:.2f} kA")

# ========== 小干扰稳定 ==========
print("\n" + "=" * 50)
print("小干扰稳定分析")
print("=" * 50)
ss_result = analyze_single_machine_infinite_bus(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=6.0, D=2.0, delta_0=np.radians(30), Pm=0.8,
    verbose=True
)
print(f"系统{'稳定' if ss_result.stable else '不稳定'}")

# ========== 暂态稳定 ==========
print("\n" + "=" * 50)
print("暂态稳定仿真")
print("=" * 50)
ts_result = simulate_single_machine_infinite_bus_classic(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    H=6.0, D=2.0, Pm=0.8, delta_0=np.radians(30),
    fault_time=0.0, fault_clearing_time=0.15,
    t_end=5.0
)
print(f"系统{'稳定' if ts_result.stable else '不稳定'}")
print(f"最大功角：{ts_result.max_delta:.2f}°")

# ========== Heffron-Phillips K1-K6 分析 ==========
print("\n" + "=" * 50)
print("Heffron-Phillips K1-K6 分析")
print("=" * 50)
hp = compute_heffron_phillips_constants(
    E_prime=1.2, V_infinity=1.0, X_total=0.5,
    Xd=1.8, Xd_prime=0.3, Xq=1.7,
    Td0_prime=8.0, H=6.0, D=0.0,
    delta_0=np.radians(30), Ka=50, Te=0.3
)
print(f"K1={hp.K1:+.4f}, K5={hp.K5:+.4f}")
print(f"K5 < 0 → AVR 负阻尼风险!" if hp.K5 < 0 else "K5 > 0 → 正阻尼")

# ========== 电压稳定性 ==========
print("\n" + "=" * 50)
print("PV 曲线电压稳定性")
print("=" * 50)
pv = compute_pv_curve(
    copy.deepcopy(buses), lines, transformers, generators,
    [], target_bus=3, lambda_max=3.0, n_points=30,
)
print(f"临界负荷因子: {pv.critical_lambda:.2f}")
print(f"鼻点电压: {pv.V_curve[pv.nose_point_index, 2]:.4f} pu")

print("\n" + "=" * 50)
print("计算完成！")
print("=" * 50)
```

---

## 附录E：Kundur 教材 Notebook 索引

项目提供 8 个 Jupyter Notebook，完整覆盖 Kundur 教材第 3-15 章的例题：

| # | Notebook | 章节 | 内容 |
|---|----------|------|------|
| 1 | `ch03_synchronous_machine.ipynb` | Ch.3 | 标幺值、Park 变换、相量图 |
| 2 | `ch04_machine_models.ipynb` | Ch.4 | 稳态/暂态/次暂态电机模型 |
| 3 | `ch08_excitation_systems.ipynb` | Ch.8 | IEEET1 励磁、参数扫描 |
| 4 | `ch12a_smib_small_signal.ipynb` | Ch.12 | Heffron-Phillips K1-K6 |
| 5 | `ch12b_two_area_small_signal.ipynb` | Ch.12 | 两区域特征值、参与因子 |
| 6 | `ch13_transient_stability.ipynb` | Ch.13 | 等面积定则、CCT、多机仿真 |
| 7 | `ch14_voltage_stability.ipynb` | Ch.14 | PV/QV 曲线、负荷模型 |
| 8 | `ch15_subsynchronous_resonance.ipynb` | Ch.15 | SSR 谐振频率、频率扫描 |

启动方式：
```
jupyter notebook examples/kundur/
```

> **下一步学习建议：**
> 1. 复制附录 D 的模板，尝试修改节点数、线路参数
> 2. 对照课本例题，验证程序结果
> 3. 打开 `examples/kundur/` 中的 Kundur 教材 Notebook，逐章学习
> 4. 探索 `powerflow/`、`shortcircuit/`、`stability/` 目录中的更多函数

---
*教程版本：v2.0 | 日期：2026-06-15 | 基于 psa4teaching v1.0*
