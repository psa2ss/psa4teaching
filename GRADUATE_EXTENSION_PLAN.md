# psa4teaching 研究生层次扩展规划

> 基于王锡凡《现代电力系统分析》教材  
> 版本：v1.1 | 时间：2026-05-16  
> 面向：电气工程/电力系统及其自动化 研究生教学

---

## 目录

1. [现有模块概览与差距分析](#1-现有模块概览与差距分析)
2. [高级潮流计算](#2-高级潮流计算)
3. [电力系统状态估计](#3-电力系统状态估计)
4. [电压稳定性分析](#4-电压稳定性分析)
5. [高级稳定分析](#5-高级稳定分析)
6. [电力系统规划与可靠性](#6-电力系统规划与可靠性)
7. [电力市场基础](#7-电力市场基础)
8. [柔性输电（FACTS）与新能源接入](#8-柔性输电facts与新能源接入)
9. [高压直流输电（HVDC）](#9-高压直流输电hvdc)
10. [扩展路线图与优先级总览](#10-扩展路线图与优先级总览)
11. [总结](#11-总结)

---

## 1. 现有模块概览与差距分析

### 1.1 现有模块（本科层次）

| 模块 | 文件 | 覆盖内容 | 参考教材 |
|------|------|----------|----------|
| **models/** | bus.py, line.py, transformer.py, generator.py, load.py | Bus, Line, Transformer, Generator, Load 基础模型 | 陈珩/李光琦 |
| **network/** | ybus.py, zbus.py | Ybus, Zbus 构造 | 陈珩 第二章 |
| **powerflow/** | newton_raphson.py, fast_decoupled.py, dc_powerflow.py | NR潮流, PQ分解, 直流潮流 | 陈珩 第三~四章 |
| **shortcircuit/** | symmetric.py, asymmetric.py, sequence_network.py, gb15544.py | 三相/不对称短路, 序网, GB15544 | 李光琦 第二章 |
| **stability/** | transient.py, small_signal.py | 暂态稳定(经典+详细), 小干扰稳定(特征值) | 李光琦 第三~四章 |

### 1.2 现有模型能力摘要

**Bus类** (models/bus.py):
- 节点类型：PQ / PV / SLACK
- 核心属性：V, delta, P, Q, P_specified, Q_specified, V_specified, Q_min, Q_max
- 方法：get_active_power_mismatch(), get_reactive_power_mismatch(), check_q_limit()

**Generator类** (models/generator.py):
- 模型类型：CLASSIC / DETAIL
- 核心属性：Xd, Xd_prime, Xd_doubleprime, Xq, Td0_prime, H, D
- 方法：compute_transient_emf(), compute_subtransient_emf(), get_inertia_constant()

**NewtonRaphsonResult** (powerflow/newton_raphson.py):
- 字段：converged, iterations, V, delta, P, Q, P_mismatch, Q_mismatch, max_mismatch, losses, history

**SmallSignalResult** (stability/small_signal.py):
- 字段：stable, eigenvalues, damping_ratios, frequencies, participation_factors, state_matrix

### 1.3 研究生扩展方向（王锡凡《现代电力系统分析》）

| # | 扩展方向 | 对应王锡凡章节 | 优先级 |
|---|----------|----------------|--------|
| 1 | 高级潮流计算 | 第三章(最优潮流)、第四章(连续潮流) | 高 |
| 2 | 电力系统状态估计 | 第五章 | 高 |
| 3 | 电压稳定性分析 | 第四章、第六章 | 高 |
| 4 | 高级稳定分析 | 第七章(低频振荡)、第八章(EEAC) | 中 |
| 5 | 电力系统规划与可靠性 | 第九章、第十章 | 中 |
| 6 | 电力市场基础 | 第十一章、第十二章 | 中 |
| 7 | FACTS与新能源接入 | 第六章、第十三章 | 高 |
| 8 | 高压直流输电（HVDC） | 第六章 6.3节、第八章 8.4节 | 高 |

---

## 2. 高级潮流计算

### 2.1 连续潮流法（Continuation Power Flow, CPF）

**教学目标**
- 理解电压稳定的概念与PV曲线
- 掌握连续潮流法的预测-校正机制
- 能够计算电压稳定临界点和最大传输功率

**参考教材**：王锡凡《现代电力系统分析》第四章 4.2节

**数学模型概述**

连续潮流通过在功率方程中引入连续参数 λ，构造扩展方程：

$$
\begin{aligned}
\mathbf{f}(\mathbf{x}, \lambda) &= \mathbf{P}_{spec} + \lambda \cdot \mathbf{P}_{step} - \mathbf{P}_{calc}(\mathbf{V}, \boldsymbol{\delta}) = 0 \\
\mathbf{g}(\mathbf{x}, \lambda) &= \mathbf{Q}_{spec} + \lambda \cdot \mathbf{Q}_{step} - \mathbf{Q}_{calc}(\mathbf{V}, \boldsymbol{\delta}) = 0 \\
\tau(\mathbf{x}, \lambda) &= (\mathbf{x} - \mathbf{x}_0)^T(\mathbf{x} - \mathbf{x}_0) + (\lambda - \lambda_0)^2 - \sigma^2 = 0
\end{aligned}
$$

其中：
- $\mathbf{x} = [\boldsymbol{\delta}^T, \mathbf{V}^T]^T$ 状态变量
- $\lambda$ 负荷增长参数
- $\sigma$ 步长

**关键算法**
1. **预测步（Predictor）**：沿切线方向预测下一解点
   $$\begin{bmatrix} \mathbf{J} & -\mathbf{p} \\ 2(\mathbf{x}-\mathbf{x}_0)^T & 2(\lambda-\lambda_0) \end{bmatrix} \begin{bmatrix} \Delta \mathbf{x} \\ \Delta \lambda \end{bmatrix} = \sigma \cdot \mathbf{e}_k$$
2. **校正步（Corrector）**：使用带弧长约束的牛顿法校正
3. **步长控制**：自适应调整 $\sigma$（收敛快则增大，困难则减小）
4. **参数切换**：鼻点附近自动切换参数化方向

**输入输出接口设计**

```python
# psa4teaching/powerflow/continuation.py

@dataclass
class CPResult:
    """连续潮流计算结果"""
    lambda_values: NDArray[np.float64]      # 负荷增长参数序列
    V_curve: NDArray[np.float64]            # 各节点电压幅值序列 [n_bus × n_points]
    P_curve: NDArray[np.float64]            # 各节点有功序列 [n_bus × n_points]
    Q_curve: NDArray[np.float64]            # 各节点无功序列 [n_bus × n_points]
    critical_lambda: float                  # 临界点 λ_critical
    critical_bus: int                       # 电压崩溃关键节点
    nose_point_index: int                   # 鼻点索引
    converged: bool
    message: str

def run_continuation_power_flow(
    buses: List[Bus],
    ybus_result: YBusResult,
    load_increase_pattern: Optional[Dict[int, Tuple[float, float]]] = None,
    # {bus_id: (P_ratio, Q_ratio)}，默认按当前负荷比例
    gen_increase_pattern: Optional[Dict[int, float]] = None,
    # {bus_id: P_ratio}，默认按当前出力比例
    start_lambda: float = 0.0,
    max_lambda: float = 5.0,
    step_size: float = 0.05,
    adaptive_step: bool = True,
    min_step: float = 0.01,
    max_step: float = 0.5,
    tolerance: float = 1e-6,
    max_iterations: int = 30,
    verbose: bool = False
) -> CPResult:
    """
    执行连续潮流计算，绘制PV曲线
    
    Args:
        buses: 节点列表
        ybus_result: Ybus结果
        load_increase_pattern: 负荷增长模式
        gen_increase_pattern: 发电机增长模式
        start_lambda: 起始λ
        max_lambda: 最大λ（停止条件）
        step_size: 初始步长
        adaptive_step: 是否自适应步长
        tolerance: 收敛精度
        max_iterations: 每步最大迭代次数
        verbose: 是否打印过程
        
    Returns:
        CPResult: 连续潮流结果，包含PV曲线和临界点
    """
    pass

def plot_pv_curve(result: CPResult, bus_ids: Optional[List[int]] = None) -> None:
    """绘制PV曲线，标注临界点"""
    pass
```

**与现有模块的关联**
- 复用 `network/ybus.py` 的 `YBusResult`（导纳矩阵）
- 复用 `models/bus.py` 的 `Bus` 类
- 参考 `powerflow/newton_raphson.py` 的雅可比矩阵构建方法
- 需要扩展雅可比矩阵，增加连续参数λ列

**优先级**：高  
**预计难度**：难

---

### 2.2 最优潮流（Optimal Power Flow, OPF）

**教学目标**
- 理解最优潮流的优化问题建模
- 掌握内点法求解OPF的基本原理
- 能够求解经济调度和无功优化问题

**参考教材**：王锡凡《现代电力系统分析》第三章

**数学模型概述**

最优潮流问题的一般形式：

$$
\begin{aligned}
\min_{\mathbf{x}, \mathbf{u}} \quad & f(\mathbf{x}, \mathbf{u}) \\
\text{s.t.} \quad & \mathbf{g}(\mathbf{x}, \mathbf{u}) = 0 \quad \text{(潮流方程)} \\
& \mathbf{h}(\mathbf{x}, \mathbf{u}) \leq 0 \quad \text{(不等式约束)}
\end{aligned}
$$

其中：
- 控制变量 $\mathbf{u} = [P_{G,i}, V_{G,i}, T_k, Q_{C,i}]^T$（发电机有功、电压、变压器变比、无功补偿）
- 状态变量 $\mathbf{x} = [\boldsymbol{\delta}, \mathbf{V}, Q_{G,i}]^T$
- 目标函数 $f$：发电成本 $f = \sum_i (a_i P_{G,i}^2 + b_i P_{G,i} + c_i)$

不等式约束：
- 发电机：$P_{G,i}^{min} \leq P_{G,i} \leq P_{G,i}^{max}$, $Q_{G,i}^{min} \leq Q_{G,i} \leq Q_{G,i}^{max}$
- 节点电压：$V_i^{min} \leq V_i \leq V_i^{max}$
- 线路功率：$|P_{ij}| \leq P_{ij}^{max}$

**关键算法**

1. **原对偶内点法（Primal-Dual Interior Point Method）**：
   - 引入松弛变量 $\mathbf{s} \geq 0$ 和对偶变量 $\mathbf{y}, \mathbf{z}$
   - 构造障碍函数（对数障碍项 $-\mu \sum \ln s_i$）
   - 迭代求解修正方程（降阶形式）：
   $$
   \begin{bmatrix}
   \mathbf{H} & \mathbf{J}^T & -\mathbf{I} \\
   \mathbf{J} & \mathbf{0} & \mathbf{0} \\
   \mathbf{S} & \mathbf{0} & \mathbf{Z}
   \end{bmatrix}
   \begin{bmatrix}
   \Delta \mathbf{x} \\
   \Delta \mathbf{y} \\
   \Delta \mathbf{z}
   \end{bmatrix}
   = -
   \begin{bmatrix}
   \nabla f + \mathbf{J}^T \mathbf{y} - \mathbf{z} \\
   \mathbf{g} \\
   \mathbf{S}\mathbf{Z}\mathbf{e} + \mu \mathbf{e}
   \end{bmatrix}
   $$
   - 障碍参数更新：$\mu \to 0$，收敛至KKT点

**输入输出接口设计**

```python
# psa4teaching/powerflow/optimal.py

@dataclass
class OPFGenerator:
    """OPF发电机参数（含成本曲线）"""
    bus: int
    a: float           # 成本系数 a（二次）
    b: float           # 成本系数 b（一次）
    c: float           # 成本系数 c（常数）
    P_min: float       # 最小有功
    P_max: float       # 最大有功
    Q_min: float       # 最小无功
    Q_max: float       # 最大无功
    V_specified: float # 电压设定值

@dataclass
class OPFResult:
    """最优潮流结果"""
    converged: bool
    objective_value: float                    # 最优目标函数值（总成本）
    P_gen: NDArray[np.float64]                # 发电机最优有功出力
    Q_gen: NDArray[np.float64]                # 发电机最优无功出力
    V: NDArray[np.float64]                    # 节点电压幅值
    delta: NDArray[np.float64]                # 节点电压相角
    lambda_P: NDArray[np.float64]             # 有功影子价格（LMP）
    lambda_Q: NDArray[np.float64]             # 无功影子价格
    mu_limits: Dict[str, NDArray]             # 约束的拉格朗日乘子
    iterations: int
    message: str

def run_optimal_power_flow(
    buses: List[Bus],
    generators: List[OPFGenerator],
    ybus_result: YBusResult,
    objective: str = "cost",           # "cost" | "loss" | "voltage_dev"
    method: str = "interior_point",    # "interior_point" | "linear_programming"
    V_limits: Tuple[float, float] = (0.95, 1.05),
    line_rating: Optional[Dict[Tuple[int, int], float]] = None,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    verbose: bool = False
) -> OPFResult:
    """
    执行最优潮流计算
    
    Args:
        buses: 节点列表
        generators: OPF发电机参数列表
        ybus_result: Ybus结果
        objective: 目标函数类型
        method: 求解方法
        V_limits: 电压上下限
        line_rating: 线路容量限制 {(i,j): limit}
        tolerance: 收敛精度
        max_iterations: 最大迭代次数
        verbose: 是否打印过程
        
    Returns:
        OPFResult: 最优潮流结果
    """
    pass
```

**与现有模块的关联**
- 复用 `powerflow/newton_raphson.py` 的雅可比矩阵计算
- 复用 `network/ybus.py` 的网络模型
- 新增 `models/opf_generator.py`（扩展发电机模型，增加成本参数）

**优先级**：高  
**预计难度**：难

---

### 2.3 三相潮流计算

**教学目标**
- 理解三相不平衡系统的建模方法
- 掌握三相牛顿-拉夫逊潮流算法
- 能够处理单相负荷、非全相运行等场景

**参考教材**：王锡凡《现代电力系统分析》第三章 3.5节

**数学模型概述**

三相系统节点电压方程（分相表示）：

$$\mathbf{I}_{abc} = \mathbf{Y}_{abc} \cdot \mathbf{V}_{abc}$$

各相导纳矩阵为 3n × 3n 矩阵：

$$\mathbf{Y}_{abc} =
\begin{bmatrix}
\mathbf{Y}_{aa} & \mathbf{Y}_{ab} & \mathbf{Y}_{ac} \\
\mathbf{Y}_{ba} & \mathbf{Y}_{bb} & \mathbf{Y}_{bc} \\
\mathbf{Y}_{ca} & \mathbf{Y}_{cb} & \mathbf{Y}_{cc}
\end{bmatrix}$$

**关键算法**
1. 构建三相导纳矩阵（考虑相间互阻抗和互导纳）
2. 三相牛顿-拉夫逊法（每节点6个方程：Pa、Qa、Pb、Qb、Pc、Qc）
3. 相分量与序分量转换（用于与对称分量法对比）

**输入输出接口设计**

```python
# psa4teaching/powerflow/three_phase.py

@dataclass
class ThreePhaseBus:
    """三相节点模型"""
    number: int
    name: Optional[str] = None
    V_a: complex = 1.0 + 0j
    V_b: complex = 1.0 + 0j
    V_c: complex = 1.0 + 0j
    P_a_spec: float = 0.0; Q_a_spec: float = 0.0
    P_b_spec: float = 0.0; Q_b_spec: float = 0.0
    P_c_spec: float = 0.0; Q_c_spec: float = 0.0
    bus_type: BusType = BusType.PQ

@dataclass
class ThreePhasePowerFlowResult:
    """三相潮流结果"""
    converged: bool
    V_abc: NDArray[np.complex128]      # [n_bus, 3] 三相电压
    P_abc: NDArray[np.float64]         # [n_bus, 3] 三相有功
    Q_abc: NDArray[np.float64]         # [n_bus, 3] 三相无功
    voltage_unbalance: NDArray[np.float64]  # 各节点电压不平衡度
    iterations: int

def run_three_phase_power_flow(
    buses_3p: List[ThreePhaseBus],
    ybus_3p: NDArray[np.complex128],
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> ThreePhasePowerFlowResult:
    """执行三相潮流计算"""
    pass
```

**与现有模块的关联**
- 扩展 `models/bus.py` 为三相版本
- 扩展 `models/line.py` 为三相版本（增加相间耦合）
- 扩展 `network/ybus.py` 构建3n×3n三相导纳矩阵
- 参考 `powerflow/newton_raphson.py` 的算法框架

**优先级**：中  
**预计难度**：难

---

### 2.4 交直流混联系统潮流

**教学目标**
- 理解高压直流输电（HVDC）的基本原理
- 掌握交直流系统潮流计算的统一迭代法
- 能够分析直流系统对交流系统的影响

**参考教材**：王锡凡《现代电力系统分析》第六章 6.3节

**数学模型概述**

换流站模型（简化）：
$$
\begin{aligned}
P_{dc,i} &= V_{dc,i} \cdot I_{dc,i} \\
P_{ac,i} &= P_{dc,i} + P_{loss,i} \\
Q_{ac,i} &= P_{ac,i} \cdot \tan \phi_i
\end{aligned}
$$

直流线路方程：
$$V_{dc,j} = V_{dc,i} - R_{dc} \cdot I_{dc}$$

控制模式：
- 整流侧：定电流控制（CC）或定功率控制（CP）
- 逆变侧：定电压控制（CV）或定熄弧角控制（CEA）

**输入输出接口设计**

```python
# psa4teaching/powerflow/acdc_powerflow.py

@dataclass
class HVDCConverter:
    """高压直流换流站模型"""
    ac_bus: int
    converter_type: str      # "rectifier" | "inverter"
    control_mode: str        # "CC" | "CP" | "CV" | "CEA"
    V_dc_nominal: float = 1.0
    I_dc_max: float = 2.0
    converter_loss: float = 0.01
    tap_min: float = 0.7
    tap_max: float = 1.3

@dataclass
class ACDCPowerFlowResult:
    """交直流潮流结果"""
    ac_result: NewtonRaphsonResult
    V_dc: NDArray[np.float64]
    I_dc: NDArray[np.float64]
    P_dc: NDArray[np.float64]
    Q_converter: NDArray[np.float64]
    converged: bool
    iterations: int

def run_acdc_power_flow(
    ac_buses: List[Bus],
    ac_ybus: YBusResult,
    converters: List[HVDCConverter],
    dc_lines: List[Tuple[int, int, float]],  # (from, to, R_dc)
    method: str = "unified",
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    verbose: bool = False
) -> ACDCPowerFlowResult:
    """执行交直流混联系统潮流计算"""
    pass
```

**与现有模块的关联**
- 复用 `powerflow/newton_raphson.py` 的交流潮流计算
- 在雅可比矩阵中增加直流变量相关列
- 新增 `models/hvdc_converter.py` 换流站模型

**优先级**：中  
**预计难度**：难

---

## 3. 电力系统状态估计

### 3.1 加权最小二乘状态估计（WLS）

**教学目标**
- 理解状态估计的基本概念和数学模型
- 掌握加权最小二乘估计的迭代求解算法
- 能够评估状态估计的精度和量测冗余度

**参考教材**：王锡凡《现代电力系统分析》第五章 5.2节

**数学模型概述**

状态估计问题：
$$\mathbf{z} = \mathbf{h}(\mathbf{x}) + \mathbf{e}$$

其中：
- $\mathbf{z}$：量测向量（节点注入功率、支路功率、节点电压幅值等）
- $\mathbf{x}$：状态向量 $\mathbf{x} = [\boldsymbol{\delta}^T, \mathbf{V}^T]^T$（N-1个相角 + N个电压幅值）
- $\mathbf{h}(\cdot)$：量测函数（由潮流方程给出）
- $\mathbf{e}$：量测误差向量，假设 $E[\mathbf{e}] = 0$, $E[\mathbf{e}\mathbf{e}^T] = \mathbf{R}$

加权最小二乘目标函数：
$$\min J(\mathbf{x}) = [\mathbf{z} - \mathbf{h}(\mathbf{x})]^T \mathbf{W} [\mathbf{z} - \mathbf{h}(\mathbf{x})]$$

其中 $\mathbf{W} = \mathbf{R}^{-1}$ 为权重矩阵。

**关键算法**

1. **迭代高斯-牛顿法**：
   $$\Delta \mathbf{x}^{(k)} = (\mathbf{H}^T \mathbf{W} \mathbf{H})^{-1} \mathbf{H}^T \mathbf{W} [\mathbf{z} - \mathbf{h}(\mathbf{x}^{(k)})]$$
   其中 $\mathbf{H} = \partial \mathbf{h} / \partial \mathbf{x}$ 为量测雅可比矩阵。

2. **标准化残差**：
   $$r_i = \frac{z_i - h_i(\mathbf{x})}{\sqrt{R_{ii}}}$$
   用于不良数据检测。

**输入输出接口设计**

```python
# psa4teaching/state_estimation/wls.py

@dataclass
class Measurement:
    """量测数据"""
    meas_type: str          # "Vm" | "Va" | "Pi" | "Qi" | "Pij" | "Qij"
    bus_i: int
    bus_j: Optional[int] = None  # 支路量测时的另一端
    value: float
    variance: float       # 方差，用于计算权重
    is_pseudo: bool = False  # 是否为伪量测

@dataclass
class WLSSResult:
    """加权最小二乘状态估计结果"""
    converged: bool
    x_est: NDArray[np.float64]       # 估计状态 [δ, V]
    V_est: NDArray[np.float64]        # 估计电压幅值
    delta_est: NDArray[np.float64]    # 估计电压相角
    residuals: NDArray[np.float64]    # 残差向量
    normalized_residuals: NDArray[np.float64]  # 标准化残差
    J_final: float                    # 最终目标函数值
    iterations: int
    observability: bool               # 系统是否可观测
    redundancy: float                 # 量测冗余度 = n_meas / n_state

def run_wls_state_estimation(
    measurements: List[Measurement],
    buses: List[Bus],
    ybus_result: YBusResult,
    include_pseudo: bool = True,
    tolerance: float = 1e-4,
    max_iterations: int = 50,
    verbose: bool = False
) -> WLSSResult:
    """
    执行加权最小二乘状态估计
    
    Args:
        measurements: 量测列表
        buses: 节点列表（提供拓扑信息）
        ybus_result: Ybus结果
        include_pseudo: 是否添加伪量测（对无实测的节点）
        tolerance: 收敛精度
        max_iterations: 最大迭代次数
        verbose: 是否打印过程
        
    Returns:
        WLSSResult: 状态估计结果
    """
    pass
```

**与现有模块的关联**
- 复用 `powerflow/newton_raphson.py` 的雅可比矩阵构建方法
- 复用 `network/ybus.py` 的网络模型
- 量测函数 $\mathbf{h}(\mathbf{x})$ 基于潮流功率方程

**优先级**：高  
**预计难度**：中

---

### 3.2 不良数据检测与辨识

**教学目标**
- 理解不良数据的来源和影响
- 掌握基于残差法的不良数据检测
- 能够辨识并剔除不良数据

**参考教材**：王锡凡《现代电力系统分析》第五章 5.3节

**数学模型概述**

**最大标准化残差检验（Largest Normalized Residual, LNR）**：
$$r_{max} = \max_i |r_i^N|$$
若 $r_{max} > \tau$（阈值，通常取3.0），则对应的量测为不良数据。

**逐次剔除算法**：
1. 执行WLS状态估计
2. 计算标准化残差 $r_i^N$
3. 若 $\max |r_i^N| > \tau$，剔除该量测
4. 返回步骤1，直至无不良数据

**输入输出接口设计**

```python
# psa4teaching/state_estimation/bad_data.py

@dataclass
class BadDataDetectionResult:
    """不良数据检测结果"""
    converged: bool
    x_est: NDArray[np.float64]
    bad_measurements: List[int]         # 不良数据索引列表
    normalized_residuals: NDArray[np.float64]
    n_iterations: int                   # 总迭代次数（含剔除过程）
    detection_threshold: float = 3.0

def detect_bad_data(
    measurements: List[Measurement],
    buses: List[Bus],
    ybus_result: YBusResult,
    threshold: float = 3.0,
    max_bad_data: int = 10,
    verbose: bool = False
) -> BadDataDetectionResult:
    """
    检测并辨识不良数据（逐次剔除算法）
    
    Args:
        measurements: 量测列表
        buses: 节点列表
        ybus_result: Ybus结果
        threshold: 标准化残差阈值（通常3.0~5.0）
        max_bad_data: 最大可剔除不良数据数量
        verbose: 是否打印过程
        
    Returns:
        BadDataDetectionResult: 检测结果
    """
    pass

def compute_chi_square_test(
    residuals: NDArray[np.float64],
    R: NDArray[np.float64],
    n_state: int,
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """
    $\chi^2$ 检验：检测是否存在不良数据
    
    Returns:
        (is_bad_data, chi_square_value)
    """
    pass
```

**与现有模块的关联**
- 依赖 `state_estimation/wls.py` 的WLS估计结果
- 复用量测雅可比矩阵和残差计算方法

**优先级**：高  
**预计难度**：中

---

### 3.3 量测冗余度分析

**教学目标**
- 理解量测冗余度的概念
- 掌握可观性分析方法
- 能够评估和优化量测配置方案

**参考教材**：王锡凡《现代电力系统分析》第五章 5.4节

**数学模型概述**

**可观性**：系统可观测当且仅当量测雅可比矩阵 $\mathbf{H}$ 行满秩。

**量测冗余度**：
$$\rho = \frac{m}{n}$$
其中 $m$ 为量测数，$n$ 为状态变量数。

**目标冗余度**：通常要求 $\rho \geq 1.5 \sim 2.0$。

**输入输出接口设计**

```python
# psa4teaching/state_estimation/redundancy.py

@dataclass
class RedundancyAnalysisResult:
    """冗余度分析结果"""
    observable: bool                  # 系统是否可观测
    redundancy_ratio: float           # 冗余度 m/n
    n_measurements: int
    n_states: int
    condition_number: float          # H^T W H 的条件数
    weak_buses: List[int]             # 量测薄弱节点
    measurement_coverage: Dict[str, float]  # 各类型量测覆盖率

def analyze_redundancy(
    measurements: List[Measurement],
    buses: List[Bus],
    ybus_result: YBusResult,
    verbose: bool = False
) -> RedundancyAnalysisResult:
    """分析量测冗余度和可观性"""
    pass

def suggest_pseudo_measurements(
    measurements: List[Measurement],
    buses: List[Bus],
    ybus_result: YBusResult,
    target_redundancy: float = 2.0
) -> List[Measurement]:
    """建议在哪些节点添加伪量测以提高冗余度"""
    pass
```

**与现有模块的关联**
- 依赖 `state_estimation/wls.py` 的量测雅可比矩阵
- 利用 `network/ybus.py` 的网络拓扑信息

**优先级**：中  
**预计难度**：易

---

## 4. 电压稳定性分析

### 4.1 连续潮流与PV/QV曲线

（已在 2.1 节详细介绍，此处不再重复）

### 4.2 电压稳定灵敏度指标

**教学目标**
- 理解电压稳定性的灵敏度分析方法
- 掌握L指标、最小奇异值等稳定指标
- 能够在线评估系统的电压稳定水平

**参考教材**：王锡凡《现代电力系统分析》第六章 6.2节

**数学模型概述**

**L指标**（基于潮流雅可比矩阵）：
$$L_j = \left| 1 - \sum_{i \in \text{gen}} \frac{V_i}{V_j} \cdot \frac{Z_{ji}}{Z_{ii}} \right|$$
稳定判据：$L_j < 1$，越接近1越不稳定。

**雅可比矩阵奇异值**：
$$\sigma_{min}(\mathbf{J}) = \lambda_{min}(\mathbf{J}^T\mathbf{J})$$
$\sigma_{min} \to 0$ 表示系统接近电压崩溃点。

**输入输接口设计**

```python
# psa4teaching/voltage_stability/indices.py

@dataclass
class VoltageStabilityIndices:
    """电压稳定指标"""
    L_indices: NDArray[np.float64]      # 各节点L指标
    min_L: float                        # 最小L指标（最危险节点）
    critical_bus_L: int                 # L指标最危险的节点
    min_singular_value: float            # 雅可比矩阵最小奇异值
    condition_number: float             # 条件数
    V_sensitivity: NDArray[np.float64]  # dV/dQ 灵敏度

def compute_voltage_stability_indices(
    buses: List[Bus],
    ybus_result: YBusResult,
    pf_result: NewtonRaphsonResult,
    method: str = "L_index",  # "L_index" | "singular_value" | "sensitivity"
    verbose: bool = False
) -> VoltageStabilityIndices:
    """计算电压稳定指标"""
    pass
```

**与现有模块的关联**
- 复用 `powerflow/newton_raphson.py` 的雅可比矩阵
- 依赖 `network/ybus.py` 的导纳矩阵
- 利用 `powerflow/continuation.py` 的PV曲线结果

**优先级**：高  
**预计难度**：中

---

### 4.3 模态分析法（Modal Analysis）

**教学目标**
- 理解模态分析的物理意义
- 掌握通过特征值分解识别薄弱区域的方法
- 能够使用模态分析指导无功补偿配置

**参考教材**：王锡凡《现代电力系统分析》第六章 6.3节

**数学模型概述**

约简雅可比矩阵（仅保留Q-V方程）：
$$\mathbf{J}_{QR} = \frac{\partial \mathbf{Q}}{\partial \mathbf{V}}$$

特征值分解：
$$\mathbf{J}_{QR} = \mathbf{U} \mathbf{\Lambda} \mathbf{V}^T$$

模态电压和模态无功功率的关系：
$$\Delta \mathbf{V} = \mathbf{v}_k \cdot p_k, \quad \Delta \mathbf{Q} = \mathbf{J}_{QR} \mathbf{v}_k \cdot p_k$$

参与因子（每个节点在每个模态中的参与程度）：
$$P_{ki} = |v_{ki} \cdot u_{ki}|$$

当某个模态的 $\lambda_k$ 接近0时，该模态对应节点区域最薄弱。

**输入输出接口设计**

```python
# psa4teaching/voltage_stability/modal.py

@dataclass
class ModalAnalysisResult:
    """模态分析结果"""
    eigenvalues: NDArray[np.float64]       # 特征值
    right_eigenvectors: NDArray[np.float64] # 右特征向量
    left_eigenvectors: NDArray[np.float64]  # 左特征向量
    participation_factors: NDArray[np.float64] # 参与因子矩阵 [n_mode × n_bus]
    critical_mode_index: int               # 临界模态索引
    weak_buses: List[int]                   # 薄弱节点（按参与因子排序）

def run_modal_analysis(
    buses: List[Bus],
    ybus_result: YBusResult,
    powerflow_result: NewtonRaphsonResult,
    n_modes: Optional[int] = None,
    verbose: bool = False
) -> ModalAnalysisResult:
    """执行模态分析"""
    pass
```

**与现有模块的关联**
- 复用 `powerflow/newton_raphson.py` 的雅可比矩阵
- 依赖 `voltage_stability/indices.py` 的基础指标

**优先级**：高  
**预计难度**：中

---

### 4.4 裕度计算

**教学目标**
- 理解电压稳定裕度的概念
- 掌握有功裕度和无功裕度的计算方法
- 能够评估系统的电压安全水平

**参考教材**：王锡凡《现代电力系统分析》第六章 6.4节

**数学模型概述**

有功裕度（Active Power Margin）：
$$M_P = \lambda_{max} - \lambda_0$$

无功裕度（Reactive Power Margin）：
$$M_Q = \sum_{i \in \text{gen}} Q_{G,i,max} - \sum_{i \in \text{gen}} Q_{G,i,0}$$

**输入输出接口设计**

```python
# psa4teaching/voltage_stability/margin.py

@dataclass
class VoltageStabilityMargin:
    """电压稳定裕度"""
    active_power_margin: float       # 有功裕度 (MW)
    reactive_power_margin: float     # 无功裕度 (MVar)
    lambda_critical: float           # 临界负荷参数
    critical_bus: int                # 临界节点
    safety_level: str                # "safe" | "warning" | "critical"

def compute_voltage_stability_margin(
    buses: List[Bus],
    ybus_result: YBusResult,
    load_increase: Dict[int, float],
    pf_result: NewtonRaphsonResult = None,
    method: str = "cpf",
    verbose: bool = False
) -> VoltageStabilityMargin:
    """计算电压稳定裕度"""
    pass
```

**与现有模块的关联**
- 调用 `powerflow/continuation.py` 的连续潮流结果
- 依赖 `voltage_stability/indices.py` 的灵敏度计算

**优先级**：高  
**预计难度**：中

---

## 5. 高级稳定分析

### 5.1 低频振荡分析与PSS设计原理

**教学目标**
- 理解低频振荡的物理机制（功角振荡）

- 掌握振荡模式的辨识方法（特征值、Prony分析）
- 理解PSS（电力系统稳定器）的补偿原理
- 能够进行PSS参数的初步设计

**参考教材**：王锡凡《现代电力系统分析》第七章 7.2节

**数学模型概述**

**低频振荡分类**：
- 局部模式（Local Mode）：0.7~2.0 Hz，涉及一台机组相对系统振荡
- 区间模式（Inter-area Mode）：0.1~0.7 Hz，涉及两组机组间振荡

**特征值分析**（扩展现有小干扰稳定分析）：
$$\mathbf{A}\mathbf{x} = \lambda\mathbf{x}$$

振荡模式由共轭复特征值对确定：
$$\lambda = \sigma \pm j\omega_d$$

- 阻尼比：$\zeta = -\sigma / \sqrt{\sigma^2 + \omega_d^2}$
- 振荡频率：$f = \omega_d / (2\pi)$
- 稳定判据：$\zeta > 0.03 \sim 0.05$（工程要求）

**输入输接口设计**

```python
# psa4teaching/stability/low_frequency_oscillation.py

@dataclass
class OscillationMode:
    """振荡模式信息"""
    eigenvalue: complex
    damping_ratio: float
    frequency: float                # Hz
    mode_type: str                 # "local" | "inter_area" | "non_oscillatory"
    associated_generators: List[int]
    participation_factors: NDArray

@dataclass
class LowFreqOscillationResult:
    """低频振荡分析结果"""
    modes: List[OscillationMode]
    poorly_damped_modes: List[int]
    state_matrix: NDArray
    participation_matrix: NDArray

def analyze_low_frequency_oscillation(
    state_matrix: NDArray,
    gen_bus_indices: List[int],
    state_names: Optional[List[str]] = None,
    damping_threshold: float = 0.05,
    freq_threshold: Tuple[float, float] = (0.1, 2.5),
    verbose: bool = False
) -> LowFreqOscillationResult:
    """分析低频振荡模式"""
    pass

def prony_analysis(
    signal: NDArray[np.float64],
    dt: float,
    n_modes: int = 4,
    verbose: bool = False
) -> List[OscillationMode]:
    """
    Prony分析法：从时域信号中辨识振荡模式
    y(t) = Σ A_i * exp(λ_i * t)
    """
    pass
```

**与现有模块的关联**
- 扩展 `stability/small_signal.py` 的 `analyze_multi_machine()` 函数
- 复用 `stability/transient.py` 的仿真结果（用于Prony分析输入）
- 新增 `stability/pss.py` PSS模型类

**优先级**：高  
**预计难度**：难

---

### 5.2 多机系统完整模型链

**教学目标**
- 掌握包含完整调速器-励磁-PSS的多机系统模型
- 理解各环节的传递函数和状态方程
- 能够构建完整的状态矩阵进行稳定分析

**参考教材**：王锡凡《现代电力系统分析》第七章

**数学模型概述**

**完整模型链组成**（每台发电机）：

1. **转子运动方程**（2阶）
2. **d轴电气方程**（1阶，Eq'）
3. **励磁系统**（IEEE ST1A简化，3阶）
4. **调速器**（简化模型，2阶）
5. **PSS**（2阶超前滞后）

单机状态变量数：2(δ,ω) + 1(Eq') + 3(AVR) + 2(Gov) + 2(PSS) = 10

**输入输接口设计**

```python
# psa4teaching/models/exciter.py

@dataclass
class ExciterModel:
    """励磁系统模型（IEEE ST1A简化）"""
    Ka: float = 200.0          # 放大倍数
    Ta: float = 0.02           # 放大环节时间常数
    Te: float = 0.5            # 励磁机时间常数
    Efd_min: float = 0.0
    Efd_max: float = 7.0
    Tr: float = 0.02
    n_states: int = 3

# psa4teaching/stability/multi_machine_detailed.py

def build_full_system_state_matrix(
    generators: List[Generator],
    exciters: Dict[int, ExciterModel],
    governors: Dict[int, Any],
    pss_models: Optional[Dict[int, Any]] = None,
    ybus_result: YBusResult = None,
    pf_result: NewtonRaphsonResult = None,
    verbose: bool = False
) -> NDArray[np.float64]:
    """构建包含完整模型链的多机系统状态矩阵"""
    pass
```

**与现有模块的关联**
- 扩展 `models/generator.py` 增加励磁/调速/PSS引用字段
- 新增 `models/exciter.py`、`models/governor.py`、`models/pss.py`
- 扩展 `stability/small_signal.py` 的多机分析功能

**优先级**：高  
**预计难度**：难

---

### 5.3 等面积法则（EEAC）扩展

**教学目标**
- 理解等面积法则的物理意义（能量守恒）
- 掌握多机系统的EEAC扩展方法
- 能够通过EEAC快速判断暂态稳定性

**参考教材**：王锡凡《现代电力系统分析》第八章 8.3节

**数学模型概述**

**单机无穷大系统等面积法则**：
临界条件：减速面积 = 加速面积

**多机系统EEAC（扩展等面积法则）**：
将n机系统等效为两机系统：
- 领先群（临界机组群）：功角超前最大的机组集合S
- 其余机组：系统其余部分A

等效惯量：
$$M_{eq} = \frac{M_S \cdot M_A}{M_S + M_A}$$

等效功角：
$$\delta_{eq} = \delta_{COA,S} - \delta_{COA,A}$$

**输入输接口设计**

```python
# psa4teaching/stability/eeac.py

@dataclass
class EEACResult:
    """EEAC分析结果"""
    stable: bool
    critical_clearing_time: float   # 临界清除时间 (s)
    equal_area_criterion_satisfied: bool
    accelerating_area: float
    decelerating_area: float
    critical_generator_group: List[int]
    equivalent_angle: NDArray[np.float64]  # 等效功角曲线

def run_eeac_analysis(
    generators: List[Generator],
    ybus_result: YBusResult,
    fault_bus: int,
    fault_start: float,
    fault_clear: float,
    verbose: bool = False
) -> EEACResult:
    """执行扩展等面积法则分析"""
    pass
```

**与现有模块的关联**
- 复用 `stability/transient.py` 的时域仿真结果
- 复用 `network/ybus.py` 的网络模型
- 扩展 `stability/small_signal.py` 的特征值分析

**优先级**：中  
**预计难度**：中

---

## 6. 电力系统规划与可靠性

### 6.1 发电系统可靠性

**教学目标**
- 理解发电系统可靠性的基本概念
- 掌握LOLP、LOLE、EENS的计算方法
- 能够进行发电容量充裕度评估

**参考教材**：王锡凡《现代电力系统分析》第九章 9.2节

**数学模型概述**

**LOLP**（Loss of Load Probability）：
$$LOLP = \sum_{i \in \text{failure states}} p_i$$

**LOLE**（Loss of Load Expectation）：
$$LOLE = LOLP \times 8760 \text{ (hours/year)}$$

**EENS**（Expected Energy Not Supplied）：
$$EENS = \sum_i p_i \cdot EENS_i$$

**输入输接口设计**

```python
# psa4teaching/reliability/generation.py

@dataclass
class GenerationUnit:
    """发电机组"""
    unit_id: int
    capacity: float       # 容量 (MW)
    forced_outage_rate: float  # 强迫停运率 FOR
    maintenance_rate: float = 0.0

@dataclass
class GenerationReliabilityResult:
    """发电系统可靠性结果"""
    LOLP: float             # 失负荷概率
    LOLE: float             # 失负荷期望 (hours/year)
    EENS: float             # 期望缺供电量 (MWh/year)
    capacity_margin: float  # 容量裕度

def compute_generation_reliability(
    units: List[GenerationUnit],
    peak_load: float,
    load_duration_curve: Optional[NDArray] = None,
    verbose: bool = False
) -> GenerationReliabilityResult:
    """计算发电系统可靠性指标"""
    pass
```

**优先级**：中  
**预计难度**：中

---

### 6.2 输电系统可靠性（N-1/N-2分析）

**教学目标**
- 理解输电系统可靠性的评估方法
- 掌握N-1准则的检验方法
- 能够进行输电系统充裕度评估

**参考教材**：王锡凡《现代电力系统分析》第九章 9.3节

**输入输接口设计**

```python
# psa4teaching/reliability/transmission.py

@dataclass
class TransmissionReliabilityResult:
    """输电系统可靠性结果"""
    n_minus_1_violations: List[Tuple[int, int]]  # 违反N-1的支路
    n_minus_2_violations: List[Tuple[int, int, int, int]]
    system_eens: float
    saifi: float    # 系统平均停电频率指数
    saidi: float    # 系统平均停电持续时间指数

def check_n_minus_one_criterion(
    buses: List[Bus],
    lines: List[Line],
    transformers: List[Transformer],
    ybus_result: YBusResult,
    verbose: bool = False
) -> TransmissionReliabilityResult:
    """检验N-1准则"""
    pass
```

**优先级**：中  
**预计难度**：中

---

## 7. 电力市场基础

### 7.1 经济调度与机组组合

**教学目标**
- 理解电力市场的基本概念
- 掌握经济调度的等微增率准则
- 了解机组组合的优化方法

**参考教材**：王锡凡《现代电力系统分析》第十一章 11.2节

**数学模型概述**

**经济调度**（Economic Dispatch）：
$$\min \sum_i C_i(P_{G,i}) \quad \text{s.t.} \quad \sum_i P_{G,i} = P_{load}$$

**机组组合**（Unit Commitment）：
$$\min \sum_i \sum_t [C_i(P_{G,i,t}) + C_{start,i} \cdot u_{i,t} + C_{shut,i} \cdot v_{i,t}]$$

**输入输接口设计**

```python
# psa4teaching/market/economic_dispatch.py

@dataclass
class GeneratorOffer:
    """发电机报价"""
    bus: int
    a: float; b: float; c: float  # 成本系数
    P_min: float; P_max: float
    status: bool = True  # 是否开机

@dataclass
class EconomicDispatchResult:
    """经济调度结果"""
    P_gen: NDArray[np.float64]
    lambda_system: float   # 系统边际电价
    total_cost: float
    converged: bool

def run_economic_dispatch(
    offers: List[GeneratorOffer],
    total_load: float,
    losses_model: Optional[str] = None,
    verbose: bool = False
) -> EconomicDispatchResult:
    """执行经济调度（等微增率准则）"""
    pass
```

**与现有模块的关联**
- 复用 `powerflow/optimal.py` 的OPF求解框架
- 扩展目标函数为多时段机组组合

**优先级**：中  
**预计难度**：中

---

### 7.2 节点边际电价（LMP）计算

**教学目标**
- 理解节点边际电价的构成
- 掌握LMP的计算方法
- 能够分析阻塞对电价的影响

**参考教材**：王锡凡《现代电力系统分析》第十二章 12.3节

**数学模型概述**

LMP由三部分组成：
$$LMP_i = \lambda + \mu_i^{line} + \mu_i^{congestion}$$

其中：
- $\lambda$：系统能量分量（统一电价）
- $\mu_i^{line}$：线路阻塞分量
- $\mu_i^{congestion}$：节点 congestin 分量

**输入输接口设计**

```python
# psa4teaching/market/lmp.py

@dataclass
class LMPResult:
    """节点边际电价结果"""
    LMP: NDArray[np.float64]   # 各节点LMP ($/MWh)
    lambda_system: float        # 系统边际电价
    congestion_rent: float     # 阻塞盈余
    loss_rent: float           # 网损盈余
    binding_constraints: List[int]  # 阻塞的线路索引

def compute_lmp(
    buses: List[Bus],
    generators: List[OPFGenerator],
    ybus_result: YBusResult,
    offers: List[GeneratorOffer],
    verbose: bool = False
) -> LMPResult:
    """计算节点边际电价（基于OPF拉格朗日乘子）"""
    pass
```

**与现有模块的关联**
- 复用 `powerflow/optimal.py` 的OPF结果
- 从OPF拉格朗日乘子提取LMP

**优先级**：中  
**预计难度**：中

---

## 8. 柔性输电（FACTS）与新能源接入

### 8.1 FACTS器件模型（SVC/STATCOM/TCSC）

**教学目标**
- 理解FACTS器件的工作原理
- 掌握SVC、STATCOM、TCSC的稳态模型
- 能够分析FACTS对电压/潮流的控制效果

**参考教材**：王锡凡《现代电力系统分析》第六章 6.4节

**数学模型概述**

**SVC**（Static Var Compensator）：
$$Q_{SVC} = -B_{SVC} \cdot V^2, \quad B_{min} \leq B_{SVC} \leq B_{max}$$

**STATCOM**（Static Synchronous Compensator）：
$$Q_{STATCOM} = V \cdot I_q, \quad |I| \leq I_{max}$$

**TCSC**（Thyristor Controlled Series Capacitor）：
$$X_{TCSC} = X_{line} \cdot k_{TCSC}, \quad k_{min} \leq k \leq k_{max}$$

**输入输接口设计**

```python
# psa4teaching/models/facts.py

@dataclass
class SVC:
    """静止无功补偿器"""
    bus: int
    B_min: float = -1.0   # 最小电纳（感性）
    B_max: float = 1.0    # 最大电纳（容性）
    V_setpoint: float = 1.0
    Q_output: float = 0.0   # 当前无功输出

@dataclass
class STATCOM:
    """静止同步补偿器"""
    bus: int
    V_dc: float = 1.0        # 直流电压
    I_max: float = 1.0        # 最大电流
    V_setpoint: float = 1.0
    mode: str = "voltage"       # "voltage" | "reactive"

@dataclass
class TCSC:
    """可控串联补偿器"""
    from_bus: int
    to_bus: int
    X_line: float
    k_min: float = 0.5
    k_max: float = 2.0
    k_current: float = 1.0

def incorporate_facts_into_ybus(
    ybus: NDArray[np.complex128],
    buses: List[Bus],
    svcs: List[SVC],
    statcoms: List[STATCOM],
    tcscs: List[TCSC],
    bus_indices: Dict[int, int]
) -> NDArray[np.complex128]:
    """将FACTS器件纳入Ybus矩阵"""
    pass
```

**与现有模块的关联**
- 扩展 `models/` 目录，新增 `facts.py`
- 修改 `network/ybus.py` 支持FACTS器件的导纳贡献
- 扩展 `powerflow/newton_raphson.py` 处理FACTS控制变量

**优先级**：高  
**预计难度**：中

---

### 8.2 双馈感应发电机（DFIG）模型

**教学目标**
- 理解风力发电的基本原理
- 掌握DFIG的稳态和动态模型
- 能够分析风电接入对系统稳定的影响

**参考教材**：王锡凡《现代电力系统分析》第十三章 13.2节

**数学模型概述**

**DFIG稳态模型**：
$$P_{e} = \frac{V^2}{2} \left( \frac{R_s}{R_s^2 + (X_s + X_m)^2} \right) + \text{terms with slip}$$

**控制方式**：
- 转子侧换流器：有功/无功解耦控制
- 网侧换流器：直流电压控制

**输入输接口设计**

```python
# psa4teaching/models/dfig.py

@dataclass
class DFIG:
    """双馈感应发电机"""
    bus: int
    name: Optional[str] = None
    P_rated: float = 1.5        # 额定有功 (MW)
    V_rated: float = 0.69        # 额定电压 (kV)
    R_s: float = 0.01           # 定子电阻 (p.u.)
    X_s: float = 0.1            # 定子漏抗 (p.u.)
    X_m: float = 3.0            # 励磁电抗 (p.u.)
    R_r: float = 0.01           # 转子电阻 (p.u.)
    X_r: float = 0.1            # 转子漏抗 (p.u.)
    H: float = 3.0              # 惯性时间常数
    slip: float = 0.0           # 滑差
    control_mode: str = "PQ"      # "PQ" | "voltage" | "MPPT"

def dfig_power_equation(
    dfig: DFIG,
    V: complex,
    slip: float
) -> Tuple[complex, complex]:
    """计算DFIG的功率输出"""
    pass
```

**与现有模块的关联**
- 扩展 `models/generator.py`，DFIG作为特殊发电机类型
- 扩展 `stability/transient.py` 增加DFIG动态模型
- 新增 `models/dfig.py`

**优先级**：高  
**预计难度**：难

---

### 8.3 光伏发电系统模型

**教学目标**
- 理解光伏发电的基本原理
- 掌握光伏逆变器的控制方式
- 能够建模光伏电站的功率特性

**参考教材**：王锡凡《现代电力系统分析》第十三章 13.3节

**输入输接口设计**

```python
# psa4teaching/models/photovoltaic.py

@dataclass
class PhotovoltaicSystem:
    """光伏发电系统"""
    bus: int
    P_max: float = 1.0          # 最大有功出力 (MW)
    V_dc: float = 0.6            # 直流侧电压 (kV)
    cos_phi: float = 1.0        # 功率因数
    control_mode: str = "PQ"      # "PQ" | "voltage"
    mppt_enabled: bool = True    # 是否启用最大功率点跟踪

def pv_power_output(
    pv: PhotovoltaicSystem,
    irradiance: float = 1.0,    # 辐照度 (p.u.)
    temperature: float = 25.0    # 温度 (°C)
) -> complex:
    """计算光伏系统功率输出"""
    pass
```

**优先级**：中  
**预计难度**：中

---

### 8.4 储能系统模型

**教学目标**
- 理解储能系统在电力系统中的作用
- 掌握储能的充放电模型
- 能够分析储能对系统稳定的支撑作用

**输入输接口设计**

```python
# psa4teaching/models/energy_storage.py

@dataclass
class EnergyStorageSystem:
    """储能系统"""
    bus: int
    E_capacity: float = 2.0     # 容量 (MWh)
    P_max_charge: float = 1.0    # 最大充电功率 (MW)
    P_max_discharge: float = 1.0  # 最大放电功率 (MW)
    efficiency: float = 0.9       # 往返效率
    SOC_min: float = 0.1          # 最小SOC
    SOC_max: float = 0.9          # 最大SOC
    SOC_current: float = 0.5       # 当前SOC

def storage_power_output(
    ess: EnergyStorageSystem,
    P_setpoint: float,            # 正值=放电，负值=充电
    dt: float
) -> complex:
    """计算储能系统功率输出，更新SOC"""
    pass
```

**优先级**：中  
**预计难度**：中


---

## 9. 高压直流输电（HVDC）

### 9.1 两端HVDC稳态模型

**教学目标**
- 理解HVDC输电的基本原理与优势
- 掌握换流站（整流侧/逆变侧）的稳态数学模型
- 掌握不同控制模式（定电流/定功率/定电压/定熄弧角）的实现
- 能够进行含HVDC的交直流混联系统潮流计算

**参考教材**：王锡凡《现代电力系统分析》第六章 6.3节

**数学模型概述**

#### 换流站基本方程

**直流电压**（整流侧）：
$$V_{dc,r} = \frac{3\sqrt{2}}{\pi} \cdot k_r \cdot V_{ac,r} \cdot \cos \alpha - \frac{3}{\pi} \cdot X_{cr} \cdot I_{dc}$$

**直流电压**（逆变侧）：
$$V_{dc,i} = \frac{3\sqrt{2}}{\pi} \cdot k_i \cdot V_{ac,i} \cdot \cos \gamma - \frac{3}{\pi} \cdot X_{ci} \cdot I_{dc}$$

**直流电流**：
$$I_{dc} = \frac{V_{dc,r} - V_{dc,i}}{R_{dc} + R_{cr} + R_{ci}}$$

其中：
- $k_r, k_i$：整流侧/逆变侧换流变压器变比
- $V_{ac,r}, V_{ac,i}$：交流侧线电压有效值
- $\alpha$：整流侧触发角（0°~90°）
- $\gamma$：逆变侧熄弧角（通常12°~18°）
- $X_{cr}, X_{ci}$：整流/逆变侧换相电抗
- $R_{dc}$：直流线路电阻

**功率关系**：
$$P_{dc} = V_{dc} \cdot I_{dc}, \quad P_{ac} = P_{dc} / (1 - \eta_{loss}), \quad Q_{ac} = P_{ac} \cdot \tan\phi$$

其中功率因数角 $\phi$ 由换流变压器漏抗和触发角/熄弧角决定。

#### 控制模式

| 控制模式 | 整流侧 | 逆变侧 | 应用场景 |
|----------|--------|--------|----------|
| CC-CV | 定电流控制 | 定电压控制 | 最常见，整流侧主控 |
| CP-CV | 定功率控制 | 定电压控制 | 功率传输指定场景 |
| CC-CEA | 定电流控制 | 定熄弧角控制 | 逆变侧安全约束 |

#### 交直流潮流求解方法

**方法一：统一迭代法（推荐）**
- 将直流变量（$I_{dc}, V_{dc,r}, V_{dc,i}$）纳入状态向量
- 交流潮流方程 + 直流方程 + 换流站方程统一求解
- 雅可比矩阵扩展（增加直流相关行列）

**方法二：交替迭代法（传统）**
- 先求解交流系统（HVDC视为PQ节点或PV节点）
- 再求解直流系统（更新换流站注入功率）
- 交替迭代直至收敛

**输入输出接口设计**

```python
# psa4teaching/models/hvdc.py

class ControlMode(Enum):
    """HVDC控制模式"""
    CC_CV = "cc_cv"          # 整流侧定电流，逆变侧定电压
    CP_CV = "cp_cv"          # 整流侧定功率，逆变侧定电压
    CC_CEA = "cc_cea"        # 整流侧定电流，逆变侧定熄弧角
    MANUAL = "manual"        # 手动指定所有参数


@dataclass
class ConverterStation:
    """换流站模型"""
    bus: int                      # 所连交流节点
    converter_type: str           # "rectifier" | "inverter"
    control_mode: ControlMode
    
    # 交流侧参数
    V_ac_nominal: float = 1.0    # 额定交流电压 (p.u.)
    k_transformer: float = 1.0   # 换流变压器变比
    X_commutating: float = 0.2   # 换相电抗 (p.u.)
    
    # 控制参数
    alpha: float = 15.0           # 触发角（整流侧，度）
    gamma: float = 15.0           # 熄弧角（逆变侧，度）
    I_dc_set: float = 1.0         # 定电流设定值 (p.u.)
    P_dc_set: float = 1.0         # 定功率设定值 (p.u.)
    V_dc_set: float = 1.0         # 定电压设定值 (p.u.)
    
    # 损耗与限幅
    loss_factor: float = 0.01     # 换流站损耗系数
    I_min: float = 0.1
    I_max: float = 2.0
    
    # 计算结果（潮流后更新）
    V_dc: float = 0.0             # 直流侧电压
    I_dc: float = 0.0             # 直流电流
    P_ac: float = 0.0             # 交流侧有功注入
    Q_ac: float = 0.0             # 交流侧无功消耗


@dataclass
class HVDCLine:
    """高压直流输电线路"""
    name: Optional[str] = None
    rectifier_bus: int             # 整流侧交流节点
    inverter_bus: int              # 逆变侧交流节点
    R_dc: float = 0.01             # 直流线路电阻 (p.u.)
    C_dc: float = 0.0              # 直流线路电容（暂态用，稳态忽略）
    I_max: float = 2.0             # 最大直流电流 (p.u.)
    P_max: float = 2.0             # 最大传输功率 (p.u.)
    V_dc_nominal: float = 1.0      # 额定直流电压 (p.u.)
    
    def get_dc_resistance(self) -> float:
        """返回直流线路电阻"""
        return self.R_dc
```

```python
# psa4teaching/powerflow/hvdc_powerflow.py

@dataclass
class HVDCPowerFlowResult:
    """含HVDC的潮流结果"""
    ac_result: NewtonRaphsonResult      # 交流系统潮流结果
    V_dc: Dict[str, float]              # 各换流站直流电压 {"r": Vr, "i": Vi}
    I_dc: float                         # 直流电流
    P_dc: float                         # 直流传输功率
    Q_rectifier: float                  # 整流侧无功消耗
    Q_inverter: float                   # 逆变侧无功消耗
    converter_angles: Dict[str, float]  # {"alpha": ..., "gamma": ...}
    converged: bool
    iterations: int
    message: str


def run_hvdc_power_flow(
    ac_buses: List[Bus],
    ac_lines: List[Line],
    ac_transformers: List[Transformer],
    hvdc_line: HVDCLine,
    rectifier: ConverterStation,
    inverter: ConverterStation,
    ybus_result: YBusResult,
    method: str = "unified",      # "unified" | "alternating"
    tolerance: float = 1e-6,
    max_iterations: int = 50,
    verbose: bool = False
) -> HVDCPowerFlowResult:
    """
    执行含HVDC的交直流混联系统潮流计算
    
    Args:
        ac_buses: 交流节点列表
        ac_lines: 交流线路列表
        ac_transformers: 交流变压器列表
        hvdc_line: HVDC线路参数
        rectifier: 整流侧换流站
        inverter: 逆变侧换流站
        ybus_result: 交流系统Ybus结果
        method: 求解方法（"unified"统一迭代 / "alternating"交替迭代）
        tolerance: 收敛精度
        max_iterations: 最大迭代次数
        verbose: 是否打印过程
        
    Returns:
        HVDCPowerFlowResult: 含HVDC潮流结果
    """
    pass


def solve_dc_system(
    hvdc_line: HVDCLine,
    rectifier: ConverterStation,
    inverter: ConverterStation,
    V_ac_rect: float,
    V_ac_inv: float,
    verbose: bool = False
) -> Tuple[float, float, float, float, float]:
    """
    求解直流子系统
    
    Args:
        V_ac_rect: 整流侧交流电压幅值
        V_ac_inv: 逆变侧交流电压幅值
    
    Returns:
        (V_dc_r, V_dc_i, I_dc, P_dc, Q_ac_r, Q_ac_i)
    """
    pass


def _build_hvdc_ybus_extension(
    ybus_ac: NDArray[np.complex128],
    hvdc_line: HVDCLine,
    rectifier: ConverterStation,
    inverter: ConverterStation,
    bus_indices: Dict[int, int]
) -> Tuple[NDArray, List[int]]:
    """构建扩展的Ybus矩阵（含HVDC等效注入修正）"""
    pass
```

**与现有模块的关联**
- 新建 `models/hvdc.py`：HVDCLine + ConverterStation 模型
- 新建 `powerflow/hvdc_powerflow.py`：交直流潮流求解器
- 扩展 `network/ybus.py`：将HVDC等效为交流节点的功率注入
- 扩展 `powerflow/newton_raphson.py`：处理HVDC节点特殊边界条件

**优先级**：高  
**预计难度**：难

---

### 9.2 多端直流输电（MTDC）

**教学目标**
- 理解多端直流系统（MTDC）的拓扑结构与运行特点
- 掌握MTDC的节点方程建模方法
- 了解主从控制、电压裕度控制、斜率控制等策略
- 能够进行含MTDC的交直流潮流计算

**参考教材**：王锡凡《现代电力系统分析》第六章 6.4节

**数学模型概述**

#### MTDC网络方程

采用节点法建立直流网络方程：

$$
\mathbf{I}_{dc} = \mathbf{G}_{dc} \cdot \mathbf{V}_{dc}
$$

其中 $\mathbf{G}_{dc}$ 为 n×n 直流导纳矩阵（由直流线路电导构成）。

对节点 $i$（换流站所在节点）：
$$P_{dc,i} = V_{dc,i} \cdot \sum_j G_{ij}(V_{dc,i} - V_{dc,j})$$

#### 控制策略对比

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **主从控制** | 一个站定电压，其余定功率 | 实现简单 | 主站故障时系统崩溃 |
| **电压裕度控制** | 多个站具备定电压能力，按裕度切换 | 有冗余 | 控制逻辑较复杂 |
| **直流电压斜率控制** | $V_i = V_{ref} - k \cdot P_i$ | 功率自动分配 | 参数整定困难 |

#### 斜率控制模型

$$V_{dc,i} = V_{ref,i} - k_{droop,i} \cdot P_{dc,i}$$

其中 $k_{droop,i}$ 为第 $i$ 个换流站的斜率系数（通常 3~10%/额定功率）。

**输入输出接口设计**

```python
# psa4teaching/models/mtdc.py

class MTDControlStrategy(Enum):
    MASTER_SLAVE = "master_slave"
    VOLTAGE_MARGIN = "voltage_margin"
    DROOP = "droop"


@dataclass
class MTDCSystem:
    """多端直流系统"""
    name: str
    dc_buses: List[int]                             # 直流节点列表
    dc_lines: List[Tuple[int, int, float]]           # (from, to, R_dc)
    converter_stations: Dict[int, ConverterStation]  # dc_bus -> station
    control_strategy: MTDControlStrategy = MTDControlStrategy.MASTER_SLAVE
    master_station: Optional[int] = None             # 主站编号（主从控制用）
    
    def build_dc_admittance_matrix(self) -> NDArray[np.float64]:
        """构建直流导纳矩阵 G_dc"""
    pass
    
    def get_dc_power_balance(self, V_dc: NDArray) -> NDArray:
        """计算各节点的直流功率平衡 P_dc_i"""
    pass


@dataclass
class DroopControlParams:
    """斜率控制参数"""
    V_ref: float = 1.0           # 参考电压
    k_droop: float = 5.0          # 斜率系数 (%/p.u.)
    P_max: float = 2.0
    P_min: float = -2.0


def run_mtdc_power_flow(
    ac_buses: List[Bus],
    ac_ybus: YBusResult,
    mtdc: MTDCSystem,
    method: str = "unified",
    verbose: bool = False
) -> HVDCPowerFlowResult:
    """执行含MTDC的交直流潮流"""
    pass
```

**与现有模块的关联**
- 基于 `models/hvdc.py` 的 ConverterStation 扩展
- 扩展 `powerflow/hvdc_powerflow.py` 支持多端求解
- 新建 `models/mtdc.py`

**优先级**：中  
**预计难度**：难

---

### 9.3 HVDC对短路和稳定的影响

**教学目标**
- 理解HVDC对短路电流的贡献与限制
- 掌握含HVDC系统的暂态稳定分析
- 理解HVDC的功率调制（Power Modulation）阻尼原理
- 能够进行含HVDC的暂态稳定时域仿真

**参考教材**：王锡凡《现代电力系统分析》第八章 8.4节

**数学模型概述**

#### HVDC对短路电流的影响

换流站对短路电流的贡献取决于控制模式：
- **定电流控制**：短路时 $I_{dc}$ 被限制在设定值附近，对短路电流贡献有限
- **定功率控制**：短路时控制系统会快速调整，影响次暂态短路电流

短路计算中换流站等效为**电流源**（不同于发电机的电压源模型）：
$$I_{fault,conv} = I_{dc,limit} \cdot \frac{3\sqrt{2}}{\pi} \cdot k \cdot \cos\alpha$$

#### HVDC暂态动态模型

**直流电压控制**（逆变侧，一阶简化）：
$$\frac{dV_{dc}}{dt} = \frac{1}{T_v}(V_{ref} - V_{dc})$$

**电流控制**（整流侧，一阶简化）：
$$\frac{dI_{dc}}{dt} = \frac{1}{T_c}(I_{ref} - I_{dc})$$

**功率调制**（紧急功率支援）：
$$P_{mod} = K_P \cdot \Delta\omega + K_{\dot{P}} \cdot \frac{dP_{ac}}{dt}$$

通过检测送端/受端的频率偏差或功角变化，快速调制直流传输功率，为交流系统提供阻尼。

**输入输出接口设计**

```python
# psa4teaching/stability/hvdc_dynamic.py

@dataclass
class HVDCDynamicModel:
    """HVDC暂态动态模型"""
    rectifier: ConverterStation
    inverter: ConverterStation
    hvdc_line: HVDCLine
    
    # 控制参数
    T_voltage: float = 0.05       # 电压控制时间常数 (s)
    T_current: float = 0.02       # 电流控制时间常数 (s)
    K_voltage: float = 5.0         # 电压控制放大倍数
    K_current: float = 10.0        # 电流控制放大倍数
    
    # 功率调制参数（可选）
    power_modulation_enabled: bool = False
    K_p_omega: float = 0.0         # 频率偏差增益
    K_p_dPdt: float = 0.0          # 有功变化率增益
    P_mod_max: float = 0.3          # 调制功率上限
    
    # 状态变量
    V_dc: float = 1.0
    I_dc: float = 1.0
    
    def derivatives(self, state: NDArray, t: float, inputs: Dict) -> NDArray:
        """返回状态导数 dstate/dt（用于暂态仿真积分）"""
    pass
    
    def get_ac_power_injection(self, V_ac_r: complex, V_ac_i: complex) -> Tuple[float, float, float, float]:
        """计算对交流系统的有功/无功注入 (P_r, Q_r, P_i, Q_i)"""
    pass


def simulate_hvdc_transient(
    ac_generators: List[Generator],
    hvdc_models: List[HVDCDynamicModel],
    ac_ybus: YBusResult,
    fault_bus: int,
    fault_time: float = 0.0,
    fault_clear: float = 0.15,
    t_end: float = 5.0,
    dt: float = 0.005,
    verbose: bool = False
) -> TransientStabilityResult:
    """
    含HVDC的暂态稳定仿真
    
    在原有发电机转子运动方程基础上，
    增加HVDC动态方程，交替求解交流网络和直流系统。
    """
    pass


def hvdc_short_circuit_contribution(
    hvdc_line: HVDCLine,
    rectifier: ConverterStation,
    inverter: ConverterStation,
    fault_bus: int,
    fault_type: str = "three_phase",
    verbose: bool = False
) -> Dict[str, float]:
    """
    计算HVDC换流站对短路电流的贡献
    
    Returns:
        {"rectifier_current": ..., "inverter_current": ..., "total_contribution": ...}
    """
    pass
```

**与现有模块的关联**
- 扩展 `stability/transient.py`：在发电机方程之外增加HVDC动态
- 扩展 `shortcircuit/symmetric.py`：增加换流站电流源模型
- 复用 `stability/small_signal.py` 的特征值分析（扩展状态矩阵）
- 新建 `stability/hvdc_dynamic.py`

**优先级**：中  
**预计难度**：难

---

### 9.4 HVDC的教学示例与可视化

**教学目标**
- 通过典型算例帮助理解HVDC运行特性
- 可视化HVDC在不同控制模式下的运行范围
- 对比交流输电与直流输电的特性

**输入输出接口设计**

```python
# examples/hvdc_demo.py

def demo_hvdc_control_modes():
    """演示不同控制模式下HVDC的运行特性"""
    pass

def demo_hvdc_pv_curve():
    """绘制HVDC功率-电压特性曲线"""
    pass

def demo_ac_vs_dc_comparison():
    """对比交流输电与直流输电的经济性"""
    pass

def demo_mtdc_power_sharing():
    """演示MTDC在不同控制策略下的功率分配"""
    pass
```

**优先级**：低  
**预计难度**：易

---

## 10. 扩展路线图与优先级总览

### 10.1 优先级总表

| 模块 | 子模块 | 王锡凡章节 | 优先级 | 难度 | 预计工作量（人天） |
|------|--------|------------|--------|------|------------------|
| **高级潮流计算** | 连续潮流 | 第四章 | **高** | 难 | 5-7 |
| | 最优潮流 | 第三章 | **高** | 难 | 7-10 |
| | 三相潮流 | 第三章 | 中 | 难 | 5-7 |
| | 交直流混联 | 第六章 | 中 | 难 | 5-7 |
| **HVDC** | 两端HVDC稳态 | 第六章 6.3节 | **高** | 难 | 5-7 |
| | 多端直流(MTDC) | 第六章 6.4节 | 中 | 难 | 5-7 |
| | HVDC暂态/短路 | 第八章 8.4节 | 中 | 难 | 5-7 |
| | HVDC教学示例 | — | 低 | 易 | 1-2 |
| **状态估计** | 加权最小二乘 | 第五章 | **高** | 中 | 3-5 |
| | 不良数据检测 | 第五章 | **高** | 中 | 2-3 |
| | 量测冗余度 | 第五章 | 中 | 易 | 1-2 |
| **电压稳定性** | 连续潮流/PV曲线 | 第四章/六章 | **高** | 中 | 3-5 |
| | 灵敏度指标 | 第六章 | **高** | 中 | 2-3 |
| | 模态分析 | 第六章 | **高** | 中 | 3-4 |
| | 裕度计算 | 第六章 | **高** | 中 | 2-3 |
| **高级稳定** | 低频振荡/PSS | 第七章 | **高** | 难 | 5-7 |
| | 多机完整模型链 | 第七章 | **高** | 难 | 7-10 |
| | EEAC扩展 | 第八章 | 中 | 中 | 3-5 |
| **规划可靠性** | 发电可靠性 | 第九章 | 中 | 中 | 3-4 |
| | 输电可靠性 | 第九章 | 中 | 中 | 3-4 |
| **电力市场** | 经济调度 | 第十一章 | 中 | 中 | 3-4 |
| | LMP计算 | 第十二章 | 中 | 中 | 3-4 |
| **FACTS/新能源** | FACTS模型 | 第六章 | **高** | 中 | 3-5 |
| | DFIG模型 | 第十三章 | **高** | 难 | 5-7 |
| | 光伏模型 | 第十三章 | 中 | 中 | 2-3 |
| | 储能模型 | 第十三章 | 中 | 中 | 2-3 |

### 10.2 建议实施路线

**第一阶段（1-2个月）—— 核心扩展**
1. 连续潮流法（CPF）→ 电压稳定分析基础
2. 加权最小二乘状态估计 → 现代EMS核心功能
3. 电压稳定灵敏度指标 → 在线评估工具
4. FACTS器件模型（SVC/STATCOM）→ 现代电网必备

**第二阶段（2-3个月）—— 深度分析**
5. 最优潮流（OPF）→ 经济调度基础
6. 两端HVDC稳态模型 → 交直流混联潮流
7. 低频振荡分析 → 小干扰稳定深度分析
8. 多机完整模型链 → 含励磁/调速/PSS
9. DFIG风电模型 → 新能源接入基础

**第三阶段（3-4个月）—— 高级专题**
10. 不良数据检测与辨识
11. 多端直流(MTDC)
12. 模态分析法
13. 等面积法则（EEAC）
14. 电力市场（LMP计算）
15. 发电/输电可靠性
16. HVDC暂态/短路影响

### 10.3 目录结构（扩展后）

```
psa4teaching/
├── models/
│   ├── bus.py          # 扩展：三相节点
│   ├── line.py         # 扩展：三相线路、FACTS
│   ├── transformer.py  # 扩展：三相变压器
│   ├── generator.py    # 扩展：励磁/调速引用
│   ├── load.py
│   ├── exciter.py     # 新增：励磁系统
│   ├── governor.py    # 新增：调速器
│   ├── pss.py         # 新增：电力系统稳定器
│   ├── facts.py       # 新增：FACTS器件
│   ├── hvdc.py               # 新增：HVDC模型
│   ├── mtdc.py               # 新增：多端直流
│   ├── dfig.py        # 新增：双馈风机
│   ├── photovoltaic.py # 新增：光伏系统
│   └── energy_storage.py # 新增：储能系统
├── network/
│   ├── ybus.py        # 扩展：FACTS、三相
│   └── zbus.py
├── powerflow/
│   ├── newton_raphson.py  # 扩展：FACTS电压控制
│   ├── fast_decoupled.py
│   ├── dc_powerflow.py
│   ├── continuation.py    # 新增：连续潮流
│   ├── optimal.py         # 新增：最优潮流
│   ├── three_phase.py    # 新增：三相潮流
│   ├── hvdc_powerflow.py     # 新增：HVDC潮流
│   └── acdc_powerflow.py # 新增：交直流潮流
├── shortcircuit/
│   ├── symmetric.py
│   ├── asymmetric.py
│   ├── sequence_network.py
│   └── gb15544.py
├── stability/
│   ├── transient.py              # 扩展：DFIG、储能
│   ├── small_signal.py           # 扩展：完整模型链
│   ├── low_frequency_oscillation.py # 新增：低频振荡
│   ├── multi_machine_detailed.py # 新增：多机详细模型
│   ├── eeac.py                  # 新增：扩展等面积法则
│   ├── hvdc_dynamic.py        # 新增：HVDC暂态
│   └── pss.py                  # 新增：PSS模型
├── state_estimation/
│   ├── wls.py                  # 新增：加权最小二乘
│   ├── bad_data.py             # 新增：不良数据检测
│   └── redundancy.py          # 新增：冗余度分析
├── voltage_stability/
│   ├── indices.py              # 新增：灵敏度指标
│   ├── modal.py               # 新增：模态分析
│   └── margin.py              # 新增：裕度计算
├── reliability/
│   ├── generation.py          # 新增：发电可靠性
│   └── transmission.py        # 新增：输电可靠性
├── market/
│   ├── economic_dispatch.py   # 新增：经济调度
│   └── lmp.py                # 新增：节点边际电价
├── examples/
│   ├── basic_usage.py
│   ├── cpm_demo.py           # 新增：连续潮流示例
│   ├── opf_demo.py           # 新增：最优潮流示例
│   ├── state_est_demo.py     # 新增：状态估计示例
│   ├── voltage_stab_demo.py  # 新增：电压稳定示例
│   ├── oscillation_demo.py   # 新增：低频振荡示例
│   ├── facts_demo.py         # 新增：FACTS示例
│   ├── hvdc_demo.py           # 新增：HVDC示例
│   └── mtdc_demo.py           # 新增：MTDC示例
└── tests/
    ├── test_models.py
    ├── test_powerflow.py
    ├── test_state_estimation.py  # 新增
    ├── test_voltage_stability.py # 新增
    ├── test_stability_advanced.py # 新增
    ├── test_hvdc.py              # 新增
    └── ...
```

---

## 11. 总结

本规划基于王锡凡《现代电力系统分析》教材，将 `psa4teaching` 从本科教学层次扩展至研究生教学层次。

**核心扩展方向**：
1. **高级潮流**：连续潮流、最优潮流（内点法）、三相潮流、交直流混联
2. **状态估计**：WLS估计、不良数据检测、冗余度分析
3. **电压稳定**：PV曲线、灵敏度指标、模态分析、裕度计算
4. **高级稳定**：低频振荡、PSS设计、多机完整模型、EEAC
5. **规划可靠性**：发电/输电可靠性、N-1/N-2分析
6. **电力市场**：经济调度、机组组合、LMP计算
7. **FACTS/新能源**：SVC/STATCOM/TCSC、DFIG、光伏、储能
8. **HVDC**：两端HVDC、多端直流(MTDC)、HVDC暂态/短路

**教学价值**：
- 为研究生提供从理论到实现的完整教学工具
- 支持前沿课题研究（新能源、电力市场、FACTS、HVDC互联）
- 模块化设计便于扩展和定制

---
*文档版本：v1.1 | 日期：2026-05-16 | 基于王锡凡《现代电力系统分析》*
