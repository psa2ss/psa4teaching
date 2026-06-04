# ENTSO-E SMIB 标准测试系统仿真教程

> 基于 ENTSO-E SG SPD 报告《Documentation on Controller Tests in Test Grid Configurations》（2013-11-26）  
> 目标读者：电气工程本科生（已修《电力系统暂态分析》）  
> 预计学习时间：4-6 学时

---

## 目录

1. [psa4teaching 与 ENTSO-E 代码的关系](#1-psa4teaching-与-entso-e-代码的关系)
2. [SMIB 系统拓扑与参数](#2-smib-系统拓扑与参数)
3. [元件数学模型](#3-元件数学模型)
   - [3.1 同步发电机（3 阶 + 双轴凸极）](#31-同步发电机3-阶--双轴凸极)
   - [3.2 TGOV1 调速器](#32-tgov1-调速器)
   - [3.3 SEXS 励磁系统](#33-sexs-励磁系统)
   - [3.4 PSS2A 电力系统稳定器](#34-pss2a-电力系统稳定器)
   - [3.5 网络方程（SMIB 特殊化）](#35-网络方程smib-特殊化)
   - [3.6 恒阻抗负荷](#36-恒阻抗负荷)
4. [仿真框架](#4-仿真框架)
5. [测试案例 1：电压参考值阶跃](#5-测试案例-1电压参考值阶跃)
6. [测试案例 2：负荷有功阶跃](#6-测试案例-2负荷有功阶跃)
7. [测试案例 3：三相短路故障](#7-测试案例-3三相短路故障)
8. [动手实践：从零实现](#8-动手实践从零实现)

---

## 1. psa4teaching 与 ENTSO-E 代码的关系

### 1.1 层次结构

```
┌─────────────────────────────────────────────────┐
│  examples/entsoe_smib_demo.py                    │  ← ENTSO-E 仿真脚本
│  examples/entsoe_smib_validation.ipynb           │  ← 验证 Notebook
├─────────────────────────────────────────────────┤
│  psa4teaching/models/                            │  ← 元件模型库
│    ├── generator.py    Generator (发电机)         │
│    ├── governor.py     TGOV1Params (调速器)       │
│    ├── exciter.py      SEXSParams (励磁系统)      │
│    ├── pss.py          PSS2AParams (稳定器)       │
│    ├── bus.py          Bus, BusType (节点)        │
│    ├── line.py         Line (线路)               │
│    ├── transformer.py  Transformer (变压器)       │
│    └── load.py         Load, LoadModel (负荷)     │
├─────────────────────────────────────────────────┤
│  psa4teaching/powerflow/    ← 潮流计算            │
│  psa4teaching/stability/    ← 暂态/小干扰稳定      │
│  psa4teaching/network/      ← 导纳矩阵构建         │
└─────────────────────────────────────────────────┘
```

### 1.2 设计原则

psa4teaching 遵循**模型-算法分离**原则：

- **models/** 目录只存放 **dataclass 数据类**：存储参数、提供单步计算接口（如 `compute(delta_omega, dt, state)`），**不包含**仿真循环逻辑。
- **entsoe_smib_demo.py** 位于 examples/ 目录，是**算法层**：组织系统拓扑、建立初始条件、运行 RK4 时间积分循环、绘图。它**调用** psa4teaching.models 中的类获取元件参数和单步输出，但仿真框架（循环、事件处理、初值求解）本身不属于 psa4teaching 核心库。

```python
# 示例：demo 脚本如何使用 psa4teaching
from psa4teaching.models import Generator, TGOV1Params, SEXSParams, PSS2AParams

gen = Generator(Sb=500.0, Vb=21.0, Xd=2.0, Xd_prime=0.35, H=4.0, ...)
gov = TGOV1Params(R=0.05, T1=0.5, T2=3.0, T3=10.0)
exc = SEXSParams(K=200.0, TA=3.0, TB=10.0, TE=0.05)

# 在仿真循环中调用
Pm, new_gov_state = gov.compute_rk4(delta_omega, dt, gov_state, P_ref)
Efd, new_exc_state = exc.compute_rk4(V_ref, Vt, V_S, dt, exc_state)
```

### 1.3 依赖关系

- `entsoe_smib_demo.py` **import** → `psa4teaching.models`（参数类）
- `entsoe_smib_demo.py` **自实现** → 网络方程、RK4 积分器、初值求解器、SMIB 状态管理
- `entsoe_smib_validation.ipynb` **import** → `entsoe_smib_demo`（调用 run/plot 函数）

---

## 2. SMIB 系统拓扑与参数

### 2.1 四节点系统

```
NGEN (21 kV) ──── NTLV (21 kV) ──── NTHV (380 kV) ──── NGRID (380 kV)
   │                  │                   │                  │
  GEN              S-GEN                T-GEN             S-GRID
  +GOV               ↑             升压变压器              ↑       ── GRID
  +AVR          机端断路器          419/21 kV          高压侧断路器   无穷大母线
  +PSS                                                      │      Sk''=2500 MVA
                                                          GRIDL
                                                      恒阻抗负荷
```

**四个节点：**

| 节点 | 名称 | 电压 (kV) | 类型 | 说明 |
|------|------|-----------|------|------|
| 1 | NGEN | 21 | PV | 发电机端 |
| 2 | NTLV | 21 | PQ | 变压器低压侧 |
| 3 | NTHV | 380 | PQ | 变压器高压侧（故障点） |
| 4 | NGRID | 380 | SLACK | 无穷大母线，V=1.05∠0° |

**线路参数（标幺值，S_base=500 MVA）：**

| 线路 | R (pu) | X (pu) |
|------|--------|--------|
| L1 (NGEN→NTLV) | 0.001 | 0.01 |
| L2 (NTHV→NGRID) | 0.022 | 0.219 |

**变压器 T1 (NTLV→NTHV)：** RT=0.0015, XT=0.16, k=419/21

### 2.2 发电机参数（S_base=500 MVA）

| 参数 | 数值 | 说明 |
|------|------|------|
| $X_d$ | 2.0 | d 轴同步电抗 |
| $X_d'$ | 0.35 | d 轴暂态电抗 |
| $X_q$ | 1.8 | q 轴同步电抗（凸极机） |
| $T_{d0}'$ | 5.143 s | d 轴开路暂态时间常数 |
| $H$ | 4.0 s | 惯性常数 |
| $D$ | 0 | 阻尼系数 |

### 2.3 控制器参数

| TGOV1 | 值 | SEXS | 值 | PSS2A | 值 |
|--------|----|------|-----|-------|-----|
| R | 0.05 | K | 200 | KS1 | 10 |
| T1 | 0.5 s | TA | 3 s | TW1~TW3 | 2 s |
| T2 | 3.0 s | TB | 10 s | T1/T2 | 0.25/0.03 |
| T3 | 10.0 s | TE | 0.05 s | T3/T4 | 0.15/0.015 |
| VMAX | 1.0 | EMAX | 4.0 | VSTMAX | 0.1 |

### 2.4 恒阻抗负荷 GRIDL（NGRID 母线）

$P_{GRIDL}=475\text{ MW}=0.95\text{ pu}$，$Q_{GRIDL}=76\text{ MVar}=0.152\text{ pu}$（在 V=1.0 pu 下）

---

## 3. 元件数学模型

### 3.1 同步发电机（3 阶 + 双轴凸极）

发电机采用 **3 阶实用模型**（李光琦《电力系统暂态分析》第四章）：

**状态变量（3 维）：**
- $\delta$ — 转子功角（rad），q 轴超前无穷大母线电压的角度
- $\omega$ — 转子转速（pu）
- $E_q'$ — d 轴暂态电势（pu）

**微分方程：**

$$\frac{d\delta}{dt} = (\omega - 1.0) \cdot \omega_{base}, \quad \omega_{base} = 2\pi f = 100\pi \text{ rad/s}$$

$$\frac{d\omega}{dt} = \frac{P_m - P_e - D \cdot \Delta\omega}{2H}, \quad \Delta\omega = \omega - 1.0$$

$$\frac{dE_q'}{dt} = \frac{E_{fd} - E_q' - (X_d - X_d') \cdot I_d}{T_{d0}'} \quad \text{(Kundur Eq. 3.177)}$$

> **关键点**：第 3 式中的 $(X_d - X_d') \cdot I_d$ 是**电枢反应项**。它表示 d 轴电枢电流 $I_d$ 的去磁效应会使暂态电势 $E_q'$ 衰减。若忽略此项（经典模型），仿真结果将与 ENTSO-E 报告严重不符。

**定子电压方程（双轴模型，dq 转子坐标系，q 轴超前 d 轴 90°）：**

$$V_{td} = -X_q \cdot I_q$$
$$V_{tq} = E_q' - X_d' \cdot I_d$$
$$V_t = \sqrt{V_{td}^2 + V_{tq}^2}$$

**电磁功率（含凸极效应）：**

$$P_e = \frac{E_q' \cdot V_\infty}{X_{d\Sigma}} \sin\delta + \frac{V_\infty^2}{2} \left(\frac{1}{X_{q\Sigma}} - \frac{1}{X_{d\Sigma}}\right) \sin(2\delta)$$

其中 $X_{d\Sigma} = X_d' + X_{ext}$，$X_{q\Sigma} = X_q + X_{ext}$。

> **关键点**：第二项（$\sin(2\delta)$ 项）是**凸极效应**产生的磁阻功率。隐极机 $X_d = X_q$ 时此项为零。本系统 $X_q=1.8 \neq X_d'=0.35$，必须保留此项，否则初始功角将从正确的 ~72° 偏移至 ~39°。

**dq 轴电流（网络解）：**

$$I_d = \frac{E_q' - V_\infty \cos\delta}{X_{d\Sigma}}$$
$$I_q = \frac{V_\infty \sin\delta}{X_{q\Sigma}}$$

**代码对应**：`compute_smib_voltages()` 函数（entsoe_smib_demo.py 第 121-182 行）。

---

### 3.2 TGOV1 调速器

TGOV1 是 IEEE 标准的简化蒸汽涡轮调速器模型（ENTSO-E Fig 2-2）。

**传递函数：**

$$G_{gov}(s) = \frac{1}{R} \cdot \frac{1}{1 + sT_1} \cdot \frac{1 + sT_2}{1 + sT_3}$$

**框图（信号流）：**

```
P_ref ──(+)──► [1/R] ──► [VMIN, VMAX] ──► [1/(1+sT1)] ──┬──► Dt ──────────────┐
           ┃                                             │                      ├──(+)──► PMECH
           ┃Δω                                          └──► [(1+sT2)/(1+sT3)] ─┘
```

**输入：** $u = P_{ref} - \Delta\omega / R$（$P_{ref}$ 恒定，仅 $\Delta\omega$ 反馈）

**状态变量（2 维动态 + 1 维存储）：**
- $x_1$：伺服系统 $1/(1+sT_1)$ 的输出
- $x_2$：汽轮机 $1/(1+sT_3)$ 的输出
- $x_3$：$P_{MECH}$（输出存储）

**离散时间递推（欧拉法）：**

$$x_1^{k+1} = x_1^k + \frac{dt}{T_1}(u_{limited} - x_1^k)$$
$$x_2^{k+1} = x_2^k + \frac{dt}{T_3}(x_1^k - x_2^k)$$
$$P_{MECH}^{k+1} = \frac{T_2}{T_3} \cdot x_1^{k+1} + \left(1 - \frac{T_2}{T_3}\right) \cdot x_2^{k+1} + D_t \cdot x_1^{k+1}$$

> **关键点**：$P_{ref}$ 在仿真中**保持恒定**。调速器仅通过 $\Delta\omega / R$（频率偏差经调差系数）感知负荷变化。若将 $P_{ref}$ 随负荷同步阶跃，则绕过了调速器动态响应，无法验证调速器模型。

**代码对应**：`TGOV1Params.compute_rk4()`（governor.py 第 185-238 行）。

---

### 3.3 SEXS 励磁系统

SEXS 是 ENTSO-E 定义的简化励磁系统（Simplified Excitation System），为**纯比例式**（禁用 PI 环节），因此存在固有稳态下垂。

**传递函数：**

$$G_{exc}(s) = \frac{1 + sT_A}{1 + sT_B} \cdot \frac{K}{1 + sT_E}$$

**两级串联：**
1. **电压调节器**（相位补偿）：$G_1(s) = \dfrac{1 + sT_A}{1 + sT_B}$
2. **励磁机**（惯性环节）：$G_2(s) = \dfrac{K}{1 + sT_E}$

**输入信号：** $V_{err} = V_{ref} - V_t + V_S$

其中 $V_S$ 来自 PSS 输出，稳态值为 0。

**状态变量（2 维动态 + 1 维存储）：**
- $x_1$：相位补偿环节 $1/(1+sT_B)$ 的输出
- $x_2$：励磁机 $1/(1+sT_E)$ 的输出
- $x_3$：$E_{fd}$（输出存储，经 $[E_{MIN}, E_{MAX}]$ 限幅）

**离散时间递推：**

$$x_1^{k+1} = x_1^k + \frac{dt}{T_B}(V_{err} - x_1^k)$$
$$y_1 = \frac{T_A}{T_B} \cdot V_{err} + \left(1 - \frac{T_A}{T_B}\right) \cdot x_1^{k+1}$$
$$x_2^{k+1} = x_2^k + \frac{dt}{T_E}(K \cdot y_1 - x_2^k)$$
$$E_{fd}^{k+1} = \text{clip}(x_2^{k+1}, E_{MIN}, E_{MAX})$$

**稳态下垂特性：**

SEXS 为纯比例式（无积分），直流增益为 $K$。稳态时：

$$E_{fd} = K \cdot (V_{ref} - V_t) \quad \Rightarrow \quad V_t = V_{ref} - \frac{E_{fd}}{K}$$

这意味着 $V_t$ 总是略低于 $V_{ref}$（当 $E_{fd} > 0$ 时）。例如测试案例 3 中 $E_{fd} \approx 2.24$，$K=200$：

$$V_t = 1.05 - 2.24/200 = 1.0388 \text{ pu}$$

> **关键点**：由于存在励磁下垂，不能用 $V_t = V_{ref}$ 作为稳态条件。联立求解时必须包含 $E_{fd} = K(V_{ref} - V_t)$ 这一方程。

**代码对应**：`SEXSParams.compute_rk4()`（exciter.py 第 182-247 行）。

---

### 3.4 PSS2A 电力系统稳定器

PSS2A 是 IEEE Std 421.5-2005 定义的双输入电力系统稳定器。

**框图（ENTSO-E Fig 2-4，简化版 N=0, M=0, T6=0, TW4=0）：**

```
Δω ──► [sTW1/(1+sTW1)] ──► [sTW2/(1+sTW2)] ──► ×KS1 ──┐
                                                         ├──(+)──► ×KS3 ──► [1+sT1] ──► [1+sT3] ──► [VSTMIN,VSTMAX] ──► V_S
Pe ──► [sTW3/(1+sTW3)] ──► [直通,TW4=0] ──► [1/(1+sT7)] ──► ×KS2 ──┘                ────        ────
                                                             1+sT2       1+sT4
```

**双通道设计：**

| 通道 | 输入信号 | 清洗环节 | 功能 |
|------|---------|---------|------|
| 通道 1（IC1=1） | $\Delta\omega$（转速偏差） | TW1, TW2 | 提取转速振荡分量 |
| 通道 2（IC2=3） | $P_e$（电磁功率） | TW3, TW4=0 | 提取功率振荡分量 |

**状态变量（6 维）：**
- $x_1, x_2$：通道 1 清洗环节 TW1, TW2 输出
- $x_3, x_4$：通道 2 清洗环节 TW3 和低通 $1/(1+sT_7)$ 输出
- $x_5, x_6$：两对超前-滞后 $(1+sT_1)/(1+sT_2)$ 和 $(1+sT_3)/(1+sT_4)$ 输出

**清洗环节（Washout）的时域实现：**

传递函数 $H(s) = \dfrac{sT_W}{1 + sT_W}$ 在时域中：

$$\frac{dx}{dt} = \frac{u - x}{T_W}, \quad y = u - x$$

**超前-滞后环节的时域实现：**

传递函数 $H(s) = \dfrac{1 + sT_{lead}}{1 + sT_{lag}}$ 在时域中：

$$\frac{dx}{dt} = \frac{u - x}{T_{lag}}, \quad y = \frac{T_{lead}}{T_{lag}} \cdot u + \left(1 - \frac{T_{lead}}{T_{lag}}\right) \cdot x$$

**输出限幅：** $V_S \in [V_{STMIN}, V_{STMAX}] = [-0.1, 0.1] \text{ pu}$

> **关键点**：PSS 输出 $V_S$ 叠加到励磁系统的电压偏差信号上：$V_{err} = V_{ref} - V_t + V_S$。通过调制励磁电压，PSS 为转子振荡提供**正阻尼**。测试案例 3 中可在 PSS 增益 $K_{S1}=0$ 时观察无 PSS 的弱阻尼振荡。

**代码对应**：`PSS2AParams.compute_stabilizing_signal_rk4()`（pss.py 第 271-300 行）、`_pss2a_derivatives()`（pss.py 第 170-222 行）。

---

### 3.5 网络方程（SMIB 特殊化）

与通用多机系统的 $I = Y \cdot V$ 不同，SMIB 系统可以解析求解。

**正常运行时：**

外部等效电抗（从发电机端到无穷大母线）：

$$X_{ext} = X_{L1} + X_T + X_{L2} = 0.01 + 0.16 + 0.219 = 0.389 \text{ pu}$$

dq 坐标系下的网络方程（Kundur §3.5）：

$$I_d = \frac{E_q' - V_\infty \cos\delta}{X_d' + X_{ext}}, \quad I_q = \frac{V_\infty \sin\delta}{X_q + X_{ext}}$$

> **技巧**：$I_d$ 的分母用 $X_d' + X_{ext}$（d 轴暂态电抗），而 $I_q$ 的分母用 $X_q + X_{ext}$（q 轴同步电抗）。这是因为 $E_q'$ 是 d 轴暂态电势，而 q 轴用同步电抗 $X_q$ 建模（当前为 3 阶模型，无 q 轴暂态动态）。

**三相短路故障期间：**

故障发生在 NTHV 节点。从发电机到故障点：

$$X_{ext}^{fault} = X_{L1} + X_T = 0.01 + 0.16 = 0.17 \text{ pu}$$

发电机端电压在故障期间降为接近零（模型中近似为 0.05 pu），$P_e \approx 0$。

**故障清除后：** 网络恢复为正常拓扑，$X_{ext}$ 恢复为 0.389 pu。

**空载运行（TC1）：**

S-GEN 断开 → $X_{ext} \to \infty$，$I_d = I_q = 0$，$P_e = 0$，$V_t = E_q'$。

**隔离运行（TC2）：**

S-GEN 断开，恒阻抗负荷 $R_L$ 直接接在 NGEN 发电机端。等效电路：

$$E_q' \angle\delta \to jX_d' \to R_L \text{（负荷）}$$

圆转子近似（无凸极）：

$$V_t = \frac{E_q' \cdot R_L}{\sqrt{R_L^2 + X_d'^2}}, \quad I_d = -\frac{E_q' \cdot X_d'}{R_L^2 + X_d'^2}$$

---

### 3.6 恒阻抗负荷

恒阻抗负荷的功率与电压平方成正比：

$$P(V) = P_0 \cdot \left(\frac{V}{V_0}\right)^2, \quad Q(V) = Q_0 \cdot \left(\frac{V}{V_0}\right)^2$$

等效阻抗（在 $V_0 = 1.0$ pu 时）：

$$Z_L = \frac{1}{P_0 - jQ_0}$$

对于 TC2 的纯电阻附加负荷（$P_L = 0.76$ pu）：

$$R_L = \frac{1}{P_L} = \frac{1}{0.76} = 1.3158 \text{ pu}$$

负荷阶跃后（$P_L' = 0.8075$ pu）：

$$R_L' = \frac{1}{0.8075} = 1.2384 \text{ pu}$$

---

## 4. 仿真框架

### 4.1 状态向量组织

SMIBState 包含 15 维状态：

```python
@dataclass
class SMIBState:
    delta: float = 0.0          # 发电机: 转子功角 (rad)
    omega: float = 1.0          # 发电机: 转速 (pu)
    Eq_prime: float = 1.0       # 发电机: d轴暂态电势 (pu)
    exc_state: np.ndarray       # 励磁系统: [x1, x2, x3]  (3维)
    gov_state: np.ndarray       # 调速器:   [x1, x2, x3]  (3维)
    pss_state: np.ndarray       # PSS:      [x1..x6]      (6维)
    Efd: float = 1.0            # 工作变量: 励磁电压输出
    PMECH: float = 0.0          # 工作变量: 机械功率输出
```

### 4.2 RK4 积分

采用经典四阶 Runge-Kutta 法，步长 $\Delta t = 1$ ms：

$$\begin{aligned}
k_1 &= f(t_n, y_n) \\
k_2 &= f(t_n + \tfrac{h}{2}, y_n + \tfrac{h}{2}k_1) \\
k_3 &= f(t_n + \tfrac{h}{2}, y_n + \tfrac{h}{2}k_2) \\
k_4 &= f(t_n + h, y_n + h k_3) \\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

### 4.3 导数函数组织

每一步计算顺序：

```
1. 网络解:  compute_smib_voltages(δ, Eq', X_ext) → Vt, Pe, Id, Iq
2. PSS:     compute_stabilizing_signal(Δω, Pe, dt, pss_state) → V_S
3. 励磁:    compute_rk4(V_ref, Vt, V_S, dt, exc_state) → Efd
4. 调速器:  compute_rk4(Δω, dt, gov_state, P_ref) → PMECH
5. 发电机:  dδ/dt, dω/dt, dEq'/dt
```

### 4.4 初始条件求解

初始条件必须满足**稳态一致性**——即所有微分方程的右侧为零，且励磁/调速器内部状态匹配。

**并网稳态（TC3）使用 fsolve 求解 3 个方程：**

$$\begin{cases}
P_e(E_q', \delta) = P_{target} & \text{有功平衡} \\
V_t(E_q', \delta) = V_{ref} - E_{fd}/K & \text{励磁下垂} \\
E_{fd} = E_q' + (X_d - X_d') \cdot I_d & \text{励磁绕组稳态}
\end{cases}$$

**隔离负荷稳态（TC2）直接解析求解：**

$$E_q' = \frac{K \cdot V_{ref}}{K \cdot \alpha + 1 - (X_d - X_d') \cdot \beta}$$

其中 $\alpha = \frac{R_L}{\sqrt{R_L^2 + X_d'^2}}$，$\beta = \frac{X_d'}{R_L^2 + X_d'^2}$

---

## 5. 测试案例 1：电压参考值阶跃

### 5.1 物理场景

发电机空载运行（S-GEN 断开），仅励磁系统投入。测试 AVR 的动态响应特性。

### 5.2 投入的数学模型

| 元件 | 状态 |
|------|------|
| 发电机（3 阶） | ✅ $\delta, \omega, E_q'$ |
| 励磁系统 SEXS | ✅ 3 维状态 |
| 调速器 TGOV1 | ✅ 状态置零（$P_{MECH}=0$） |
| PSS PSS2A | ❌ 复位（$V_S=0$） |
| 网络方程 | 空载：$V_t=E_q', P_e=0, I_d=I_q=0$ |

### 5.3 微分方程构成

$$\begin{cases}
\frac{d\delta}{dt} = (\omega - 1.0) \cdot \omega_{base} \\
\frac{d\omega}{dt} = \frac{0 - 0 - D \cdot \Delta\omega}{2H} \quad (P_m=0, P_e=0) \\
\frac{dE_q'}{dt} = \frac{E_{fd} - E_q'}{T_{d0}'} \quad (I_d=0)
\end{cases}$$

励磁系统动态（3 维状态，见 §3.3）。

**总状态维数：** 发电机 3 + 励磁 3 + 调速器 3（冻结）+ PSS 6（冻结）= **15 维**。

### 5.4 初始条件（$t < 0.1$ s）

$$V_{ref}(0) = 1.0, \quad V_t(0) = 1.0, \quad E_{fd}(0) = 1.0, \quad E_q'(0) = 1.0$$

励磁状态初始化为 $x_1 = V_{err}/K = 0.005$，$x_2 = x_3 = 1.0$。

### 5.5 事件（$t = 0.1$ s）

$$V_{ref}: 1.0 \to 1.05 \text{ pu} \quad (+5\% \text{ 阶跃})$$

### 5.6 期望响应（ENTSO-E 报告 Fig 5-1, 5-2）

- **$U_{NGEN}$**（Fig 5-1）：从 1.0 pu 上升并稳定至 1.045 pu（略低于 1.05，因励磁下垂）
- **$E_{FD}$**（Fig 5-2）：快速上升，约需 0.5~1 s 完成动态响应

### 5.7 仿真参数

$$\Delta t = 0.001 \text{ s}, \quad t_{end} = 2.0 \text{ s}, \quad \text{事件时刻} = 0.1 \text{ s}$$

---

## 6. 测试案例 2：负荷有功阶跃

### 6.1 物理场景

S-GEN 断开，发电机孤立运行。NGEN 节点接入恒阻抗负荷（纯电阻 $R_L$）。PSS 必须关断。测试**调速器**的动态响应。

### 6.2 投入的数学模型

| 元件 | 状态 |
|------|------|
| 发电机（3 阶） | ✅ $\delta, \omega, E_q'$ |
| 励磁系统 SEXS | ✅ 3 维状态 |
| 调速器 TGOV1 | ✅ 3 维状态（核心测试对象） |
| PSS PSS2A | ❌ $K_{S1}=0$（关断） |
| 网络方程 | 隔离负荷模型（§3.5） |

### 6.3 微分方程构成

$$\begin{cases}
\frac{d\delta}{dt} = (\omega - 1.0) \cdot \omega_{base} \\
\frac{d\omega}{dt} = \frac{P_m - P_e - D \cdot \Delta\omega}{2H} \\
\frac{dE_q'}{dt} = \frac{E_{fd} - E_q' - (X_d - X_d') \cdot I_d}{T_{d0}'}
\end{cases}$$

其中 $P_e, I_d, V_t$ 由隔离负荷网络方程求解（§3.5, §3.6）。

调速器动态（3 维状态，见 §3.2），$P_{ref}$ **保持恒定**。

励磁系统动态（3 维状态，见 §3.3）。

**总状态维数：** 15 维。

### 6.4 初始条件（$t < 0.1$ s）

$$V_{ref} = 1.0, \quad P_L = 380 \text{ MW} = 0.76 \text{ pu}, \quad R_L = 1/0.76 = 1.3158 \text{ pu}$$

解析求解稳态（含励磁下垂）：

$$E_q'(0) = \frac{K \cdot V_{ref}}{K\alpha + 1 - (X_d - X_d')\beta} = 1.0311$$

$$V_t(0) = \alpha E_q' = 0.9965 \text{ pu}$$

$$E_{fd}(0) = E_q' + (X_d - X_d') \cdot (- \beta E_q') = 0.7099 \text{ pu}$$

$$P_{ref} = P_e(0) = E_q'^2 R_L / (R_L^2 + X_d'^2) = 0.7546 \text{ pu}$$

调速器状态初始化为 $x_1 = x_2 = x_3 = P_e(0)$；励磁状态 $x_1 = E_{fd}/K$，$x_2 = x_3 = E_{fd}$。

### 6.5 事件（$t = 0.1$ s）

$$\Delta P_L = +0.05 \times \frac{475}{500} = +0.0475 \text{ pu}$$

$$R_L: 1.3158 \to 1.2384 \text{ pu}$$

### 6.6 期望响应（ENTSO-E 报告 Fig 5-3 ~ 5-6）

- **$U_{NGEN}$**（Fig 5-3）：电压受励磁系统调节，稳定在 ~0.997 pu
- **$P_G, P_{MECH}$**（Fig 5-4）：$P_e$ 瞬间跟随负荷；$P_{MECH}$ 经调速器延迟后跟踪，有超调
- **$Q_G$**（Fig 5-5）：纯电阻负荷，机端无功为 0
- **$\omega_G$**（Fig 5-6）：转速从 1.0 降至最低 ~0.976，后恢复至 ~0.998（对应 $R=0.05$ 的稳态下垂 $\Delta\omega = -R \cdot \Delta P = -0.05 \times 0.0475 = -0.0024$）

### 6.7 仿真参数

$$\Delta t = 0.001 \text{ s}, \quad t_{end} = 15.0 \text{ s}, \quad \text{事件时刻} = 0.1 \text{ s}$$

---

## 7. 测试案例 3：三相短路故障

### 7.1 物理场景

基准潮流工况（S-GEN 闭合），全部控制器（AVR + TGOV1 + PSS2A）投入。NTHV 母线发生金属性三相短路 0.1 s 后清除。测试**全系统的暂态稳定能力**。

### 7.2 投入的数学模型

| 元件 | 状态 |
|------|------|
| 发电机（3 阶） | ✅ $\delta, \omega, E_q'$（含凸极效应） |
| 励磁系统 SEXS | ✅ 3 维状态（含 $E_{MAX}=4.0$ 限幅） |
| 调速器 TGOV1 | ✅ 3 维状态 |
| PSS PSS2A | ✅ 6 维状态（双通道，转速+功率） |
| 网络方程 | 并网 SMIB（§3.5），含故障拓扑切换 |

### 7.3 微分方程构成

与 TC2 相同形式的发电机方程，但 $P_e, I_d, I_q$ 由**并网网络方程**（§3.5）求解，取决于 $X_{ext}$（正常/故障）。

$$\begin{cases}
\frac{d\delta}{dt} = (\omega - 1.0) \cdot \omega_{base} \\
\frac{d\omega}{dt} = \frac{P_m - P_e - D \cdot \Delta\omega}{2H} \\
\frac{dE_q'}{dt} = \frac{E_{fd} - E_q' - (X_d - X_d') \cdot I_d}{T_{d0}'}
\end{cases}$$

励磁 + 调速器 + PSS 全部动态参与。

**总状态维数：** 15 维。

### 7.4 初始条件（$t < 0.1$ s，并网稳态）

使用 fsolve 联立求解 3 方程（§4.4），得到：

$$\delta(0) \approx 72.1^\circ, \quad E_q'(0) \approx 0.917, \quad V_t(0) \approx 1.039 \text{ pu}$$

$$E_{fd}(0) \approx 2.242 \text{ pu}, \quad P_{MECH}(0) = 0.95 \text{ pu}$$

$$V_{ref} = 1.05 \text{ pu}, \quad V_\infty = 1.05 \angle 0^\circ$$

### 7.5 事件序列

| 时间 | 事件 | $X_{ext}$ |
|------|------|-----------|
| $t < 0.1$ s | 正常并网运行 | 0.389 pu |
| $t = 0.1$ s | NTHV 三相金属性短路 | 0.17 pu（至故障点） |
| $t = 0.2$ s | 故障清除 | 0.389 pu（恢复） |
| $t > 0.2$ s | 故障后振荡衰减 | 0.389 pu |

**故障期间**：$V_t \approx 0.05$ pu，$P_e \approx 0$。转子因 $P_m - P_e > 0$ 而加速，$\omega$ 上升。

**故障清除后**：$P_e$ 恢复，$P_e > P_m$ 使转子减速。系统在 PSS 提供的正阻尼下振荡衰减。

### 7.6 期望响应（ENTSO-E 报告 Fig 5-7 ~ 5-14）

| 图号 | 输出量 | 期望行为 |
|------|--------|---------|
| Fig 5-7 | $U_{NGEN}$ | 故障→0.05，清除→恢复振荡 |
| Fig 5-8 | $E_{FD}$ | 强励至 $E_{MAX}=4.0$，清除后回落 |
| Fig 5-9 | $P_G$ | 故障→0，清除→功率振荡 |
| Fig 5-10 | $Q_G$ | 故障期间大幅波动 |
| Fig 5-11 | $\omega_G$ | 加速→减速→振荡衰减 |
| Fig 5-12 | $V_{OTHSG}$ | PSS 输出在 $\pm 0.1$ pu 限幅内 |
| Fig 5-13 | $P_{GRIDL}$ | NGRID 负荷有功（V=1.05 恒定） |
| Fig 5-14 | $Q_{GRIDL}$ | NGRID 负荷无功（V=1.05 恒定） |

### 7.7 仿真参数

$$\Delta t = 0.001 \text{ s}, \quad t_{end} = 10.0 \text{ s}, \quad t_{fault\_start} = 0.1 \text{ s}, \quad t_{fault\_clear} = 0.2 \text{ s}$$

---

## 8. 动手实践：从零实现

### 8.1 建议的实现步骤

**第 1 步：搭建系统框架（~30 分钟）**

1. 定义 `SMIBState` dataclass（15 维状态）
2. 实现 `compute_smib_voltages()` 网络方程（含凸极效应、故障条件、空载条件）
3. 实现 `rk4_integrate()` 通用积分框架

**第 2 步：实现控制器单步计算（~1 小时）**

4. 在 TGOV1Params 中实现 `compute(delta_omega, dt, state, P_ref)`（先用欧拉法验证，再升级 RK4）
5. 在 SEXSParams 中实现 `compute(V_ref, Vt, V_S, dt, state)`
6. 在 PSS2AParams 中实现 `compute(delta_omega, Pe, dt, state)`

**第 3 步：实现测试案例 1（~30 分钟）**

7. 编写 `run_test_case_1()`：空载条件、V_ref 阶跃、仅励磁投入
8. 验证：$V_t$ 应从 1.0 上升至接近 1.05

**第 4 步：实现测试案例 2（~1 小时）**

9. 推导隔离负荷的解析稳态解（含励磁下垂）
10. 编写 `run_test_case_2()`：NGEN 节点 $R_L$ 负荷、$P_{ref}$ 恒定、阶跃 +0.0475 pu
11. 验证：$\omega$ 最低点 ~0.976，$\omega$ 稳态 ~0.998

**第 5 步：实现测试案例 3（~1 小时）**

12. 使用 fsolve 求解并网稳态初始条件（3 方程联立）
13. 编写 `run_test_case_3()`：故障拓扑切换、全部控制器投入
14. 验证：故障期间 $V_t \approx 0$，$E_{fd}$ 达 4.0 限幅

**第 6 步：绘图与对比**

15. 绘制全部 14 张图（Fig 5-1 ~ 5-14）
16. 添加 PSS 对比（$K_{S1}=0$ vs $K_{S1}=10$），观察阻尼差异

### 8.2 常见错误检查清单

| # | 常见错误 | 症状 | 修正方法 |
|---|---------|------|---------|
| 1 | 忘记 $(X_d-X_d')I_d$ 电枢反应项 | $E_q'$ 动态错误，电压响应不符 | 在 $dE_q'/dt$ 中加入 $(X_d-X_d')I_d$ |
| 2 | 忘记凸极效应 $\sin(2\delta)$ 项 | 初始功角 ~39° 而非 ~72° | $P_e$ 公式中加入磁阻功率项 |
| 3 | 使用 $T_d'$（短路）而非 $T_{d0}'$（开路） | 时间常数偏小，响应过快 | 使用 $T_{d0}'=5.143$ s |
| 4 | $P_{ref}$ 在 TC2 中随负荷阶跃 | 调速器响应被绕过 | $P_{ref}$ 保持恒定 = $P_e(0)$ |
| 5 | 初始条件未考虑励磁下垂 | 初始瞬态，$V_t$ 漂移 | 联立 $E_{fd} = E_q' + (X_d-X_d')I_d$ 和 $E_{fd}=K(V_{ref}-V_t)$ |
| 6 | TC2 负荷位置在 NGRID 而非 NGEN | $X_{ext}=0.389$ 而非 $X_d'=0.35$ | 负荷电抗 $R_L$ 直接串联 $jX_d'$ |
| 7 | 欧拉法积分在 15 s 仿真中漂移 | 长期仿真精度差 | 使用 RK4 |
| 8 | 忽略励磁限幅 $[E_{MIN}, E_{MAX}]$ | 故障期间 $E_{fd}$ 无限增长 | 在 SEXS 输出处施加 clip |

### 8.3 参考代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| 系统构建 | `examples/entsoe_smib_demo.py` | 36-114 |
| 网络方程 | `examples/entsoe_smib_demo.py` | 121-182 |
| 并网稳态求解 | `examples/entsoe_smib_demo.py` | 185-250 |
| 隔离负荷稳态 | `examples/entsoe_smib_demo.py` | 253-314 |
| SMIBState 定义 | `examples/entsoe_smib_demo.py` | 339-364 |
| 状态导数 | `examples/entsoe_smib_demo.py` | 371-457 |
| RK4 积分器 | `examples/entsoe_smib_demo.py` | 460-508 |
| TC1 实现 | `examples/entsoe_smib_demo.py` | 579-630 |
| TC2 实现 | `examples/entsoe_smib_demo.py` | 642-840 |
| TC3 实现 | `examples/entsoe_smib_demo.py` | 852-943 |
| 绘图函数 | `examples/entsoe_smib_demo.py` | 949-1070 |
| TGOV1 模型 | `psa4teaching/models/governor.py` | 40-238 |
| SEXS 模型 | `psa4teaching/models/exciter.py` | 44-247 |
| PSS2A 模型 | `psa4teaching/models/pss.py` | 40-300 |
| Generator 模型 | `psa4teaching/models/generator.py` | — |

### 8.4 验证标准

完成实现后，运行 `entsoe_smib_validation.ipynb` 的全部单元格，确认：

| 测试案例 | 关键验证指标 | 容差范围 |
|---------|-------------|---------|
| TC1 | $V_t$ 终值 | $1.045 \pm 0.002$ pu |
| TC2 | $\omega$ 最低点 | $0.976 \pm 0.002$ pu |
| TC2 | $\omega$ 稳态值 | $0.998 \pm 0.001$ pu |
| TC3 | 故障期间 $V_t$ 最小值 | $< 0.06$ pu |
| TC3 | 初始功角 $\delta$ | $72^\circ \pm 3^\circ$ |
| TC3 | $E_{fd}$ 故障期间最大值 | $\geq 3.9$ pu |
| PSS 对比 | 有 PSS 时 $\omega$ 3-4 周期衰减 | 无 PSS 时持续振荡 |

---

## 参考文献

1. ENTSO-E SG SPD, *Documentation on Controller Tests in Test Grid Configurations*, 2013-11-26
2. P. Kundur, *Power System Stability and Control*, McGraw-Hill, 1994
3. IEEE Std 421.5-2016, *IEEE Recommended Practice for Excitation System Models for Power System Stability Studies*
4. 李光琦，《电力系统暂态分析》（第三版），中国电力出版社
5. 陈珩，《电力系统稳态分析》（第三版），中国电力出版社
