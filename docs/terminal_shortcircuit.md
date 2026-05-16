# 机端三相短路电流计算 — 程序设计文档

> 基于 Wang et al. (2012) "From mathematical analysis to experimental calculation:  
> teaching three-phase short-circuits of a synchronous generator"  
> International Journal of Electrical Engineering Education, Vol.49, No.4

---

## 1. 功能概述

本模块实现同步发电机机端三相短路电流的计算，支持三种不同精度的计算方法：

| 方法 | 函数名 | 精度 | 计算复杂度 | 适用场景 |
|------|--------|------|------------|----------|
| **精确数学分析法** | `calculate_terminal_shortcircuit_mathematical()` | 最高 | 高 | 理论研究、精确分析 |
| **三段式实验法** | `calculate_terminal_shortcircuit_experimental()` | 中等 | 中 | 教学演示、工程分析 |
| **工程简化法** | `calculate_terminal_shortcircuit_simplified()` | 实用 | 低 | 规划设计、速算 |

---

## 2. 数学模型

### 2.1 发电机机端短路的物理过程

机端三相短路发生时，短路电流包含三个过程：
1. **次暂态过程**（0~0.06s）：Xd″、Xq″ 起作用，电流最大
2. **暂态过程**（0.06~1s）：Xd′、Xq′ 起作用，电流衰减
3. **稳态过程**（>1s）：Xd、Xq 起作用，电流趋于稳定

### 2.2 精确数学分析法（对应论文 ia₁(t)）

#### Park 变换下的发电机方程

磁链方程：
$$
\begin{bmatrix}
\psi_d \\ \psi_q \\ \psi_f \\ \psi_D \\ \psi_Q
\end{bmatrix}
=
\begin{bmatrix}
X_d & 0 & X_{ad} & X_{ad} & 0 \\
0 & X_q & 0 & 0 & X_{aq} \\
X_{ad} & 0 & X_F & X_{ad} & 0 \\
X_{ad} & 0 & X_{ad} & X_D & 0 \\
0 & X_{aq} & 0 & 0 & X_Q
\end{bmatrix}
\begin{bmatrix}
i_d \\ i_q \\ i_f \\ i_D \\ i_Q
\end{bmatrix}
$$

电压方程（Laplace 域）：
$$
\begin{bmatrix}
U_d(s) \\ U_q(s) \\ 0 \\ 0 \\ 0
\end{bmatrix}
=
\begin{bmatrix}
r + sX_d(s) & -\omega X_q(s) & \cdots \\
\omega X_d(s) & r + sX_q(s) & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
\begin{bmatrix}
i_d(s) \\ i_q(s) \\ \vdots
\end{bmatrix}
$$

其中运算电抗：
$$X_d(s) = X_d \frac{(1+sT'_{d0})(1+sT''_{d0})}{(1+sT'_d)(1+sT''_d)}$$

$$X_q(s) = X_q \frac{(1+sT'_{q0})(1+sT''_{q0})}{(1+sT'_q)(1+sT''_q)}$$

#### 故障分量求解

故障分量电流（Laplace 域）：
$$\Delta i_d(s) = \frac{X_q(s)U_{d0} + X(s)U_{q0}}{rX(s) + sX_d(s)X_q(s) + r^2}$$

$$\Delta i_q(s) = \frac{-X_d(s)U_{q0} + X(s)U_{d0}}{rX(s) + sX_d(s)X_q(s) + r^2}$$

其中 $X(s) = X_d(s)X_q(s) - X^2(s)$，$X(s) = X_{ad}X_D/(X_{ad}+X_D)$

#### 分母多项式

对论文参数，分母 $M(s)$ 为 6 阶多项式：
$$M(s) = 42793.6865s^6 + 6159.1388s^5 + 43062.3642s^4 + 4370.1648s^3 + 113.1738s^2 + 0.1094s$$

求根得到 6 个特征值，再通过部分分式展开得到时域表达式。

#### 反 Park 变换

$$i_a(t) = i_d(t)\cos(\theta_0 + \omega t) - i_q(t)\sin(\theta_0 + \omega t)$$

---

### 2.3 三段式实验法（对应论文 ia₂(t)）

#### 基本假设

1. 将短路过程分为三个**独立连续**的阶段
2. 各阶段使用不同的电抗值：

| 阶段 | 时间范围 | d轴电抗 | q轴电抗 | 考虑电阻 |
|------|----------|---------|---------|----------|
| 次暂态 | 0~0.06s | X″d | X″q | 否 |
| 暂态 | 0.06~1s | X′d | X′q(≈Xq) | 否 |
| 稳态 | >1s | Xd | Xq | 否 |

#### 电流表达式

$$
\begin{aligned}
i_a(t) = &\frac{E_0}{X_d} + \left(\frac{E'_q}{X'_d} - \frac{E_0}{X_d}\right)e^{-t/T'_d} + \left(\frac{E''_q}{X''_d} - \frac{E'_q}{X'_d}\right)e^{-t/T''_d} \cos(\theta_0 + \omega t) \\
&+ \frac{U_0}{2}\left(\frac{1}{X''_d} - \frac{1}{X'_d}\right)e^{-t/T_a}\cos(\theta_0 + \omega t + \delta_0) \\
&+ \frac{U_0}{2}\left(\frac{1}{X''_d} - \frac{1}{X'_d}\right)e^{-t/T_a}\cos(\theta_0 + \omega t - \delta_0)
\end{aligned}
$$

其中时间常数：
$$T'_d = T'_{d0}\frac{X'_d}{X_d}, \quad T''_d = T''_{d0}\frac{X''_d}{X'_d}, \quad T_a = \frac{1}{r\omega\frac{2}{X''_d + X''_q}}$$

---

### 2.4 工程简化法（对应论文 ia₃(t)）

#### 进一步假设

1. 设 $X″_d = X″_q$（忽略 dq 轴差异）
2. 忽略倍频分量（设 $\delta_0 = 0$）
3. 电压标幺值 $E_0 ≈ 1.0, U_0 ≈ 1.0$

#### 振幅包络线公式

$$
i_{ac}(t) = \left[\frac{1}{X_d} + \left(\frac{1}{X'_d} - \frac{1}{X_d}\right)e^{-t/T'_d} + \left(\frac{1}{X''_d} - \frac{1}{X'_d}\right)e^{-t/T''_d}\right] \times \frac{\sqrt{2}}{2}
$$

$$
i_{dc}(t) = -\frac{\sqrt{2}}{X''_d} e^{-t/T_a}
$$

有效值：
$$i(t) = \sqrt{i_{ac}^2(t) + i_{dc}^2(t)}$$

---

## 3. 接口文档

### 3.1 参数类

```python
@dataclass
class GeneratorSCParams:
    """发电机短路计算参数（论文 Table 1）"""
    # 电阻与电抗
    r: float = 0.0045           # 定子电阻 (p.u.)
    Xd: float = 0.95           # d轴同步电抗
    Xd_prime: float = 0.33     # d轴暂态电抗 X'd
    Xd_doubleprime: float = 0.21  # d轴次暂态电抗 X"d
    Xq: float = 0.71           # q轴同步电抗
    Xq_prime: float = 0.71     # q轴暂态电抗 X'q (≈Xq)
    Xq_doubleprime: float = 0.22  # q轴次暂态电抗 X"q
    
    # 时间常数
    Td0_prime: float = 2800.0    # d轴暂态开路时间常数 T'd0 (s)
    Td0_doubleprime: float = 30.0  # d轴次暂态开路时间常数 T"d0 (s)
    Tq0_prime: float = 0.0       # q轴暂态开路时间常数 T'q0
    Tq0_doubleprime: float = 68.0   # q轴次暂态开路时间常数 T"q0 (s)
    
    # 运行参数
    U0: float = 1.0            # 额定电压 (p.u.)
    PF: float = 0.98            # 功率因数（超前为正）
    omega: float = 314.159       # 角频率 2π×50 Hz (rad/s)
```

### 3.2 结果类

```python
@dataclass
class TerminalSCResult:
    """机端三相短路计算结果"""
    t: np.ndarray                # 时间向量 (s)
    ia: np.ndarray                # a相电流 (p.u.)
    ib: np.ndarray                # b相电流
    ic: np.ndarray                # c相电流
    id: np.ndarray                # d轴电流
    iq: np.ndarray                # q轴电流
    
    # 分量分解（用于绘图）
    fundamental: np.ndarray       # 基频分量
    dc_component: np.ndarray    # 直流分量
    double_freq: np.ndarray     # 倍频分量（仅精确方法有）
    
    # 包络线
    ac_envelope: np.ndarray      # 交流包络线 |i_ac(t)|
    dc_envelope: np.ndarray      # 直流包络线 |i_dc(t)|
    
    method: str                  # "mathematical" | "experimental" | "simplified"
    params: GeneratorSCParams   # 使用的参数
    
    def plot(self, ax=None, show_components=False, t_range=None, title=None):
        """绘制短路电流曲线
        
        Args:
            ax: matplotlib 轴对象（可选，用于子图）
            show_components: 是否显示各分量
            t_range: 显示的时间范围，如 (0, 5)
            title: 图标题
        """
    
    def get_envelope(self):
        """获取电流包络线（有效值）"""
```

### 3.3 核心函数

```python
def calculate_terminal_shortcircuit_mathematical(
    params: GeneratorSCParams,
    t_end: float = 10.0,
    dt: float = 0.001,
    theta0: float = 0.0,
    verbose: bool = False
) -> TerminalSCResult:
    """
    精确数学分析法（论文 ia₁(t)）
    
    通过 Laplace 变换 + 部分分式展开求解精确的短路电流表达式。
    
    Args:
        params: 发电机参数
        t_end: 仿真时长 (s)
        dt: 时间步长 (s)
        theta0: 初始转子角 (rad)
        verbose: 是否打印中间过程
        
    Returns:
        TerminalSCResult: 包含三相电流和分量分解
        
    论文对应：Section "Mathematical expression"（第7~9页）
    """
```

```python
def calculate_terminal_shortcircuit_experimental(
    params: GeneratorSCParams,
    t_end: float = 10.0,
    dt: float = 0.001,
    theta0: float = 0.0,
    verbose: bool = False
) -> TerminalSCResult:
    """
    三段式实验法（论文 ia₂(t)）
    
    将短路过程分为次暂态、暂态、稳态三个阶段，
    每个阶段使用不同的电抗值和时间常数。
    
    论文对应：Section "Experimental expression"（第11~13页）
    """
```

```python
def calculate_terminal_shortcircuit_simplified(
    params: GeneratorSCParams,
    t_end: float = 10.0,
    dt: float = 0.001,
    theta0: float = 0.0,
    verbose: bool = False
) -> TerminalSCResult:
    """
    工程简化法（论文 ia₃(t)）
    
    假设 X″d = X″q，忽略倍频分量，适用于工程速算。
    
    论文对应：Section "Experimental expression with further simplifications"（第13页）
    """
```

```python
def plot_comparison(
    results: List[TerminalSCResult],
    t_range: Tuple[float, float] = (0, 5),
    save_path: str = None
) -> None:
    """
    绘制多种方法的对比曲线（对应论文 Fig.7/8/9）
    
    Args:
        results: 多种方法的结果列表
        t_range: 显示的时间范围
        save_path: 图片保存路径（可选）
    """
```

---

## 4. 论文参数验证

### 4.1 算例参数（论文 Table 1）

```python
# 论文标准参数
paper_params = GeneratorSCParams(
    r=0.0045,
    Xd=0.95, Xd_prime=0.33, Xd_doubleprime=0.21,
    Xq=0.71, Xq_prime=0.71, Xq_doubleprime=0.22,
    Td0_prime=2800.0, Td0_doubleprime=30.0,
    Tq0_prime=0.0, Tq0_doubleprime=68.0,
    U0=1.0, PF=0.98, omega=2*np.pi*50
)
```

### 4.2 验证点（论文第8页）

| 量 | 论文值 | 计算值 | 误差 |
|-----|--------|--------|------|
| Uq(0) | 0.8550 | — | — |
| Ud(0) | 0.5186 | — | — |
| Iq(0) | 0.7347 | — | — |
| Id(0) | 0.6784 | — | — |
| 特征根 s₁ | 0 | — | — |
| 特征根 s₂ | -0.0010 | — | — |
| 特征根 s₃ | -0.0475 | — | — |
| 特征根 s₄ | -0.0535 | — | — |
| 特征根 s₅,₆ | -0.0209 ± j0.9995 | — | — |

### 4.3 论文曲线对比

| 图号 | 内容 | 对应函数 |
|------|------|----------|
| Fig.3 | ia₁(t) 各分量 | `mathematical()` → `plot(show_components=True)` |
| Fig.4 | ia₂(t) 各分量 | `experimental()` → `plot(show_components=True)` |
| Fig.5 | ia₃(t) 有效值曲线 | `simplified()` → `get_envelope()` |
| Fig.7 | ia₁ vs ia₂ 总电流对比 | `plot_comparison([r1, r2])` |
| Fig.8 | 三种方法交流/直流分量对比 | `plot_comparison([r1, r2, r3])` |
| Fig.9 | 三种方法总电流对比 | `plot_comparison([r1, r2, r3])` |

---

## 5. 使用示例

### 5.1 基本使用

```python
import numpy as np
from psa4teaching.shortcircuit.terminal_shortcircuit import (
    GeneratorSCParams,
    calculate_terminal_shortcircuit_mathematical,
    calculate_terminal_shortcircuit_experimental,
    calculate_terminal_shortcircuit_simplified,
    plot_comparison,
)

# 使用论文参数
params = GeneratorSCParams()

# 方法1：精确数学法
result_math = calculate_terminal_shortcircuit_mathematical(params, t_end=10.0, dt=0.001)
result_math.plot(title="Fig.3/ia1(t) - 精确数学法")

# 方法2：三段式实验法
result_exp = calculate_terminal_shortcircuit_experimental(params, t_end=10.0)
result_exp.plot(title="Fig.4/ia2(t) - 三段式实验法")

# 方法3：工程简化法
result_simp = calculate_terminal_shortcircuit_simplified(params, t_end=10.0)
result_simp.plot(title="Fig.5/ia3(t) - 工程简化法")
```

### 5.2 三种方法对比（复现论文 Fig.7/8/9）

```python
# 计算三种方法
params = GeneratorSCParams()
r1 = calculate_terminal_shortcircuit_mathematical(params)
r2 = calculate_terminal_shortcircuit_experimental(params)
r3 = calculate_terminal_shortcircuit_simplified(params)

# 绘制对比（Fig.7 - 总电流对比）
plot_comparison([r1, r2], t_range=(0, 5), save_path="fig7_comparison.png")

# 绘制对比（Fig.8 - 分量对比）
plot_comparison([r1, r2, r3], t_range=(0, 0.5), save_path="fig8_components.png")

# 绘制对比（Fig.9 - 总电流对比）
plot_comparison([r1, r2, r3], t_range=(0, 10), save_path="fig9_total.png")
```

### 5.3 自定义参数

```python
# 修改发电机参数
custom_params = GeneratorSCParams(
    r=0.005,
    Xd=1.2, Xd_prime=0.35, Xd_doubleprime=0.25,
    Xq=1.1, Xq_doubleprime=0.28,
    Td0_prime=3000.0, Td0_doubleprime=35.0,
    U0=1.05, PF=0.95
)

result = calculate_terminal_shortcircuit_experimental(custom_params)
result.plot(title="自定义参数 - 三段式实验法", t_range=(0, 3))
```

---

## 6. 曲线说明

### 6.1 论文曲线特征

| 曲线 | 特征 | 物理意义 |
|------|------|----------|
| **基频分量** | 持续存在，幅值逐渐衰减至稳态 | 反映 Xd→X′d→X″d 的衰减 |
| **直流分量** | 起始幅值大，快速衰减（Ta ≈ 48s） | 反映定子暂态过程 |
| **倍频分量** | 仅在精确方法中存在，很快衰减 | 由 dq 轴参数不对称引起 |

### 6.2 误差分析（论文 Table 2）

| 阶段 | ia₂(t) 平均误差 | 相对误差 |
|------|------------------|----------|
| 次暂态段 (0~0.06s) | 0.0505 p.u. | 0.88% |
| 暂态段 (0.06~1s) | 0.0690 p.u. | 2.06% |
| 稳态段 (>1s) | 0.0461 p.u. | 2.27% |

---

## 7. 实现注意事项

1. **Laplace 求解**：精确方法中分母 M(s) 是 6 阶多项式，用 `np.roots()` 求根
2. **部分分式展开**：对实部相同的共轭复根要合并为实数形式
3. **反 Park 变换**：注意 θ = ωt + θ₀，三相相差 120°
4. **时间常数计算**：注意区分开路时间常数（T₀）和短路时间常数（T）
5. **绘图**：使用 `matplotlib`，并设置论文风格（黑白打印友好）

---

*文档版本：v1.0 | 日期：2026-05-16 | 基于 Wang et al. (2012)*
