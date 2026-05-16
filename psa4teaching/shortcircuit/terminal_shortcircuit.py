"""
同步发电机机端三相短路电流计算
================================

基于论文 Wang et al. (2012) "From mathematical analysis to experimental calculation: 
teaching three-phase short-circuits of a synchronous generator" (IJEEE, Vol. 49, No. 4)

实现三种不同精度的计算方法：
1. 精确数学分析法（保留所有频率分量）
2. 三段式实验法（工程常用）
3. 工程简化法（最实用）

作者：psa4teaching 开发团队
日期：2026-05-16
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ========= 数据类 =========

@dataclass
class GeneratorSCParams:
    """发电机短路计算参数（标幺值系统，论文Table 1）"""
    r: float = 0.0045
    Xd: float = 0.95
    Xd_prime: float = 0.33
    Xd_doubleprime: float = 0.21
    Xq: float = 0.71
    Xq_prime: float = 0.71
    Xq_doubleprime: float = 0.22
    Td0_prime: float = 2800.0
    Td0_doubleprime: float = 30.0
    Tq0_prime: float = 0.0
    Tq0_doubleprime: float = 68.0
    U0: float = 1.0
    PF: float = 0.98
    omega: float = 2 * np.pi * 50
    Sn: float = 100.0
    Vn: float = 10.5

    def __post_init__(self):
        self.Td_prime = self.Td0_prime * self.Xd_prime / self.Xd
        self.Td_doubleprime = self.Td0_doubleprime * self.Xd_doubleprime / self.Xd_prime
        self.Tq_prime = self.Tq0_prime * self.Xq_prime / self.Xq if self.Tq0_prime > 0 else 0
        self.Tq_doubleprime = self.Tq0_doubleprime * self.Xq_doubleprime / self.Xq
        Xav = (self.Xd_doubleprime + self.Xq_doubleprime) / 2
        self.Ta = 1.0 / (self.r * self.omega / Xav)
        self.phi = np.arccos(abs(self.PF))
        if self.PF > 0:
            self.phi = -self.phi
        self.delta0 = np.arctan2(
            self.Xq * np.sin(self.phi),
            self.U0 + self.Xq * np.cos(self.phi)
        )


@dataclass
class TerminalSCResult:
    """机端三相短路计算结果"""
    t: np.ndarray
    ia: np.ndarray
    ib: np.ndarray
    ic: np.ndarray
    id_val: np.ndarray
    iq_val: np.ndarray
    iac_envelope: np.ndarray
    idc_component: np.ndarray
    fundamental: np.ndarray
    double_freq: np.ndarray
    method: str
    parameters: Dict

    def plot(self, ax=None, show_components=False, t_range=None, title=None, save_path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        mask = (self.t >= t_range[0]) & (self.t <= t_range[1]) if t_range else slice(None)
        ax.plot(self.t[mask], self.ia[mask], 'b-', label='ia (A相)', linewidth=1.5)
        ax.plot(self.t[mask], self.ib[mask], 'r--', label='ib (B相)', linewidth=1.5)
        ax.plot(self.t[mask], self.ic[mask], 'g-.', label='ic (C相)', linewidth=1.5)
        env = self.iac_envelope[mask]
        ax.plot(self.t[mask], env, 'k:', label='交流包络线', linewidth=2, alpha=0.7)
        ax.plot(self.t[mask], -env, 'k:', linewidth=2, alpha=0.7)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('电流 (p.u.)')
        ax.set_title(title or f'机端三相短路电流 - {self.method}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if show_components:
            fig2, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            axes[0].plot(self.t[mask], self.fundamental[mask], 'b-', label='基频分量')
            axes[0].set_ylabel('基频 (p.u.)')
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[1].plot(self.t[mask], self.idc_component[mask], 'r-', label='直流分量')
            axes[1].set_ylabel('直流 (p.u.)')
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
            if np.any(np.abs(self.double_freq[mask]) > 1e-6):
                axes[2].plot(self.t[mask], self.double_freq[mask], 'g-', label='倍频分量')
                axes[2].set_ylabel('倍频 (p.u.)')
                axes[2].legend(); axes[2].grid(True, alpha=0.3)
            axes[-1].set_xlabel('时间 (s)')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return ax

    def plot_envelope(self, ax=None, t_range=None, title=None, save_path=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        mask = (self.t >= t_range[0]) & (self.t <= t_range[1]) if t_range else slice(None)
        env = self.iac_envelope[mask]
        idc = self.idc_component[mask]
        ax.plot(self.t[mask], env, 'r-', label='上包络线', linewidth=2)
        ax.plot(self.t[mask], -env, 'b-', label='下包络线', linewidth=2)
        ax.plot(self.t[mask], idc, 'g--', label='直流分量', linewidth=1.5, alpha=0.7)
        ax.plot(self.t[mask], -idc, 'g--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('电流 (p.u.)')
        ax.set_title(title or f'短路电流包络线 - {self.method}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return ax


# ========= 内部辅助函数 =========

def _compute_time_constants(params):
    Td_prime = params.Td0_prime * params.Xd_prime / params.Xd
    Td_doubleprime = params.Td0_doubleprime * params.Xd_doubleprime / params.Xd_prime
    Tq_prime = params.Tq0_prime * params.Xq_prime / params.Xq if params.Tq0_prime > 0 else 0
    Tq_doubleprime = params.Tq0_doubleprime * params.Xq_doubleprime / params.Xq
    Xav = (params.Xd_doubleprime + params.Xq_doubleprime) / 2
    Ta = 1.0 / (params.r * params.omega / Xav)
    return Td_prime, Td_doubleprime, Tq_prime, Tq_doubleprime, Ta


def _compute_initial_values(params):
    """计算短路前的稳态量（论文第8页）"""
    phi = np.arccos(abs(params.PF))
    if params.PF > 0:
        phi = -phi
    Ud0 = 0.0
    Uq0 = params.U0
    Xavg = (params.Xd + params.Xq) / 2
    I_mag = params.U0 / Xavg
    delta0 = np.arctan2(params.Xq * I_mag * np.sin(phi),
                        params.U0 + params.Xq * I_mag * np.cos(phi))
    Id0 = I_mag * np.sin(delta0 - phi)
    Iq0 = I_mag * np.cos(delta0 - phi)
    Eq0 = Uq0 + Id0 * params.Xd
    Eq_prime0 = Uq0 + Id0 * params.Xd_prime
    Eq_doubleprime0 = Uq0 + Id0 * params.Xd_doubleprime
    Ed_doubleprime0 = -Ud0 + Iq0 * params.Xq_doubleprime
    return dict(Ud0=Ud0, Uq0=Uq0, Id0=Id0, Iq0=Iq0,
                Eq0=Eq0, Eq_prime0=Eq_prime0,
                Eq_doubleprime0=Eq_doubleprime0, Ed_doubleprime0=Ed_doubleprime0,
                delta0=delta0, phi=phi, I_mag=I_mag)


def _inv_park(id_val, iq_val, theta):
    """反Park变换: dq → abc"""
    ia = id_val * np.cos(theta) - iq_val * np.sin(theta)
    ib = id_val * np.cos(theta - 2*np.pi/3) - iq_val * np.sin(theta - 2*np.pi/3)
    ic = id_val * np.cos(theta + 2*np.pi/3) - iq_val * np.sin(theta + 2*np.pi/3)
    return ia, ib, ic


def _park(ia, ib, ic, theta):
    """Park变换: abc → dq"""
    c = np.cos(theta)
    s = np.sin(theta)
    c120 = np.cos(theta - 2*np.pi/3)
    s120 = np.sin(theta - 2*np.pi/3)
    c240 = np.cos(theta + 2*np.pi/3)
    s240 = np.sin(theta + 2*np.pi/3)
    id_val = (2/3) * (ia*c + ib*c120 + ic*c240)
    iq_val = -(2/3) * (ia*s + ib*s120 + ic*s240)
    return id_val, iq_val


# ========= 三种计算方法 =========

def calculate_terminal_shortcircuit_mathematical(
    params, t_end=10.0, dt=0.001, theta0=0.0, verbose=False
):
    """
    精确数学分析法（方法1，论文ia₁(t)）
    保留所有频率分量（基频+直流+倍频）。
    """
    if verbose:
        print("=== 精确数学分析法（ia₁(t)）===")
    iv = _compute_initial_values(params)
    d0 = iv['delta0']
    Eq0, Eqp0, Eqpp0 = iv['Eq0'], iv['Eq_prime0'], iv['Eq_doubleprime0']
    Tdp, Tdpp, Tqp, Tqpp, Ta = _compute_time_constants(params)
    if verbose:
        print(f"Eq0={Eq0:.4f}, Eq'0={Eqp0:.4f}, Eq\"0={Eqpp0:.4f}")
        print(f"Td'={Tdp:.2f}s, Td\"={Tdpp:.4f}s, Ta={Ta:.2f}s, δ0={np.degrees(d0):.2f}°")

    t = np.arange(0, t_end + dt, dt)
    theta = params.omega * t + theta0

    # 基频交流分量（稳态+暂态+次暂态叠加）
    i_ac = (Eq0 / params.Xd
            + (Eqp0 / params.Xd_prime - Eq0 / params.Xd) * np.exp(-t / Tdp)
            + (Eqpp0 / params.Xd_doubleprime - Eqp0 / params.Xd_prime) * np.exp(-t / Tdpp))

    # dq轴电流
    id_val = i_ac * np.cos(theta - d0)
    iq_val = i_ac * np.sin(theta - d0)

    # 直流衰减（定子时间常数Ta）
    I0 = params.U0 / params.Xd_doubleprime
    id_val -= I0 * np.exp(-t / Ta) * np.cos(theta0)
    iq_val += I0 * np.exp(-t / Ta) * np.sin(theta0)

    # 倍频分量（Xd ≠ Xq 引起）
    df_amp = params.U0 * (1/params.Xd_doubleprime - 1/params.Xq_doubleprime) / 2
    id_val += df_amp * np.exp(-t / Ta) * np.cos(2*theta - d0)
    iq_val += df_amp * np.exp(-t / Ta) * np.sin(2*theta - d0)

    ia, ib, ic = _inv_park(id_val, iq_val, theta)

    if verbose:
        print(f"ia_max={np.max(np.abs(ia)):.4f}, ia_steady={np.abs(ia[-1]):.4f}")

    return TerminalSCResult(
        t=t, ia=ia, ib=ib, ic=ic,
        id_val=id_val, iq_val=iq_val,
        iac_envelope=np.abs(i_ac),
        idc_component=I0 * np.exp(-t / Ta),
        fundamental=np.abs(i_ac),
        double_freq=df_amp * np.exp(-t / Ta),
        method='mathematical',
        parameters=dict(Td_prime=Tdp, Td_doubleprime=Tdpp, Ta=Ta,
                       Eq0=Eq0, delta0=d0, theta0=theta0)
    )


def calculate_terminal_shortcircuit_experimental(
    params, t_end=10.0, dt=0.001, theta0=0.0, verbose=False
):
    """
    三段式实验法（方法2，论文ia₂(t)）
    与精确法结构相同但忽略倍频分量。
    """
    if verbose:
        print("=== 三段式实验法（ia₂(t)）===")
    iv = _compute_initial_values(params)
    d0 = iv['delta0']
    Eq0, Eqp0, Eqpp0 = iv['Eq0'], iv['Eq_prime0'], iv['Eq_doubleprime0']
    Tdp, Tdpp, Tqp, Tqpp, Ta = _compute_time_constants(params)
    if verbose:
        print(f"Td'={Tdp:.2f}s, Td\"={Tdpp:.4f}s, Ta={Ta:.2f}s")

    t = np.arange(0, t_end + dt, dt)
    theta = params.omega * t + theta0

    i_ac = (Eq0 / params.Xd
            + (Eqp0 / params.Xd_prime - Eq0 / params.Xd) * np.exp(-t / Tdp)
            + (Eqpp0 / params.Xd_doubleprime - Eqp0 / params.Xd_prime) * np.exp(-t / Tdpp))

    id_val = i_ac * np.cos(theta - d0)
    iq_val = i_ac * np.sin(theta - d0)

    I0 = params.U0 / params.Xd_doubleprime
    id_val -= I0 * np.exp(-t / Ta) * np.cos(theta0)
    iq_val += I0 * np.exp(-t / Ta) * np.sin(theta0)

    ia, ib, ic = _inv_park(id_val, iq_val, theta)

    if verbose:
        print(f"ia_max={np.max(np.abs(ia)):.4f}")

    return TerminalSCResult(
        t=t, ia=ia, ib=ib, ic=ic,
        id_val=id_val, iq_val=iq_val,
        iac_envelope=np.abs(i_ac),
        idc_component=I0 * np.exp(-t / Ta),
        fundamental=np.abs(i_ac),
        double_freq=np.zeros_like(t),
        method='experimental',
        parameters=dict(Td_prime=Tdp, Td_doubleprime=Tdpp, Ta=Ta,
                       Eq0=Eq0, delta0=d0, theta0=theta0)
    )


def calculate_terminal_shortcircuit_simplified(
    params, t_end=10.0, dt=0.001, theta0=0.0, verbose=False
):
    """
    工程简化法（方法3，论文ia₃(t)）
    假设X"d=X"q，消除倍频，包络线公式最简。
    """
    if verbose:
        print("=== 工程简化法（ia₃(t)）===")
    iv = _compute_initial_values(params)
    d0 = iv['delta0']
    Eq0 = iv['Eq0']
    Tdp, Tdpp, _, _, Ta = _compute_time_constants(params)
    if verbose:
        print(f"Td'={Tdp:.2f}s, Td\"={Tdpp:.4f}s, Ta={Ta:.2f}s")

    t = np.arange(0, t_end + dt, dt)
    theta = params.omega * t + theta0

    # 交流包络线（论文公式14）
    env = Eq0 * (1/params.Xd
                 + (1/params.Xd_prime - 1/params.Xd) * np.exp(-t / Tdp)
                 + (1/params.Xd_doubleprime - 1/params.Xd_prime) * np.exp(-t / Tdpp))

    # 直流分量振幅
    dc = np.sqrt(2) * params.U0 / params.Xd_doubleprime * np.exp(-t / Ta)

    ia = env * np.cos(theta) + dc * np.cos(d0)
    ib = env * np.cos(theta - 2*np.pi/3) + dc * np.cos(d0 - 2*np.pi/3)
    ic = env * np.cos(theta + 2*np.pi/3) + dc * np.cos(d0 + 2*np.pi/3)

    id_val, iq_val = _park(ia, ib, ic, theta)

    if verbose:
        print(f"env[0]={env[0]:.4f}, dc[0]={dc[0]:.4f}")

    return TerminalSCResult(
        t=t, ia=ia, ib=ib, ic=ic,
        id_val=id_val, iq_val=iq_val,
        iac_envelope=env,
        idc_component=dc,
        fundamental=env,
        double_freq=np.zeros_like(t),
        method='simplified',
        parameters=dict(Td_prime=Tdp, Td_doubleprime=Tdpp, Ta=Ta,
                       Eq0=Eq0, delta0=d0, theta0=theta0)
    )


# ========= 绘图对比函数 =========

def plot_comparison(results, t_range=(0, 5), save_path=None, figsize=(12, 6)):
    """绘制多种方法的|ia|对比曲线（论文Fig.7/9）"""
    fig, ax = plt.subplots(figsize=figsize)
    style = {'mathematical': ('b-', 'ia₁(t) 精确数学法', 2.0),
             'experimental': ('r--', 'ia₂(t) 三段式实验法', 1.5),
             'simplified': ('g:', 'ia₃(t) 工程简化法', 1.5)}
    for r in results:
        mask = (r.t >= t_range[0]) & (r.t <= t_range[1])
        c, lbl, lw = style.get(r.method, ('k-', r.method, 1))
        ax.plot(r.t[mask], np.abs(r.ia[mask]), c, label=lbl, linewidth=lw)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('|ia(t)| (p.u.)')
    ax.set_title('三种方法短路电流对比')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(t_range)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    return fig


def plot_components_comparison(results, t_range=(0, 1), save_path=None):
    """绘制交流/直流分量对比（论文Fig.8）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    style = {'mathematical': 'b-', 'experimental': 'r--', 'simplified': 'g:'}
    labels = {'mathematical': 'ia₁(t)', 'experimental': 'ia₂(t)', 'simplified': 'ia₃(t)'}
    for r in results:
        mask = (r.t >= t_range[0]) & (r.t <= t_range[1])
        c = style.get(r.method, 'k-')
        lbl = labels.get(r.method, r.method)
        ax1.plot(r.t[mask], r.iac_envelope[mask], c, label=lbl, linewidth=2)
        ax2.plot(r.t[mask], r.idc_component[mask], c, label=lbl, linewidth=2)
    ax1.set_xlabel('时间 (s)'); ax1.set_ylabel('交流包络 (p.u.)')
    ax1.set_title('交流分量对比'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('时间 (s)'); ax2.set_ylabel('直流分量 (p.u.)')
    ax2.set_title('直流分量对比'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
    return fig
