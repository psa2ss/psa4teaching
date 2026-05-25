"""
等面积准则 + 摇摆曲线 交互展示工具（v3 - 几何优先）
================================================

基于《电力系统暂态分析》教材，展示单机无穷大系统暂态稳定的
等面积准则机理。

核心设计：
1. 等面积图：基于几何关系直接计算，不依赖复杂仿真参数
2. 摇摆曲线：用时域仿真生成，可选是否显示
3. 横轴：转子位置角 δ（°）
4. 纵轴正向（第一象限）：有功 P（p.u.）
5. 纵轴负向（第四象限）：时间 t（s）

作者：psa4teaching 开发团队
日期：2026-05-16
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 默认无 GUI
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass, field
from typing import Optional

# 中文字体
try:
    matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


@dataclass
class EqualAreaParams:
    """等面积准则参数（物理意义优先）"""
    Pm: float = 1.0           # 机械功率 (p.u.)
    E: float = 1.2             # 发电机暂态电势 (p.u.)
    V: float = 1.0             # 无穷大母线电压 (p.u.)
    X_pre: float = 0.4          # 故障前总电抗 (p.u.)
    X_post: float = 0.8         # 故障后总电抗 (p.u.)
    H: float = 5.0              # 惯性常数 (s)
    D: float = 0.0              # 阻尼系数
    t_clear: float = 0.20       # 故障切除时间 (s)
    t_end: float = 3.0          # 仿真时长 (s)
    show_swing: bool = True      # 是否显示摇摆曲线


class EqualAreaInteractive:
    """
    等面积准则交互展示工具
    
    用法：
        tool = EqualAreaInteractive()
        tool.generate_figure('demo.png')    # 生成静态图
        tool.show()                         # 交互界面（需 GUI）
    """

    def __init__(self, params: Optional[EqualAreaParams] = None):
        self.params = params or EqualAreaParams()
        p = self.params

        # 功率系数（三段工况）
        self.Pmax_pre = p.E * p.V / p.X_pre
        self.Pmax_post = p.E * p.V / p.X_post
        self.Pmax_fault = 0  # 三相短路，Pe ≈ 0

        # 关键角度（弧度）
        self.delta0 = np.arcsin(min(p.Pm / self.Pmax_pre, 1.0))
        self.delta_s = np.arcsin(min(p.Pm / self.Pmax_post, 1.0)) if self.Pmax_post >= p.Pm else None
        self.delta_max_stable = np.pi - self.delta_s if self.delta_s else np.pi

        # 临界切除角
        self.delta_crit = self._calc_critical_angle()

        # 仿真结果
        self.result = None
        self.t_clear = p.t_clear

    def _calc_critical_angle(self) -> float:
        """二分法求临界切除角 δ_crit"""
        Pm = self.params.Pm
        d0 = self.delta0
        dm = self.delta_max_stable

        def diff(dc):
            # 加速面积 A_acc = Pm * (δc - δ0)
            A_acc = Pm * (dc - d0)
            # 减速面积 A_dec = Pmax_post * [cos(δc) - cos(δ_max)] - Pm * (δ_max - δc)
            A_dec = self.Pmax_post * (np.cos(dc) - np.cos(dm)) - Pm * (dm - dc)
            return A_acc - A_dec

        lo, hi = d0 + 1e-6, dm - 1e-6
        if lo >= hi:
            return dm
        for _ in range(60):
            mid = (lo + hi) / 2
            if diff(mid) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def _acc_area(self, dc: float) -> float:
        """加速面积 A_acc = Pm × (δc - δ0)"""
        return self.params.Pm * (dc - self.delta0)

    def _dec_area(self, dc: float) -> float:
        """减速面积 A_dec = Pmax_post × [cos(δc) - cos(δ_max)] - Pm × (δ_max - δc)"""
        dm = self.delta_max_stable
        return self.Pmax_post * (np.cos(dc) - np.cos(dm)) - self.params.Pm * (dm - dc)

    def _run_simulation(self):
        """运行时域仿真（导入来自 psa4teaching）"""
        try:
            from psa4teaching.stability.transient import simulate_single_machine_infinite_bus_classic
            self.result = simulate_single_machine_infinite_bus_classic(
                E_prime=self.params.E, V_infinity=self.params.V,
                X_total=self.params.X_pre,
                X_total_fault=10.0,  # 近似三相短路
                X_total_post=self.params.X_post,
                H=self.params.H, D=self.params.D,
                Pm=self.params.Pm, delta_0=self.delta0,
                fault_time=0.0, fault_clearing_time=self.t_clear,
                t_end=self.params.t_end, dt=0.002,
                stability_limit=500.0,
            )
        except Exception as e:
            print(f"仿真出错: {e}")
            self.result = None

    # ─────── 绘图：等面积图（核心） ───────

    def _draw_equal_area_axes(self, ax, t_clear: float = None):
        """在给定 axes 上绘制等面积图"""
        p = self.params
        ax.set_title('等面积准则 — 功角特性图', fontsize=13, fontweight='bold', pad=10)

        # 1. 三段 P-δ 曲线
        d = np.linspace(0, np.pi, 500)
        d_deg = np.degrees(d)

        ax.plot(d_deg, self.Pmax_pre * np.sin(d), 'b-', lw=2.5,
                label=r'故障前 $P_{\max}=' + f'{self.Pmax_pre:.2f}' + r'$')
        ax.plot(d_deg, np.zeros_like(d), 'r-', lw=2.5,
                label=r'故障中 $P=0$ (三相短路)')
        ax.plot(d_deg, self.Pmax_post * np.sin(d), 'g-', lw=2.5,
                label=r"故障后 $P'_{\max}=" + f'{self.Pmax_post:.2f}' + r'$')

        # 2. 机械功率线
        ax.axhline(p.Pm, color='k', ls='--', lw=1.5, label=f'$P_m={p.Pm}$')

        # 3. 关键点
        ax.plot(np.degrees(self.delta0), p.Pm, 'ko', ms=8, zorder=10,
                label=r'$\delta_0=' + f'{np.degrees(self.delta0):.1f}' + r'^\circ$')

        if self.delta_s:
            ax.plot(np.degrees(self.delta_s), p.Pm, 'gv', ms=7, zorder=10,
                    label=r'$\delta_s=' + f'{np.degrees(self.delta_s):.1f}' + r'^\circ$')
            ax.plot(np.degrees(self.delta_max_stable), p.Pm, 'rv', ms=7, zorder=10,
                    label=r'$\delta_{\max}=' + f'{np.degrees(self.delta_max_stable):.1f}' + r'^\circ$')

        # 4. 临界切除角线
        dc_crit_deg = np.degrees(self.delta_crit)
        ax.axvline(dc_crit_deg, color='gray', ls=':', lw=1, alpha=0.5,
                    label=r'$\delta_{crit}=' + f'{dc_crit_deg:.1f}' + r'^\circ$')

        # 5. 当前切除角 + 面积填充
        if t_clear is not None:
            self.t_clear = t_clear
            dc = self._find_clearing_angle_from_time(t_clear)
            if dc and dc > self.delta0 + 0.01:
                dc_deg = np.degrees(dc)
                ax.plot([dc_deg], [p.Pm], 'o', color='magenta', ms=12, zorder=15,
                        label=r'$\delta_c=' + f'{dc_deg:.1f}' + r'^\circ$')
                ax.axvline(dc_deg, color='magenta', ls='-', lw=1.5, alpha=0.4)

                # 加速面积
                da = np.linspace(self.delta0, dc, 300)
                ax.fill_between(np.degrees(da), 0, p.Pm,
                                color='#ffbb78', alpha=0.6, label='$A_{acc}$')
                A_acc = self._acc_area(dc)

                # 减速面积
                dd = np.linspace(dc, self.delta_max_stable, 300)
                P_upper = np.maximum(self.Pmax_post * np.sin(dd), p.Pm)
                ax.fill_between(np.degrees(dd), p.Pm, P_upper,
                                color='#98df8a', alpha=0.6, label='$A_{dec}$')
                A_dec = self._dec_area(dc)

                # 面积数值
                mid_acc = np.degrees((self.delta0 + dc) / 2)
                ax.text(mid_acc, p.Pm * 0.4, f'$A_{{acc}}$\n{A_acc:.3f}',
                        ha='center', fontsize=11, color='darkorange',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

                mid_dec = np.degrees((dc + self.delta_s) / 2) if self.delta_s else dc_crit_deg
                ax.text(mid_dec, p.Pm + self.Pmax_post * 0.12, f'$A_{{dec}}$\n{A_dec:.3f}',
                        ha='center', fontsize=11, color='green',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

                # 稳定性判断
                ratio = A_acc / A_dec * 100 if A_dec > 0 else float('inf')
                if ratio < 100:
                    status = f'稳定 (A_acc/A_dec = {ratio:.1f}% < 100%)'
                    color = 'green'
                else:
                    status = f'失稳 (A_acc/A_dec = {ratio:.1f}% ≥ 100%)'
                    color = 'red'
                ax.text(0.98, 0.92, status, transform=ax.transAxes,
                        ha='right', va='top', fontsize=11, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.9))

        ax.set_ylabel('有功功率 P (p.u.)', fontsize=12)
        ax.set_xlim(0, 180)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8.5, ncol=2, framealpha=0.9)

    def _find_clearing_angle_from_time(self, t_clear: float) -> float:
        """根据 t_clear 反推 δc（简化：从等面积准则几何关系）"""
        # 从加速面积 = Pm * (δc - δ0) 和加速时间 t_clear
        # 简化：δc = δ0 + (Pm / (2H)) * t_clear²（等加速度）
        # 但更准确的是从仿真结果取
        if self.result is not None and len(self.result.time) > 0:
            idx = np.argmin(np.abs(self.result.time - t_clear))
            return np.radians(self.result.delta[idx])
        # 无仿真时：用简化公式
        a = self.params.Pm / (2 * self.params.H)  # 角加速度 p.u.
        delta_c_rad = self.delta0 + 0.5 * a * t_clear ** 2
        return min(delta_c_rad, self.delta_max_stable)

    # ─────── 绘图：摇摆曲线 ───────

    def _draw_swing_axes(self, ax):
        """绘制摇摆曲线"""
        if not self.params.show_swing:
            ax.text(0.5, 0.5, '（未启用摇摆曲线）', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='gray')
            ax.set_xlabel('δ (°)', fontsize=11)
            ax.set_ylabel('t (s)', fontsize=11)
            return

        if self.result is None:
            self._run_simulation()

        if self.result is None or len(self.result.delta) == 0:
            ax.text(0.5, 0.5, '仿真失败', transform=ax.transAxes,
                    ha='center', va='center', fontsize=11, color='red')
            return

        delta_deg = self.result.delta  # 度数
        t = self.result.time

        color = 'green' if self.result.stable else 'red'
        ax.plot(delta_deg, t, color=color, lw=2.2, label=r'$\delta(t)$')

        # 标记点
        ax.plot(delta_deg[0], t[0], 'ko', ms=6, label=r'$\delta_0$')
        idx_c = np.argmin(np.abs(t - self.t_clear))
        ax.plot(delta_deg[idx_c], self.t_clear, 'o', color='magenta', ms=10, zorder=10,
                label=r'$\delta_c$ @ t=' + f'{self.t_clear:.2f}s')

        # 临界角线
        ax.axvline(np.degrees(self.delta_crit), color='red', ls='--', lw=1, alpha=0.5,
                    label=r'$\delta_{crit}$')

        # 标题
        status = '稳定 ✓' if self.result.stable else '失稳 ✗'
        ax.set_title(f'摇摆曲线 δ(t) — {status}',
                     color=color, fontweight='bold', fontsize=13, pad=10)

        ax.set_xlabel('转子位置角 δ (°)', fontsize=12)
        ax.set_ylabel('时间 t (s)', fontsize=12)
        ax.invert_yaxis()  # 第四象限：y 轴负向为时间增加
        ax.set_xlim(0, 180)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # ─────── 完整图 ───────

    def draw(self, save_path=None, t_clear: float = None):
        """绘制完整两子图，可选保存"""
        if t_clear is None:
            t_clear = self.t_clear

        self._run_simulation()

        fig, (ax_eq, ax_sw) = plt.subplots(
            2, 1, figsize=(13, 9.5), sharex=True,
            gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.12}
        )

        self._draw_equal_area_axes(ax_eq, t_clear=t_clear)
        self._draw_swing_axes(ax_sw)

        fig.suptitle('等面积准则与暂态稳定性 — 单机无穷大系统',
                      fontsize=15, fontweight='bold', y=0.99)

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f'已保存: {save_path}')
        return fig

    # ─────── 对比图 ───────

    def generate_comparison(self, t_clear_list=None, save_path=None):
        """生成不同切除时间对比图（2行 × N列）"""
        if t_clear_list is None:
            t_clear_list = [0.10, 0.20, 0.30, 0.40]

        n = len(t_clear_list)
        fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5), sharex='col')

        for i, tc in enumerate(t_clear_list):
            # 仿真
            self.t_clear = tc
            self._run_simulation()

            # 上排：等面积图（简化版）
            ax_top = axes[0, i]
            d = np.linspace(0, np.pi, 300)
            d_deg = np.degrees(d)
            ax_top.plot(d_deg, self.Pmax_pre * np.sin(d), 'b-', lw=2)
            ax_top.plot(d_deg, self.Pmax_post * np.sin(d), 'g-', lw=2)
            ax_top.axhline(self.params.Pm, color='k', ls='--', lw=1.2)

            dc = self._find_clearing_angle_from_time(tc)
            if dc and dc > self.delta0 + 0.01:
                da = np.linspace(self.delta0, dc, 200)
                ax_top.fill_between(np.degrees(da), 0, self.params.Pm,
                                    color='#ffbb78', alpha=0.5)
                dd = np.linspace(dc, self.delta_max_stable, 200)
                ax_top.fill_between(np.degrees(dd), self.params.Pm,
                                    np.maximum(self.Pmax_post * np.sin(dd), self.params.Pm),
                                    color='#98df8a', alpha=0.5)
                ax_top.axvline(np.degrees(dc), color='magenta', ls='-', lw=1.2, alpha=0.7)

            ax_top.set_xlim(0, 180)
            ax_top.set_ylim(bottom=0)
            ax_top.grid(True, alpha=0.3)
            ax_top.set_title(f'$t_{{clear}}={tc:.2f}$s', fontsize=12)

            if i == 0:
                ax_top.set_ylabel('P (p.u.)', fontsize=11)

            # 下排：摇摆曲线
            ax_bot = axes[1, i]
            if self.result and len(self.result.delta) > 0:
                delta_deg = self.result.delta
                t = self.result.time
                color = 'green' if self.result.stable else 'red'
                ax_bot.plot(delta_deg, t, color=color, lw=2)
                ax_bot.axvline(np.degrees(self.delta_crit), color='r', ls='--', lw=1, alpha=0.5)

            ax_bot.set_xlim(0, 180)
            ax_bot.invert_yaxis()
            ax_bot.grid(True, alpha=0.3)

            if i == 0:
                ax_bot.set_ylabel('t (s)', fontsize=11)
            if i == n // 2:
                axes[0, i].text(0.5, 1.12, '等面积准则', transform=axes[0, i].transAxes,
                                ha='center', fontsize=13, fontweight='bold')
                axes[1, i].text(0.5, 1.15, '摇摆曲线', transform=axes[1, i].transAxes,
                                ha='center', fontsize=13, fontweight='bold')

        axes[1, -1].set_xlabel('δ (°)', fontsize=11)
        fig.suptitle('不同故障切除时间下的暂态稳定对比',
                     fontsize=15, fontweight='bold', y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
            print(f'已保存: {save_path}')
        return fig

    # ─────── 交互模式 ───────

    def show(self, t_clear: float = None):
        """启动交互界面（需要 TkAgg 等 GUI 后端）"""
        matplotlib.use('TkAgg')
        if t_clear is not None:
            self.t_clear = t_clear
        self._run_simulation()

        self.fig, (self.ax_eq, self.ax_sw) = plt.subplots(
            2, 1, figsize=(13, 9.5), sharex=True,
            gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.12}
        )

        self._draw_equal_area_axes(self.ax_eq, t_clear=self.t_clear)
        self._draw_swing_axes(self.ax_sw)

        # Slider
        ax_sl = plt.axes([0.15, 0.015, 0.55, 0.028])
        self.slider = Slider(ax_sl, '$t_{clear}$ (s)', 0.01, 1.0,
                              valinit=self.t_clear, valstep=0.01, color='magenta')
        self.slider.on_changed(self._on_slider)

        # Button
        ax_btn = plt.axes([0.78, 0.01, 0.12, 0.04])
        self.btn = Button(ax_btn, '更新', color='lightyellow')
        self.btn.on_clicked(self._on_btn)

        self.fig.suptitle('等面积准则交互工具 — 拖动 Slider 查看不同切除时间',
                          fontsize=14, fontweight='bold', y=0.995)

        plt.show()

    def _on_slider(self, val):
        self.t_clear = val

    def _on_btn(self, event):
        self.t_clear = self.slider.val
        self._run_simulation()

        self.ax_eq.clear()
        self.ax_sw.clear()
        self._draw_equal_area_axes(self.ax_eq, t_clear=self.t_clear)
        self._draw_swing_axes(self.ax_sw)
        self.fig.canvas.draw_idle()


# ─────── 便捷函数 ───────

def show_equal_area_interactive(params=None, t_clear=None):
    """启动等面积准则交互工具"""
    tool = EqualAreaInteractive(params)
    tool.show(t_clear=t_clear)
    return tool


if __name__ == '__main__':
    # 参数：Pm=1.0, Pmax_pre=3.0, Pmax_post=1.5
    params = EqualAreaParams(
        Pm=1.0, E=1.2, V=1.0,
        X_pre=0.4, X_post=0.8,
        H=5.0, D=0.0,
        t_clear=0.20, t_end=3.0,
        show_swing=True,
    )
    tool = EqualAreaInteractive(params)

    # 单图
    tool.draw(save_path='/tmp/equal_area_v3.png')
    print('单图: /tmp/equal_area_v3.png')

    # 对比图
    tool.generate_comparison(
        t_clear_list=[0.10, 0.20, 0.30, 0.40],
        save_path='/tmp/equal_area_cmp_v3.png'
    )
    print('对比图: /tmp/equal_area_cmp_v3.png')
