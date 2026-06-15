"""
Kundur 两区域四机系统参数
========================

定义 Kundur 教材中的经典两区域系统（Two-Area System），包含：
- 4 台发电机（每区域 2 台）
- 11 个节点
- 2 个负荷节点（含并联电容器）
- 12 条线路（含双回联络线）
- 4 台变压器

系统参数来源：
    Kundur P. Power System Stability and Control, McGraw-Hill, 1994:
    - 小干扰稳定分析：Section 12.5, Table 12.5, Figure 12.5
    - 暂态稳定分析：Section 13.7, Example 13.2

系统基准值：100 MVA, 230 kV（输电电压等级）

拓扑结构
--------
Area 1 (G1, G2):
    节点1(G1)──变压器──节点5──线路──节点6──线路──节点7(负荷L7)
    节点2(G2)──变压器──节点6

Area 2 (G3, G4):
    节点3(G3)──变压器──节点11──线路──节点10──线路──节点9(负荷L9)
    节点4(G4)──变压器──节点10

联络线（双回，接在节点7和节点9之间）:
    节点7 ──┬── 线路(110km) ──┬── 节点8 ──┬── 线路(110km) ──┬── 节点9
            └── 线路(110km) ──┘            └── 线路(110km) ──┘

关键结果（用于验证）:
    - 联络线功率：Area 1 → Area 2 ≈ 400 MW
    - 区域间振荡模式：f ≈ 0.56 Hz, ζ ≈ -0.02（无 PSS 时）
    - 局部振荡模式：f ≈ 1.1-1.2 Hz
"""

import numpy as np
from typing import Dict, List, Tuple

from psa4teaching.models.bus import Bus, BusType
from psa4teaching.models.line import Line
from psa4teaching.models.transformer import Transformer
from psa4teaching.models.generator import Generator, GeneratorModelType
from psa4teaching.models.load import Load, LoadModel
from psa4teaching.models.ieeet1 import IEEET1Params


# 系统基准值
S_BASE = 100.0       # 系统基准容量 (MVA)
V_BASE = 230.0       # 输电电压基准 (kV)
F_BASE = 60.0        # 系统频率 (Hz)

# 发电机额定值
GEN_MVA = 900.0      # 单台发电机额定容量 (MVA)
GEN_KV = 20.0        # 发电机额定电压 (kV)


def create_kundur_two_area_system() -> Dict:
    """创建 Kundur 两区域四机系统

    返回包含所有系统模型对象的字典，可直接用于潮流计算、
    小干扰稳定分析和暂态稳定仿真。

    Returns:
        Dict，包含以下键：
            - 'buses': List[Bus] — 11 个节点
            - 'lines': List[Line] — 12 条线路（含双回联络线）
            - 'transformers': List[Transformer] — 4 台变压器
            - 'generators': List[Generator] — 4 台发电机
            - 'loads': List[Load] — 2 个负荷
            - 'exciters': List[IEEET1Params] — 4 个 IEEET1 励磁系统
            - 'S_base': float — 系统基准容量
            - 'f_base': float — 系统频率
            - 'gen_mva': float — 发电机额定容量
            - 'gen_kv': float — 发电机额定电压

    Example:
        >>> from psa4teaching.data import create_kundur_two_area_system
        >>> sys = create_kundur_two_area_system()
        >>> buses = sys['buses']
        >>> generators = sys['generators']
        >>> print(f"节点数: {len(buses)}, 发电机: {len(generators)}")
        节点数: 11, 发电机: 4
    """
    # 基准转换系数：将发电机自身基准（900 MVA）转换为系统基准（100 MVA）
    base_ratio = S_BASE / GEN_MVA

    # ================================================================
    # 1. 创建节点 (11个)
    # ================================================================
    buses = _create_buses()

    # ================================================================
    # 2. 创建线路 (12条)
    # ================================================================
    lines = _create_lines()

    # ================================================================
    # 3. 创建变压器 (4台)
    #    Xt = 0.15 pu (on 900 MVA) → 对100 MVA基准转换
    # ================================================================
    transformers = _create_transformers(base_ratio)

    # ================================================================
    # 4. 创建发电机 (4台)
    #    参数对发电机自身基准（900 MVA），暂态/次暂态电抗值
    # ================================================================
    generators = _create_generators()

    # ================================================================
    # 5. 创建负荷 (2个)
    #    L7: 967 MW + j100 Mvar, 含 200 Mvar 并联电容器
    #    L9: 1767 MW + j100 Mvar, 含 350 Mvar 并联电容器
    # ================================================================
    loads = _create_loads()

    # ================================================================
    # 6. 创建励磁系统 (4个, IEEET1)
    #    采用快速励磁（晶闸管型），KA = 200
    # ================================================================
    exciters = _create_exciters()

    return {
        'buses': buses,
        'lines': lines,
        'transformers': transformers,
        'generators': generators,
        'loads': loads,
        'exciters': exciters,
        'S_base': S_BASE,
        'f_base': F_BASE,
        'gen_mva': GEN_MVA,
        'gen_kv': GEN_KV,
    }


def _create_buses() -> List[Bus]:
    """创建两区域系统的 11 个节点"""
    return [
        # ---- Area 1 ----
        # Bus 1: Generator 1 机端 (PV → 潮流中设为 SLACK)
        Bus(number=1, name="G1", bus_type=BusType.SLACK,
            V_specified=1.03, base_kv=GEN_KV),

        # Bus 2: Generator 2 机端 (PV)
        Bus(number=2, name="G2", bus_type=BusType.PV,
            P_specified=7.0,   # 700 MW / 100 MVA
            V_specified=1.01,
            Q_min=-3.0, Q_max=3.0,
            base_kv=GEN_KV),

        # Bus 3: Generator 3 机端 (PV, Area 2)
        Bus(number=3, name="G3", bus_type=BusType.PV,
            P_specified=7.19,  # 719 MW / 100 MVA, 略大于 G4 以承担联络线损耗
            V_specified=1.03,
            Q_min=-3.0, Q_max=3.0,
            base_kv=GEN_KV),

        # Bus 4: Generator 4 机端 (PV, Area 2)
        Bus(number=4, name="G4", bus_type=BusType.PV,
            P_specified=7.0,   # 700 MW / 100 MVA
            V_specified=1.01,
            Q_min=-3.0, Q_max=3.0,
            base_kv=GEN_KV),

        # ---- 输电网络节点 (230 kV) ----
        # Bus 5: 高压母线 Area 1 (G1 变压器高压侧)
        Bus(number=5, name="Bus5", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=V_BASE),

        # Bus 6: 高压母线 Area 1 (G2 变压器高压侧)
        Bus(number=6, name="Bus6", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=V_BASE),

        # Bus 7: 负荷母线 Area 1 (L7 + 并联电容器)
        Bus(number=7, name="Load7", bus_type=BusType.PQ,
            P_specified=-9.67,   # 967 MW 负荷
            Q_specified=-1.0,    # 100 Mvar 负荷（电容器补偿后）
            base_kv=V_BASE),

        # Bus 8: 联络线中间节点 (Area 1 侧)
        Bus(number=8, name="Tie8", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=V_BASE),

        # Bus 9: 负荷母线 Area 2 (L9 + 并联电容器)
        Bus(number=9, name="Load9", bus_type=BusType.PQ,
            P_specified=-17.67,  # 1767 MW 负荷
            Q_specified=-1.0,    # 100 Mvar 负荷（电容器补偿后）
            base_kv=V_BASE),

        # Bus 10: 高压母线 Area 2 (G4 变压器高压侧)
        Bus(number=10, name="Bus10", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=V_BASE),

        # Bus 11: 高压母线 Area 2 (G3 变压器高压侧)
        Bus(number=11, name="Bus11", bus_type=BusType.PQ,
            P_specified=0.0, Q_specified=0.0, base_kv=V_BASE),
    ]


def _create_lines() -> List[Line]:
    """创建两区域系统的输电线路

    线路参数（在 100 MVA, 230 kV 基准下）:
        R = 0.0001 pu/km, X = 0.001 pu/km, B = 0.00175 pu/km
    """
    # 单位长度线路参数 (pu/km)
    R_per_km = 0.0001
    X_per_km = 0.001
    B_per_km = 0.00175

    def line_params(km):
        """给定长度（km），返回 (R, X, B)"""
        return (R_per_km * km, X_per_km * km, B_per_km * km)

    lines = []

    # ---- Area 1 ----
    # 母线5-6: 25 km
    R, X, B = line_params(25)
    lines.append(Line(from_bus=5, to_bus=6, R=R, X=X, B=B, name="L5-6"))

    # 母线6-7: 10 km
    R, X, B = line_params(10)
    lines.append(Line(from_bus=6, to_bus=7, R=R, X=X, B=B, name="L6-7"))

    # 母线7-5: 25 km（形成环网，Area 1 内部）
    # 实际系统中母线5和7直接相连，提供辅助路径
    R, X, B = line_params(25)
    lines.append(Line(from_bus=5, to_bus=7, R=R, X=X, B=B, name="L5-7"))

    # ---- 联络线（双回，7-8 和 8-9）----
    # 母线7-8 (回路1): 110 km
    R, X, B = line_params(110)
    lines.append(Line(from_bus=7, to_bus=8, R=R, X=X, B=B, name="Tie7-8a"))
    # 母线7-8 (回路2): 110 km
    lines.append(Line(from_bus=7, to_bus=8, R=R, X=X, B=B, name="Tie7-8b"))

    # 母线8-9 (回路1): 110 km
    R, X, B = line_params(110)
    lines.append(Line(from_bus=8, to_bus=9, R=R, X=X, B=B, name="Tie8-9a"))
    # 母线8-9 (回路2): 110 km
    lines.append(Line(from_bus=8, to_bus=9, R=R, X=X, B=B, name="Tie8-9b"))

    # ---- Area 2 ----
    # 母线9-10: 10 km
    R, X, B = line_params(10)
    lines.append(Line(from_bus=9, to_bus=10, R=R, X=X, B=B, name="L9-10"))

    # 母线10-11: 25 km
    R, X, B = line_params(25)
    lines.append(Line(from_bus=10, to_bus=11, R=R, X=X, B=B, name="L10-11"))

    # 母线11-9: 25 km（形成环网，Area 2 内部）
    R, X, B = line_params(25)
    lines.append(Line(from_bus=11, to_bus=9, R=R, X=X, B=B, name="L11-9"))

    # 母线5-7 已在上面添加；还需添加母线6-5（另一条 Area 1 内部线路）
    # 实际上形成环网拓扑:
    #   5──6    G1变压器接在5，G2接在6
    #   │  │
    #   7──┘    负荷接在7
    #   ││
    #   8│      联络线双回
    #   ││
    #   9──10   G3变压器接在11→9，G4接在10
    #      │
    #     11    负荷接在9

    # 添加缺失的连接
    # 母线6-5 (已有)
    # 母线6-7 (已有)
    # 母线5-7 (已有)
    # 还需要: 母线8-7 和 7-8 已经添加了双回

    return lines


def _create_transformers(base_ratio: float) -> List[Transformer]:
    """创建 4 台发电机升压变压器

    参数（发电机 900 MVA 基准）:
        XT = 0.15 pu
        RT ≈ 0.0

    转换为系统基准（100 MVA）:
        XT_sys = XT * base_ratio = 0.15 * (100/900) = 0.01667 pu
    """
    XT_sys = 0.15 * base_ratio
    RT_sys = 0.0

    return [
        # Area 1
        Transformer(from_bus=1, to_bus=5, RT=RT_sys, XT=XT_sys,
                    k=1.0, name="T1"),
        Transformer(from_bus=2, to_bus=6, RT=RT_sys, XT=XT_sys,
                    k=1.0, name="T2"),
        # Area 2
        Transformer(from_bus=3, to_bus=11, RT=RT_sys, XT=XT_sys,
                    k=1.0, name="T3"),
        Transformer(from_bus=4, to_bus=10, RT=RT_sys, XT=XT_sys,
                    k=1.0, name="T4"),
    ]


def _create_generators() -> List[Generator]:
    """创建 4 台同步发电机

    参数（发电机 900 MVA 基准，Kundur Table 12.5）:
        Xd  = 1.8     Xq  = 1.7
        Xd' = 0.3     Xq' = 0.55
        Xd" = 0.25    Xq" = 0.25
        Xl  = 0.2
        Ra  = 0.0025
        Td0' = 8.0 s  Tq0' = 0.4 s
        Td0" = 0.03 s Tq0" = 0.05 s
        H   = 6.5 s (G1, G2), H = 6.175 s (G3, G4)
        D   = 0.0
    """
    return [
        # G1: Area 1, slack 区域发电机
        Generator(
            bus=1, name="G1",
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.25,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.25,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=6.5, D=0.0,
            Sb=GEN_MVA, Vb=GEN_KV,
            model_type=GeneratorModelType.DETAIL,
        ),

        # G2: Area 1
        Generator(
            bus=2, name="G2",
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.25,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.25,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=6.5, D=0.0,
            Sb=GEN_MVA, Vb=GEN_KV,
            model_type=GeneratorModelType.DETAIL,
        ),

        # G3: Area 2
        Generator(
            bus=3, name="G3",
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.25,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.25,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=6.175, D=0.0,
            Sb=GEN_MVA, Vb=GEN_KV,
            model_type=GeneratorModelType.DETAIL,
        ),

        # G4: Area 2
        Generator(
            bus=4, name="G4",
            Xd=1.8, Xd_prime=0.3, Xd_doubleprime=0.25,
            Xq=1.7, Xq_prime=0.55, Xq_doubleprime=0.25,
            Td0_prime=8.0, Td0_doubleprime=0.03,
            Tq0_prime=0.4, Tq0_doubleprime=0.05,
            H=6.175, D=0.0,
            Sb=GEN_MVA, Vb=GEN_KV,
            model_type=GeneratorModelType.DETAIL,
        ),
    ]


def _create_loads() -> List[Load]:
    """创建负荷模型

    L7 (Bus 7): 967 MW + j100 Mvar
    L9 (Bus 9): 1767 MW + j100 Mvar

    各负荷节点含并联电容器补偿：
    - Bus 7: 200 Mvar 电容器
    - Bus 9: 350 Mvar 电容器

    净无功 = 感性负荷 - 容性补偿
       Bus 7: 100 - 200 = -100 Mvar（净发出无功）
       Bus 9: 100 - 350 = -250 Mvar（净发出无功）

    在建模中，Q_specified = (感性负荷 - 电容器) / S_BASE
    实际潮流中 Q_specified 为负值表示吸收无功。
    这里的净载荷可以在潮流计算前进一步调整。

    使用恒功率模型（CONSTANT_POWER），对应于 Kundur 中的静态负荷假设。
    """
    return [
        # L7: 967 MW + j100 Mvar, 含 200 Mvar 电容器
        Load(
            bus=7,
            P0=9.67,       # 967 MW / 100 MVA
            Q0=1.0,        # 100 Mvar / 100 MVA
            V0=1.0,
            model_type=LoadModel.CONSTANT_POWER,
        ),
        # L9: 1767 MW + j100 Mvar, 含 350 Mvar 电容器
        Load(
            bus=9,
            P0=17.67,      # 1767 MW / 100 MVA
            Q0=1.0,        # 100 Mvar / 100 MVA
            V0=1.0,
            model_type=LoadModel.CONSTANT_POWER,
        ),
    ]


def _create_exciters() -> List[IEEET1Params]:
    """创建 4 个 IEEET1 励磁系统

    采用晶闸管型快速励磁（Kundur Section 12.5）:
        KA = 200, TA = 0.02 s
        KE = 1.0, TE = 0.5 s
        KF = 0.05, TF = 1.0 s

    所有 4 台发电机使用相同的励磁参数（忽略差异用于教学演示）。
    """
    exciter_params = IEEET1Params(
        KA=200.0, TA=0.02,
        KE=1.0, TE=0.5,
        KF=0.05, TF=1.0,
        VR_MIN=-1.0, VR_MAX=5.0,
        Efd_MIN=0.0, Efd_MAX=5.0,
    )

    return [exciter_params] * 4


# ================================================================
# 便捷查询函数
# ================================================================

def get_area_generators() -> Tuple[List[int], List[int]]:
    """获取各区域的发电机节点编号

    Returns:
        (area1_gens, area2_gens): 区域 1 和区域 2 的发电机节点列表
    """
    return [1, 2], [3, 4]


def get_load_buses() -> List[int]:
    """获取负荷节点编号"""
    return [7, 9]


def get_tie_line_buses() -> Tuple[int, int]:
    """获取联络线端口节点编号"""
    return (7, 9)


def get_generator_parameters_table() -> str:
    """打印发电机参数表（Markdown 格式）"""
    return """
| 参数 | G1 | G2 | G3 | G4 | 单位 |
|------|----|----|----|----|------|
| 额定容量 | 900 | 900 | 900 | 900 | MVA |
| 额定电压 | 20 | 20 | 20 | 20 | kV |
| Xd | 1.8 | 1.8 | 1.8 | 1.8 | pu |
| Xq | 1.7 | 1.7 | 1.7 | 1.7 | pu |
| Xd' | 0.3 | 0.3 | 0.3 | 0.3 | pu |
| Td0' | 8.0 | 8.0 | 8.0 | 8.0 | s |
| H | 6.5 | 6.5 | 6.175 | 6.175 | s |
| D | 0.0 | 0.0 | 0.0 | 0.0 | pu |
| 励磁 | IEEET1 | IEEET1 | IEEET1 | IEEET1 | - |
| KA | 200 | 200 | 200 | 200 | pu |
"""


__all__ = [
    "create_kundur_two_area_system",
    "get_area_generators",
    "get_load_buses",
    "get_tie_line_buses",
    "get_generator_parameters_table",
    "S_BASE",
    "V_BASE",
    "F_BASE",
    "GEN_MVA",
    "GEN_KV",
]
