# psa4teaching 项目规划

## 项目定位
面向本科教学的电力系统分析 Python 包，覆盖潮流、短路、稳定三大模块。
参考教材：陈珩《电力系统稳态分析》、李光琦《电力系统暂态分析》、Kundur《Power System Stability and Control》。

## 技术栈
- Python >= 3.8，核心依赖仅 NumPy >= 1.20
- 测试：pytest >= 6.0
- 可视化和文档：matplotlib, jupyter, sphinx
- 包管理：setuptools (setup.py)

## 架构原则
1. **模型与算法分离**：models/ 只存数据类（dataclass），算法在对应模块中
2. **可读性优先**：变量命名与教材一致，注释充分
3. **教学友好**：中间结果可查（如 NR 迭代历史 history），verbose 参数控制输出
4. **函数式接口**：算法函数接受模型对象，返回命名的 result dataclass

## 关键当前状态（2026-06-02）
- 最近工作集中在：
  - 修复 TGOV1 调速器 P_ref 功率参考值 bug
  - 重构 SMIB 仿真，对齐 ENTSO-E 标准
  - Bus 类添加 base_kv 属性
  - 控制器模型（governor/exciter/pss）已建好框架
- examples/entsoe_smib_validation.ipynb 有未提交修改
- GRADUATE_EXTENSION_PLAN.md 是研究生扩展路线图

## 常用命令
- 安装：`pip install -e .`
- 测试：`pytest` 或 `pytest --cov=psa4teaching`
- 单个模块测试：`pytest tests/test_models.py -v`
- 示例运行：`python examples/basic_usage.py`
- Notebook：`jupyter notebook cases/`

## 短期待办（根据 GRADUATE_EXTENSION_PLAN 优先级）
1. 连续潮流法 (Continuation Power Flow)
2. 最优潮流 (OPF)
3. 加权最小二乘状态估计 (WLS)
4. GFL/GFM 逆变器控制模型
5. 电压稳定性分析 (PV/QV 曲线)

## 文件修改注意事项
- 所有新增模型放在 psa4teaching/models/ 下，遵循 dataclass 模式
- 新算法需返回 dataclass result 对象
- 新增公共 API 需在 models/__init__.py 或对应模块 __init__.py 中导出
- 测试覆盖：每个新模块需对应 tests/ 中的测试文件
- 不要直接修改 cases/ 目录下的教学 Notebook（它们是给学生用的），新验证案例放 examples/
- Windows 环境注意编码问题（open 文件时指定 encoding='utf-8'）
- 注意 .ipynb_checkpoints/ 不要提交
