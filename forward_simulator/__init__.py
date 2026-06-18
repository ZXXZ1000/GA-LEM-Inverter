"""FastScape 正演工具，独立于 GA 反演工具。

提供一个干净入口：给定现今 DEM、降雨、uplift 场（可时变），跑一次 FastScape，
输出地形演化序列、累积剥蚀、平均剥蚀率和净抬升场。

普通用户只需要：
1. 编辑项目根目录的 ``forward_config.ini``。
2. （可选）编辑 ``uplift_time.py`` 自定义时间函数。
3. 运行 ``python forward_run.py``。
"""

from forward_simulator.workflow import run_forward_simulation

__all__ = ["run_forward_simulation"]
