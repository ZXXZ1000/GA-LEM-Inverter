# FastScape 正演 demo

不要和反演工具的 `demo/data/` 混淆——本目录只承载正演工具的样例。

## 一键 demo

复用 `demo/data/demo1/demo_dem.tif` 作现今 DEM、`demo_true_uplift.npy` 作 uplift 形态场，
跑一段 0.5 Ma 的 python 时间函数演化（mode=python，周期化倍率）。

```bash
python forward_run.py demo/forward/forward_config_demo.ini
```

完成后看终端打印的输出目录里的 `summary.md`。

## 文件清单

- `forward_config_demo.ini` —— demo 配置（py uplift_time 路径）
- `uplift_time_demo.py`     —— demo 时间函数（周期化倍率）

## 想自己跑？

回到项目根目录的 `forward_config.ini` 改路径，再 `python forward_run.py`。
