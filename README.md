# GA-LEM-Inverter

GA-LEM-Inverter 是一个基于 Fastscape 景观演化模型和遗传算法的构造隆升场反演工具。当前版本已经整理为统一入口：普通用户只需要安装环境、修改 `config.ini`、运行 `python runner.py`。

## 新用户最快开始

第一次使用时，不需要改任何代码，也不需要准备自己的数据。项目已经内置了轻量 demo 数据。

### 1. 安装环境

macOS / Linux / Windows Git Bash：

```bash
bash tools/environment/setup_environment.sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\environment\setup_environment.ps1
```

安装脚本会把环境创建到项目根目录的 `./.conda`，并自动安装 FastScape、Pecube、LPIPS、地理空间依赖和编译工具。

### 2. 运行最快 demo

安装完成后激活环境：

```bash
conda activate ./.conda
```

然后直接运行：

```bash
python runner.py
```

默认 `config.ini` 已经指向：

```text
demo/data/demo1/demo_dem.tif
demo/data/demo1/demo_thermo_samples.csv
```

这个 demo 用于快速确认完整程序能跑起来，并生成一组轻量反演结果。

### 3. 查看结果

运行结束后，终端会打印：

```text
运行完成。结果目录: demo/outputs/xxxx_main
优先查看: demo/outputs/xxxx_main/summary.md
```

先打开 `summary.md`，再按编号查看 `figures/` 里的图片。每次运行都会生成一个新的编号和时间戳目录，不会覆盖旧结果。

### 4. 运行完整耦合 demo

如果想看真实区域 DEM、断层、研究区、Pecube 热年代学约束和 staged uplift history 的完整链路，运行：

```bash
python runner.py --config demo/configs/demo3_staged_smoke.ini
```

这个 demo 比默认 demo 慢，但能展示 FastScape + Pecube + GA 联合约束反演的完整流程。

## 一键安装

克隆仓库后进入项目目录：

```bash
git clone https://github.com/ZXXZ1000/GA-LEM-Inverter.git
cd GA-LEM-Inverter
```

macOS / Linux / Windows Git Bash：

```bash
bash tools/environment/setup_environment.sh
```

Windows PowerShell / CMD：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\environment\setup_environment.ps1
# 或在 CMD / 双击运行：
.\tools\environment\setup_environment.bat
```

安装脚本会诊断基础工具，安装或复用 Miniconda，并把项目环境创建到仓库根目录的 `./.conda`。依赖版本已锁定，避免 Fastscape、xarray-simlab、zarr、numpy 等包之间的兼容性问题。安装阶段还会初始化 LPIPS Alex 视觉相似度模型，并自动编译内置 Pecube，把 `Pecube/Test/Vtk` 放到 `vendor/pecube/bin/`。

环境相关脚本统一收在 `tools/environment/`，包括安装脚本、Windows wrapper、环境检查脚本和安装日志。根目录只保留日常使用入口和配置文件。

更详细的环境说明见 `tools/environment/ENVIRONMENT_SETUP.md`，锁定依赖清单见 `tools/environment/requirements_pinned.txt`。英文说明见根目录 `README_EN.md`。

安装完成后激活环境：

```bash
conda activate ./.conda
```

然后检查环境：

```bash
python tools/environment/test_environment.py
```

## 一键运行

默认 `config.ini` 已经指向 `demo/data/demo1/demo_dem.tif`，不改配置也可以直接运行：

```bash
python runner.py
```

程序启动时会先打印诊断信息，包括实验模式、输入文件、输出目录、网格大小、`scale_factor/K`、GA 规模、`n_jobs` 和当前预设。缺文件、缺依赖或配置错误会给出中文提示。

## 三种实验模式

在 `config.ini` 顶部修改：

```ini
[Run]
mode = main
```

可选模式：

- `main`：真实 DEM 主优化反演。默认 demo 会用内置 DEM 生成一组轻量但有意义的反演结果。
- `synthetic`：合成地形验证实验。用于确认 Fastscape 正演、GA 反演和评价指标链路可用。
- `k_sensitivity`：`scale_factor/K` 敏感性实验。用于比较不同降维因子对反演效果的影响。
- `pecube_coupled`：FastScape 序列转 Pecube 的耦合 smoke 验证。正常安装完成后可直接运行。

普通用户和新脚本都应通过 `python runner.py` 加 `config.ini` 的 `[Run] mode` 运行。`SyntheticExperiment`、`KSensitivityExperiment` 等旧类只保留兼容用途，不作为新的使用入口。

旧入口 `main.py`、`run_synthetic_experiment.py`、`k_sensitivity_experiment.py` 仍保留为兼容 wrapper，但推荐始终使用 `python runner.py`。

## 配置文件

普通用户只需要编辑 `config.ini`。每一项都有中文注释，说明参数含义和什么时候需要改。

最常改的几项：

- `[Run] mode`：选择 `main`、`synthetic`、`k_sensitivity` 或 `pecube_coupled`。
- `[Data] terrain_path`：真实 DEM 路径，`main` 模式必填。
- `[Data] fault_shp_path`：断层 Shapefile，可填 `none` 跳过。
- `[Data] study_area_shp_path`：研究区 Shapefile，可填 `none` 跳过。
- `[Optimization] scale_factor`：隆升场降维因子，也就是 K。
- `[Optimization] population_size`、`max_iterations`：GA 搜索规模。
- `[Optimization] search_strategy`：默认 `staged`，按 `coarse -> refine -> verify` 多阶段搜索；`single` 只保留给调试和极快 smoke。
- `[Optimization] enable_fitness_cache`：默认 `true`，重复候选解会复用已有 fitness，避免重复运行 FastScape + Pecube。
- `[Optimization] uplift_min` / `uplift_max` / `uplift_precision`：隆升率搜索范围和步长，单位 `mm/yr`。程序内部用整数编码搜索，例如 `0.1..1.0 mm/yr` 配合 `0.1` 步长会变成 `1..10`，进入 FastScape 和输出图件前自动还原为真实隆升率；FastScape 内部会再统一换算成 `m/yr`。
- `[Optimization] n_jobs`：并行任务数；`-1` 使用全部 CPU 核心。Pecube 约束启用时，每个候选解会写入独立 `pecube/evaluations/eval_*` 目录，可以和 FastScape 一起并行。
- `[Optimization] diversity_*`：控制停滞后的多样性注入，用于跳出局部最优；默认混合随机个体、当前最优扰动和地形 prior。
- `[Optimization] mutation_*`：控制自适应变异率；停滞时可临时增加变异，但不会超过配置上限。
- `[OptimizationStage1/2/3]`：staged 搜索的每阶段参数。后一阶段会继承前一阶段最优解，并继续围绕它搜索。
- `[Model] rainfall_factor`：FastScape 官方 `FlowAccumulator.runoff`，表示单位面积地表径流/降雨系数。`1.0` 保持默认均一降雨；大于 `1` 表示更强径流，小于 `1` 表示更弱径流，必须为正数。这个参数不会改写 `Ksp`。
- `[Rainfall] mode`：默认 `uniform`，使用一个常数 runoff；设为 `python` 时会加载 `rainfall_model.py` 中的 `rainfall(x, y, z, t_ma, params)`，每个 FastScape step 返回一张降雨/径流矩阵。函数里的 `t_ma` 是距今 Ma，`z` 是当前地形高程。输出图会保存 `rainfall_preview.png` 和 `fastscape_forcing_inputs.png`，便于先看降雨场再解释结果。
- `[Model] boundary_left/right/top/bottom`：FastScape 四边边界，顺序为 `left,right,top,bottom`。`fixed_value` 表示闭合/固定边界，`core` 表示开放出水口，`looped` 表示周期边界；四项都填写时会覆盖单值 `boundary_status`。demo3 默认 `bottom = core`，其余边闭合。
- `[Pecube] enabled`：默认 `auto`；配置了 `sample_observations` 时，`main` 模式会把 Pecube 热年代学 loss 接进 GA fitness。
- `[Pecube] sample_observations`：热年代学样品 CSV；设为 `none` 即关闭 Pecube 约束。推荐使用 Pecube 原生宽表格式：`SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT`。
- `[Pecube] spatial_grid`：默认 `auto`，会从输入 DEM 的 CRS 和 transform 自动推导 Pecube 的 `lon0/lat0/dlon/dlat`。
- `[Pecube] observation_coordinate_system`：默认 `geographic`，样品 CSV 使用真实 `lon,lat`；也可用 `projected/dem_crs` 输入 DEM 投影坐标，或用 `grid_index` 输入 DEM 行列索引。
- `[Pecube] sea_level_temperature` / `lapse_rate`：自动生成 `temp0,temp1...` 地表温度场，公式为 `surface_temp = sea_level_temperature - lapse_rate * topography_km`。`lapse_rate` 用正数表示海拔越高越冷。
- `[Pecube] thickness` / `basal_temperature` / `thermal_diffusivity`：Pecube 热模型参数，会写入每次候选解的 `Pecube.in`。
- `[Pecube] total_time_myr`：Pecube 热史时间窗，单位 Ma，必须和 `[Model] time_total` 对齐。例如 `time_total = 2e6` 年时写 `total_time_myr = 2.0`；`time_total = 10e6` 年时写 `total_time_myr = 10.0`。程序会在启动时检查，避免 FastScape 地形历史和 Pecube 热史时间轴错配。
- `[Pecube] nskip`：Pecube 读取地形网格的水平抽样步长。默认 `4`，优化搜索时显著减少中间数据和磁盘占用；最终高精度单次验证可调回 `1` 或 `2`。
- `[Pecube] run_vtk`：默认 `false`，不生成 `.vtk` 三维体数据。GA 搜索会运行大量候选解，开启 VTK 很容易产生几十 GB 输出；只有需要 ParaView 后处理时才建议临时开启。
- `[Pecube] run_test`：默认 `false`，优化搜索时不运行 Pecube 的 `Test` 检查程序，避免每个候选解额外写出 VTK 检查文件；调试 `Pecube.in` 时可临时改成 `true`。
- `[Pecube] include_uniform_velocity_field`：默认 `false`。Pecube 本身可叠加一个全域速度场，但主优化已经把 GA/FastScape 的 uplift 网格写入 `uplift0,uplift1...`，默认关闭额外速度场，避免重复剥露导致预测年龄系统性偏年轻。
- `[Fitness] terrain_loss_weight` / `thermo_loss_weight`：组合目标函数权重。
- `[Fitness] thermo_loss_scale`：把热年代学原始 normalized RMSE 映射到 0-1 的尺度，使用平滑有界函数保留候选之间的差异。

正式实验建议先用 demo 跑通，再逐步替换 DEM 和调大 GA 参数。

## 降雨 / 径流模型

降雨参数走 FastScape 官方 `FlowAccumulator.runoff` 输入，不会折进 `Ksp`。因此二者含义不同：

- `Ksp`：河流侵蚀系数，表示岩性、断层弱化或侵蚀效率等空间差异。
- `rainfall/runoff`：单位面积径流/降雨因子，表示水量差异，会影响汇流和河流侵蚀通量。

最简单用法是均一降雨：

```ini
[Rainfall]
mode = uniform
value = 1.0
```

`value = 1.0` 表示 FastScape 默认均一 runoff；大于 `1` 表示更强径流，小于 `1` 表示更弱径流，但必须大于 `0`。

需要空间、时间或高程相关降雨时，改成 Python 模型：

```ini
[Rainfall]
mode = python
module_path = ./rainfall_model.py
function = rainfall
dynamic = true
min = 0.1
max = 5.0
base = 1.0
```

程序会加载 `module_path` 里的函数：

```python
def rainfall(x, y, z, t_ma, params):
    ...
    return runoff
```

函数参数含义：

- `x, y`：当前 FastScape 网格坐标，单位 m，二维矩阵。
- `z`：当前地形高程，单位 m，二维矩阵。
- `t_ma`：距今时间，单位 Ma；`0` 表示现今。
- `params`：`[Rainfall]` 里除 `mode/module_path/function/dynamic/min/max` 外的键值，按字符串传入，例如 `base = 1.0`。

返回值约束：

- 可以返回一个正数 scalar，也可以返回一个二维矩阵。
- 如果返回矩阵，shape 必须和 `z` 完全一致，也就是当前 DEM/FastScape/Ksp 运行网格。
- 所有值必须是有限正数，不能有 `NaN`、`Inf`、`0` 或负数。
- `min/max` 可选；设置后会把函数输出裁剪到该范围内，并在 FastScape 动态运行和预览图中使用同一规则。

项目根目录的 `rainfall_model.py` 是可直接修改的模板；非线性示例在 `demo/rainfall/nonlinear_rainfall_demo.py`，对应配置是 `demo/configs/demo3_nonlinear_rainfall_smoke.ini`。主流程会输出 `rainfall_preview.png` 和 `fastscape_forcing_inputs.png`，建议先看这两张图确认降雨场和 Ksp/DEM 的空间关系。

## GA 优化策略

默认主优化使用 `staged` 多阶段 GA，而不是单阶段无脑搜索。原因是一次 fitness 评价会串联：

```text
uplift field
  -> FastScape forward model
  -> Pecube thermochronology prediction
  -> terrain loss + thermo loss
```

这个目标函数计算贵、维度高，并且地形约束和热年代学约束可能竞争。默认三阶段含义是：

- `coarse`：高变异、较强探索，先找到可行区域。
- `refine`：继承 coarse 最优解，降低变异，在较好区域内精修。
- `verify`：保留前阶段最优，使用较低变异检查稳定性。

如果只想调试代码链路，可在 `[Optimization]` 中设：

```ini
search_strategy = single
```

但正式 demo3 反演建议保留：

```ini
search_strategy = staged
enable_fitness_cache = true
n_jobs = -1
```

每次主优化会额外输出 GA 诊断文件：

```text
tables/ga_history.csv
metrics/ga_metrics.json
metrics/stage_metrics.json
```

其中 `ga_history.csv` 记录每代 best/mean/std fitness、unique chromosome 数量、cache hit/miss、mutation probability 和 diversity injection。`ga_metrics.json` 汇总本次 GA 搜索策略、best stage、cache 命中率和注入次数。`stage_metrics.json` 分阶段记录每个 stage 的参数和 best fitness。

demo3 已提供两个 staged 配置：

```bash
python runner.py --config demo/configs/demo3_staged_smoke.ini
python runner.py --config demo/configs/demo3_staged_medium_smoke.ini
```

`demo3_staged_smoke.ini` 用于快速验收链路；`demo3_staged_medium_smoke.ini` 用于代码和单元测试完成后的中等长度验收，目标运行时间控制在 2-3 小时。

## 初始地形

FastScape 初始地形在 `[Model]` 里配置：

```ini
[Model]
initial_topography = random
initial_elevation = 0.0
initial_topography_seed = 42
```

`initial_topography = random` 是默认值，表示使用 FastScape 的随机白噪声初始地形；`initial_topography_seed` 控制这张随机初始地形，固定 seed 可以复现实验。`initial_topography = flat` 表示全域同一个初始海拔，海拔值由 `initial_elevation` 指定，单位 m；例如 `initial_topography = flat` 且 `initial_elevation = 0.0` 就是从 0 m 平面开始演化。

主流程输出会保存 `figures/experiment_parameters.png`，里面列出本次实际使用的初始地形、seed、时间窗、GA 参数、uplift history 等关键配置；也会保存 `arrays/initial_topography.npy` 供复查。分阶段 uplift history 启用时，`figures/topography_history_summary.png` 会显示初始地形、阶段转折时刻地形和 0 Ma 地形。

## Pecube 耦合

Pecube 以内置 vendor engine 形式接入。它既可以用 `mode = pecube_coupled` 做独立 smoke 验证，也可以在 `mode = main` 中作为热年代学约束参与 GA 搜索。

当前耦合边界需要明确：GA 默认优化一个二维空间 uplift 场，FastScape 连续正演并输出多时间步 DEM/topography 序列，Pecube 接收同一套 `topography_series + uplift_series` 来计算热年代学约束。若 `[UpliftHistory] enabled = false`，`uplift_series` 是同一个平均场重复多帧；若开启 `stage_multiplier`，GA 优化的是：

```text
U(x,y,t) = U_base(x,y) * m_stage
```

也就是一个空间形态场 `U_base(x,y)` 加少数阶段倍率 `m_stage`，不是多个自由二维 uplift field。

Pecube 目录模式要求 `topo0` 是 time zero，也就是现今地形；后续 `topo1,topo2...` 是更老的地形。主优化里 FastScape 输出的是正演序列，程序会在写入 Pecube 前自动反转为“现今到过去”的顺序，避免把早期随机地形误当成现今边界。

Pecube 的地表温度序列不是常数占位。默认情况下，程序会对每一帧 Pecube 地形自动生成 `temp0,temp1...`：

```text
surface_temp = sea_level_temperature - lapse_rate * topography_km
```

因此高海拔区域会进入更冷的地表边界条件，地形历史和热模型会共同影响预测热年代学年龄。若通过 Python API 显式传入 `temperature_series`，则会覆盖这个自动生成结果。

Pecube 的 `uplift0,uplift1...` 已经承接 GA 当前候选解的隆升场，单位 `km/Myr`。由于 `1 mm/yr = 1 km/Myr`，主优化传入的 `0.5..1.5 mm/yr` 可直接作为 Pecube uplift 网格。默认不会再写 Pecube 的 `npoint=-1` 全域速度场；只有显式设置 `include_uniform_velocity_field = true` 时才会叠加 `velocity_km_per_myr`。

时间变化 uplift history 采用低维阶段倍率，而不是自由 `time,y,x` 三维场。推荐使用 `bounded` 模式，直接给每个时间阶段设置实际 multiplier 的上下界：

```ini
[UpliftHistory]
enabled = true
mode = stage_multiplier
multiplier_search_mode = bounded
multiplier_precision = 0.1
normalize_time_weighted_mean = false

[UpliftStage1]
start_ma = 10
end_ma = 6
multiplier_min = 0.4
multiplier_max = 0.9

[UpliftStage2]
start_ma = 6
end_ma = 3
multiplier_min = 0.8
multiplier_max = 1.3

[UpliftStage3]
start_ma = 3
end_ma = 0
multiplier_min = 1.1
multiplier_max = 1.8
```

这表示：

```text
10-6 Ma: multiplier 搜索 0.4..0.9，实际 uplift = U_base * multiplier
6-3 Ma : multiplier 搜索 0.8..1.3，实际 uplift = U_base * multiplier
3-0 Ma : multiplier 搜索 1.1..1.8，实际 uplift = U_base * multiplier
```

每个 `[UpliftStageN]` 就是一个阶段，`start_ma/end_ma` 直接写阶段时间，`multiplier_min/max` 直接写该阶段的实际 multiplier 范围。阶段块必须从 `[UpliftStage1]` 开始连续编号，时间必须从过去到现在连续排列，最后一个 `end_ma` 必须是 `0`；少写、断开或时间窗和 `[Model] time_total` 不一致，程序会直接报错。`bounded` 模式下这些上下界就是最终进入 FastScape/Pecube 的硬约束，不会再被归一化放大。优化输出会额外保存 `figures/uplift_history_summary.png`、`arrays/stage_uplift.npy`、`arrays/cumulative_stage_uplift.npy` 和 `arrays/stage_multipliers.npy`。

如果只是想让所有阶段共用一个 multiplier 范围，可以使用 `free` 模式：

```ini
[UpliftHistory]
enabled = true
mode = stage_multiplier
multiplier_search_mode = free
stage_times_ma = 10,6,3,0
multiplier_min = 0.5
multiplier_max = 1.5
multiplier_precision = 0.1
normalize_time_weighted_mean = false
```

`normalize_time_weighted_mean = true` 是高级选项，仅建议在 `free` 模式下使用。它会按阶段时长重缩放 multiplier，使时间加权平均值等于 1，因此会改变实际上下界。`bounded` 模式会强制按 `false` 处理，保证用户写的每阶段上下界就是实际约束。`mode = free` 和 `mode = bounded` 也可作为 `stage_multiplier + multiplier_search_mode` 的简写。

为了避免优化搜索输出过大，默认 Pecube 配置采用轻量输出：

```ini
[Pecube]
run_test = false
run_vtk = false
nskip = 4
save_ptt_paths = false
```

`run_test = false` 表示优化搜索时不执行 Pecube 的 `Test` 检查程序，因为它也可能写出 VTK 检查文件。`run_vtk = false` 表示不执行 Pecube 的 VTK 后处理程序，因此不会为每个候选解生成 `.vtk` 三维体数据。`nskip = 4` 表示 Pecube 每 4 个 DEM/topo 格点采样一次，用较低水平分辨率计算热年代学预测。正式最终解如果需要更高精度，可以把 `nskip` 调成 `1` 或 `2`，并只在单次后处理时临时打开 `run_test = true` 或 `run_vtk = true`。

仍未实现的是多个自由时间场：

```text
U_stage_1(y,x), U_stage_2(y,x), U_stage_3(y,x)
```

以及任意 `uplift__rate(time,y,x)` 三维 forcing。原因在于当前 Python FastScape `basic_model` 原生输入仍只支持标量或二维 `(y,x)` uplift；本项目通过自定义 `TimeScaledUplift` process 支持“一个空间场 + 阶段倍率”的连续正演。若未来要优化多个自由场，必须额外加入强正则、低维 delta field 或更多独立热史约束，否则不可识别性会很强。

组合目标函数会先把两个约束归一化到 0-1，再做加权平均：

```text
terrain_loss = clip(terrain_loss_raw, 0, 1)
thermo_loss = thermo_loss_raw / (thermo_loss_raw + thermo_loss_scale)
total_loss = (
    terrain_loss_weight * terrain_loss
    + thermo_loss_weight * thermo_loss
) / (terrain_loss_weight + thermo_loss_weight)
```

其中：

- `terrain_loss_raw` 来自地形相似度，当前为 `1 - terrain_similarity`。
- 对每个热年代学样品，Pecube 输出预测年龄 `predicted_age`，样品 CSV 提供实测年龄 `observed_age` 和误差 `sigma`。
- 单个样品的归一化残差为 `(predicted_age - observed_age) / effective_sigma`。
- `effective_sigma = max(sigma, thermo_sigma_min, observed_age * thermo_sigma_relative)`。这个下限用于避免 1-3 Ma 年轻样品因为 `sigma=0.1 Ma` 这类很小的实验误差在 GA 中获得不成比例的权重。
- `thermo_loss_raw` 是所有样品归一化残差的 RMSE。
- `thermo_loss_scale` 控制热年代学 RMSE 映射到 0-1 的速度。它不是硬截断；即使 `thermo_loss_raw > thermo_loss_scale`，不同候选之间的热年代学差异仍会进入 GA 排序。
- `terrain_loss`、`thermo_loss`、`total_loss` 才是进入 GA 搜索的 0-1 loss；`terrain_loss_raw` 和 `thermo_loss_raw` 会保存在表格里用于诊断。

### 热年代学样品 CSV 怎么写

`sample_observations` 是一个 CSV 文件路径。推荐输入格式直接采用 Pecube 原生宽表，这样用户看到的列名和 Pecube 文档、论文数据整理方式一致，程序只负责坐标检查和复制到本次 Pecube project。

推荐表头：

```text
SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT
```

字段含义：

| 字段 | 含义 | 单位 / 写法 |
| --- | --- | --- |
| `SAMPLE` | 样品编号 | 字符串；同一物理样品多个重复测年可拆成 `S01_AHE_1`、`S01_AHE_2` |
| `LON` | 经度 | 默认真实经度，WGS84 / EPSG:4326 |
| `LAT` | 纬度 | 默认真实纬度，WGS84 / EPSG:4326 |
| `HEIGHT` | 样品高程 | m |
| `AHE` / `DAHE` | AHe 年龄 / 1σ 误差 | Ma |
| `AFT` / `DAFT` | AFT 年龄 / 1σ 误差 | Ma |
| `ZHE` / `DZHE` | ZHe 年龄 / 1σ 误差 | Ma |
| `ZFT` / `DZFT` | ZFT 年龄 / 1σ 误差 | Ma |

最小示例：

```csv
SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT
S01,103.6400,31.3800,3901,2.4,0.3,5.8,0.8,7.1,1.0,18.3,2.1
S01_AHE_2,103.6400,31.3800,3901,2.7,0.4,,,,,,
S02,103.8120,31.5260,2300,,,,,6.4,0.9,,
```

规则：

- 一行可以包含同一样品的多个体系，例如 AHe + AFT + ZHe。
- 如果同一样品同一体系有多个重复年龄，Pecube 原生宽表无法在同一行表达，应拆成多行，并给不同 `SAMPLE` 名称，例如 `S01_AHE_1`、`S01_AHE_2`。
- 年龄列和误差列必须成对出现，例如填了 `AHE` 就要填 `DAHE`。
- 所有年龄和误差都用 Ma；不要写 ka、year 或 Myr 文本。
- 默认 `observation_coordinate_system = geographic`，所以 `LON/LAT` 必须是真实经纬度。程序会读取 DEM 的 CRS/transform，把 DEM 网格自动转换为 Pecube 的经纬度网格，并检查样品是否落在 DEM/Pecube 范围内。

默认 demo 样品在 `demo/data/demo1/demo_thermo_samples.csv`，使用真实经纬度坐标。当前推荐格式支持 `AHE`、`AFT`、`ZHE`、`ZFT` 四组 Pecube 原生列。若不需要热年代学约束，把 `sample_observations = none`。

`demo/data/demo3/demo3_thermo_samples.csv` 保留原始收集样品，包含不同热年代学体系和从 Ma 到数百 Ma 的年龄。10 Ma demo 不能同时解释这些长期年龄，因此配置文件默认使用 `demo/data/demo3/demo3_thermo_samples_10ma.csv`，只保留 `observed_age <= 10 Ma` 的 AHe/AFT/ZHe 样品。若要使用 ZFT 或数十到数百 Ma 的 ZHe，需要把 FastScape/Pecube 的时间窗、热参数和运动学设置扩展到相同地质时间尺度。

旧版 long-table 仍然兼容，但不再作为推荐格式：

```csv
sample_id,lon,lat,elevation,system,observed_age,sigma
LMW-01,103.64,31.38,3901,ZHe,51.5,1.7
LMW-01,103.64,31.38,3901,AFT,56.5,19.8
```

程序会自动识别两种 CSV 表头。原生宽表会直接写入 Pecube project；旧版 long-table 会先转换成 Pecube 原生宽表。

启用后会新增这些输出：

```text
tables/predicted_thermochronology.csv
tables/pecube_fitness_history.csv
figures/*pecube_observed_vs_predicted_ages.png
figures/*pecube_residual_spatial_map.png
pecube/evaluations/
pecube/best/
```

文件架构如下：

```text
vendor/pecube/
├── source/                # Pecube 上游 Fortran 源码、README、docs、LICENSE
├── bin/                   # build_pecube.sh 编译出的 Pecube/Test/Vtk，可执行文件不提交
├── build/                 # 预留构建缓存，不提交
├── projects/              # 预留批量运行目录，不提交
├── UPSTREAM.md            # 上游仓库、commit、license 记录
└── LICENSE                # Pecube GPLv3 license 副本

ga_lem_inverter/integrations/
├── pecube.py              # PecubeEngine：对外 Python API，负责调用 Test/Pecube 并返回 PecubeResult
├── pecube_project.py      # PecubeProjectBuilder：把 FastScape 数组写成 Pecube.in + topo/uplift/temp 文件
├── pecube_parser.py       # PecubeOutputParser：读取 output/*.csv
├── pecube_loss.py         # ThermochronologyLoss：旧扩展点
└── pecube_fitness.py      # PecubeFitnessEvaluator：主优化中的热年代学 loss、预测表和图件

ga_lem_inverter/workflows/
├── main_inversion.py      # main 模式可选调用 PecubeFitnessEvaluator
└── pecube_coupled.py      # pecube_coupled 模式，生成 FastScape 序列并交给 PecubeEngine
```

调用链固定为：

```text
config.ini
  -> runner.py 读取 [Run] mode
  -> main_inversion.py 或 pecube_coupled.py 生成 topography/uplift/temperature 序列
  -> PecubeProjectBuilder 写 pecube/PGB01/input/Pecube.in 和 data/fastscape/topo0,uplift0,temp0...
  -> PecubeEngine 调 vendor/pecube/bin/Test 和 vendor/pecube/bin/Pecube
  -> PecubeOutputParser 读取 pecube/PGB01/output/*.csv
  -> Python 按 sample_observations 抽取预测年龄并计算 thermo_loss
  -> predicted_thermochronology.csv、pecube_fitness_history.csv 和图件
```

Python API 入口是：

```python
from ga_lem_inverter.integrations.pecube import PecubeEngine
```

一键安装脚本会自动编译 Fortran engine。只有需要手动重编译 Pecube 时，再运行：

```bash
bash tools/environment/build_pecube.sh
```

编译产物会写入 `vendor/pecube/bin/`。独立 smoke 模式会把 Pecube project 写入 `pecube/PGB01/`；主优化会把每次评价写入 `pecube/evaluations/eval_*/PGB01/`，并把最佳可验证评价复制到 `pecube/best/`。`PGB01` 是 Pecube Fortran 程序兼容的 5 字符项目名。

## 输出目录

每次运行都会生成独立目录：

```text
demo/outputs/0001_2026-05-12_17-45-30_main/
demo/outputs/0002_2026-05-12_18-10-02_synthetic/
```

目录结构固定：

```text
summary.md
run_manifest.json
config_used.ini
logs/
figures/
arrays/
metrics/
```

优先打开 `summary.md`，里面有本次运行模式、关键参数、主要指标和输出位置。`figures/` 内的图片会按生成顺序自动编号，例如 `01_original_dem.png`、`02_rotated_dem.png`，便于按流程查看。`run_manifest.json` 会记录 git commit、Python 版本、关键依赖版本、配置副本和输出文件清单。

## Demo 数据

仓库内置 demo 文件统一放在 `demo/` 下：`demo/data/` 是输入数据，`demo/outputs/` 是默认运行输出。`demo/data/demo1/demo_dem.tif` 和 `demo/data/demo1/demo_true_uplift.npy` 用于默认 `main` demo 的轻量验证。默认配置不要求断层或研究区 Shapefile；如果没有提供这些文件，程序会自动使用 DEM 全域和均一侵蚀系数场。

## 常见调整

运行慢：

- 减小 `[Optimization] population_size`
- 减小 `[Optimization] max_iterations`
- 增大 `[Optimization] scale_factor`
- 大 DEM 先把 `[Optimization] ratio` 设为 `0.25` 或 `0.5`

结果太粗：

- 减小 `[Optimization] scale_factor`
- 增大 GA 种群和迭代数
- 检查 `uplift_min` / `uplift_max` 是否覆盖合理隆升率范围

路径错误：

- 使用绝对路径，或把数据放在项目目录下再用相对路径
- Windows 路径可以写成 `C:/path/to/file.tif`
- 可选 Shapefile 不使用时写 `none`

## 引用

如果您在研究中使用本工具，请引用相关论文。引用信息将在论文发表后更新。

## 联系

- xiangzhao@zju.edu.cn
