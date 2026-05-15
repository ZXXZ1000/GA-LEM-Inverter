# GA-LEM-Inverter

GA-LEM-Inverter 是一个基于 Fastscape 景观演化模型和遗传算法的构造隆升场反演工具。当前版本已经整理为统一入口：普通用户只需要安装环境、修改 `config.ini`、运行 `python runner.py`。

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
- `[Model] boundary_left/right/top/bottom`：FastScape 四边边界，顺序为 `left,right,top,bottom`。`fixed_value` 表示闭合/固定边界，`core` 表示开放出水口，`looped` 表示周期边界；四项都填写时会覆盖单值 `boundary_status`。demo3 默认 `bottom = core`，其余边闭合。
- `[Pecube] enabled`：默认 `auto`；配置了 `sample_observations` 时，`main` 模式会把 Pecube 热年代学 loss 接进 GA fitness。
- `[Pecube] sample_observations`：热年代学样品 CSV；设为 `none` 即关闭 Pecube 约束。用户侧 schema 固定为“一行一个样品-体系观测”，程序会自动转换成 Pecube 原生 A-file。
- `[Pecube] spatial_grid`：默认 `auto`，会从输入 DEM 的 CRS 和 transform 自动推导 Pecube 的 `lon0/lat0/dlon/dlat`。
- `[Pecube] observation_coordinate_system`：默认 `geographic`，样品 CSV 使用真实 `lon,lat`；也可用 `projected/dem_crs` 输入 DEM 投影坐标，或用 `grid_index` 输入 DEM 行列索引。
- `[Pecube] sea_level_temperature` / `lapse_rate`：自动生成 `temp0,temp1...` 地表温度场，公式为 `surface_temp = sea_level_temperature - lapse_rate * topography_km`。`lapse_rate` 用正数表示海拔越高越冷。
- `[Pecube] thickness` / `basal_temperature` / `thermal_diffusivity`：Pecube 热模型参数，会写入每次候选解的 `Pecube.in`。
- `[Pecube] nskip`：Pecube 读取地形网格的水平抽样步长。默认 `4`，优化搜索时显著减少中间数据和磁盘占用；最终高精度单次验证可调回 `1` 或 `2`。
- `[Pecube] run_vtk`：默认 `false`，不生成 `.vtk` 三维体数据。GA 搜索会运行大量候选解，开启 VTK 很容易产生几十 GB 输出；只有需要 ParaView 后处理时才建议临时开启。
- `[Pecube] run_test`：默认 `false`，优化搜索时不运行 Pecube 的 `Test` 检查程序，避免每个候选解额外写出 VTK 检查文件；调试 `Pecube.in` 时可临时改成 `true`。
- `[Pecube] include_uniform_velocity_field`：默认 `false`。Pecube 本身可叠加一个全域速度场，但主优化已经把 GA/FastScape 的 uplift 网格写入 `uplift0,uplift1...`，默认关闭额外速度场，避免重复剥露导致预测年龄系统性偏年轻。
- `[Fitness] terrain_loss_weight` / `thermo_loss_weight`：组合目标函数权重。
- `[Fitness] thermo_loss_scale`：把热年代学原始 normalized RMSE 映射到 0-1 的尺度，使用平滑有界函数保留候选之间的差异。

正式实验建议先用 demo 跑通，再逐步替换 DEM 和调大 GA 参数。

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

## Pecube 耦合

Pecube 以内置 vendor engine 形式接入。它既可以用 `mode = pecube_coupled` 做独立 smoke 验证，也可以在 `mode = main` 中作为热年代学约束参与 GA 搜索。

当前耦合边界需要明确：GA 优化的是一个二维空间隆升场，FastScape 使用这个同一个 uplift 场连续正演，并输出多时间步 DEM/topography 序列；Pecube 接收这组 DEM history 以及同一个 uplift 场重复形成的 uplift 序列来计算热年代学约束。也就是说，当前版本支持“静态空间 uplift 场 + 地形约束 + Pecube 热年代学约束”的联合反演。

Pecube 目录模式要求 `topo0` 是 time zero，也就是现今地形；后续 `topo1,topo2...` 是更老的地形。主优化里 FastScape 输出的是正演序列，程序会在写入 Pecube 前自动反转为“现今到过去”的顺序，避免把早期随机地形误当成现今边界。

Pecube 的地表温度序列不是常数占位。默认情况下，程序会对每一帧 Pecube 地形自动生成 `temp0,temp1...`：

```text
surface_temp = sea_level_temperature - lapse_rate * topography_km
```

因此高海拔区域会进入更冷的地表边界条件，地形历史和热模型会共同影响预测热年代学年龄。若通过 Python API 显式传入 `temperature_series`，则会覆盖这个自动生成结果。

Pecube 的 `uplift0,uplift1...` 已经承接 GA 当前候选解的隆升场，单位 `km/Myr`。由于 `1 mm/yr = 1 km/Myr`，主优化传入的 `0.5..1.5 mm/yr` 可直接作为 Pecube uplift 网格。默认不会再写 Pecube 的 `npoint=-1` 全域速度场；只有显式设置 `include_uniform_velocity_field = true` 时才会叠加 `velocity_km_per_myr`。

为了避免优化搜索输出过大，默认 Pecube 配置采用轻量输出：

```ini
[Pecube]
run_test = false
run_vtk = false
nskip = 4
save_ptt_paths = false
```

`run_test = false` 表示优化搜索时不执行 Pecube 的 `Test` 检查程序，因为它也可能写出 VTK 检查文件。`run_vtk = false` 表示不执行 Pecube 的 VTK 后处理程序，因此不会为每个候选解生成 `.vtk` 三维体数据。`nskip = 4` 表示 Pecube 每 4 个 DEM/topo 格点采样一次，用较低水平分辨率计算热年代学预测。正式最终解如果需要更高精度，可以把 `nskip` 调成 `1` 或 `2`，并只在单次后处理时临时打开 `run_test = true` 或 `run_vtk = true`。

尚未实现的是“时间变化 uplift history”作为优化变量，即 `time x y x` 的隆升历史。原因在于当前使用的 FastScape/xarray-simlab `basic_model` 接口只声明 `uplift__rate` 支持标量或二维 `(y, x)` 场，不支持一次正演中直接输入带时间维的 uplift forcing。不能把一次 FastScape 正演硬拆成多个彼此独立的小正演来伪造 uplift history，因为那会丢掉连续地形状态。后续若要支持时间变化 uplift，需要先实现可靠的 FastScape 连续状态继承，或改用/扩展支持 time-varying uplift forcing 的 FastScape 接口。

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

样品 CSV 至少包含：

```text
sample_id,lon,lat,elevation,system,observed_age,sigma
```

默认 demo 样品在 `demo/data/demo1/demo_thermo_samples.csv`，使用真实经纬度 `lon,lat`。`main` 模式会读取 DEM 的 CRS/transform，把 DEM 网格自动转换为 Pecube 的经纬度网格；样品点会自动校验是否落在 DEM/Pecube 范围内。当前支持 `AHe`、`ZHe`、`AFT`、`ZFT` 和 Pecube 输出中的常见 Ar 系列列名。若不需要热年代学约束，把 `sample_observations = none`。

`demo/data/demo3/demo3_thermo_samples.csv` 保留原始收集样品，包含不同热年代学体系和从 Ma 到数百 Ma 的年龄。10 Ma demo 不能同时解释这些长期年龄，因此配置文件默认使用 `demo/data/demo3/demo3_thermo_samples_10ma.csv`，只保留 `observed_age <= 10 Ma` 的 AHe/AFT/ZHe 样品。若要使用 ZFT 或数十到数百 Ma 的 ZHe，需要把 FastScape/Pecube 的时间窗、热参数和运动学设置扩展到相同地质时间尺度。

`sample_observations` 的固定 schema 是：

```csv
sample_id,lon,lat,elevation,system,observed_age,sigma
LMW-01,103.64,31.38,3901,ZHe,51.5,1.7
LMW-01,103.64,31.38,3901,AFT,56.5,19.8
```

这里的规则只有两条：
- 一行只表示一个样品在一个热年代学体系下的一条观测。
- 同一个 `sample_id` 如果有多个体系，就重复多行；程序内部会自动聚合成 Pecube 原生 `SAMPLE/LON/LAT/HEIGHT/AHE/DAHE/...` 结构。

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
