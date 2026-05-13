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

默认 `config.ini` 已经指向 `demo/data/demo_dem.tif`，不改配置也可以直接运行：

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
- `[Optimization] uplift_min` / `uplift_max` / `uplift_precision`：隆升率搜索范围和步长，单位 `mm/yr`。程序内部用整数编码搜索，例如 `0.1..1.0 mm/yr` 配合 `0.1` 步长会变成 `1..10`，进入 FastScape 和输出图件前自动还原为真实隆升率；FastScape 内部会再统一换算成 `m/yr`。
- `[Optimization] n_jobs`：并行任务数；`-1` 使用全部 CPU 核心。Pecube 约束启用时，每个候选解会写入独立 `pecube/evaluations/eval_*` 目录，可以和 FastScape 一起并行。
- `[Pecube] enabled`：默认 `auto`；配置了 `sample_observations` 时，`main` 模式会把 Pecube 热年代学 loss 接进 GA fitness。
- `[Pecube] sample_observations`：热年代学样品 CSV；设为 `none` 即关闭 Pecube 约束。
- `[Pecube] spatial_grid`：默认 `auto`，会从输入 DEM 的 CRS 和 transform 自动推导 Pecube 的 `lon0/lat0/dlon/dlat`。
- `[Pecube] observation_coordinate_system`：默认 `geographic`，样品 CSV 使用真实 `lon,lat`；也可用 `projected/dem_crs` 输入 DEM 投影坐标，或用 `grid_index` 输入 DEM 行列索引。
- `[Fitness] terrain_loss_weight` / `thermo_loss_weight`：组合目标函数权重。
- `[Fitness] thermo_loss_scale`：把热年代学原始 normalized RMSE 压到 0-1 的尺度。

正式实验建议先用 demo 跑通，再逐步替换 DEM 和调大 GA 参数。

## Pecube 耦合

Pecube 以内置 vendor engine 形式接入。它既可以用 `mode = pecube_coupled` 做独立 smoke 验证，也可以在 `mode = main` 中作为热年代学约束参与 GA 搜索。

组合目标函数会先把两个约束归一化到 0-1，再做加权平均：

```text
terrain_loss = clip(terrain_loss_raw, 0, 1)
thermo_loss = clip(thermo_loss_raw / thermo_loss_scale, 0, 1)
total_loss = (
    terrain_loss_weight * terrain_loss
    + thermo_loss_weight * thermo_loss
) / (terrain_loss_weight + thermo_loss_weight)
```

其中：

- `terrain_loss_raw` 来自地形相似度，当前为 `1 - terrain_similarity`。
- 对每个热年代学样品，Pecube 输出预测年龄 `predicted_age`，样品 CSV 提供实测年龄 `observed_age` 和误差 `sigma`。
- 单个样品的归一化残差为 `(predicted_age - observed_age) / sigma`。
- `thermo_loss_raw` 是所有样品归一化残差的 RMSE。
- `thermo_loss_scale` 用来把热年代学 RMSE 压到 0-1，避免它因为数值量级大而压过地形约束。
- `terrain_loss`、`thermo_loss`、`total_loss` 才是进入 GA 搜索的 0-1 loss；`terrain_loss_raw` 和 `thermo_loss_raw` 会保存在表格里用于诊断。

样品 CSV 至少包含：

```text
sample_id,lon,lat,elevation,system,observed_age,sigma
```

默认 demo 样品在 `demo/data/demo_thermo_samples.csv`，使用真实经纬度 `lon,lat`。`main` 模式会读取 DEM 的 CRS/transform，把 DEM 网格自动转换为 Pecube 的经纬度网格；样品点会自动校验是否落在 DEM/Pecube 范围内。当前支持 `AHe`、`ZHe`、`AFT`、`ZFT` 和 Pecube 输出中的常见 Ar 系列列名。若不需要热年代学约束，把 `sample_observations = none`。

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

仓库内置 demo 文件统一放在 `demo/` 下：`demo/data/` 是输入数据，`demo/outputs/` 是默认运行输出。`demo/data/demo_dem.tif` 和 `demo/data/demo_true_uplift.npy` 用于默认 `main` demo 的轻量验证。默认配置不要求断层或研究区 Shapefile；如果没有提供这些文件，程序会自动使用 DEM 全域和均一侵蚀系数场。

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
