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

安装脚本会诊断基础工具，安装或复用 Miniconda，并把项目环境创建到仓库根目录的 `./.conda`。依赖版本已锁定，避免 Fastscape、xarray-simlab、zarr、numpy 等包之间的兼容性问题。

环境相关脚本统一收在 `tools/environment/`，包括安装脚本、Windows wrapper、环境检查脚本和安装日志。根目录只保留日常使用入口和配置文件。

更详细的环境说明见 `ga_lem_inverter/docs/ENVIRONMENT_SETUP.md`，锁定依赖清单见 `ga_lem_inverter/docs/requirements_pinned.txt`。

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
- `pecube_coupled`：FastScape 序列转 Pecube 的耦合 smoke 验证。首次使用前需要先编译内置 Pecube engine。

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
- `[Optimization] n_jobs`：并行任务数，Windows 首次 demo 建议保持 `1`。
- `[Pecube] enabled`：默认 `auto`，会跟随 `[Run] mode` 自动启用或跳过 Pecube。

正式实验建议先用 demo 跑通，再逐步替换 DEM 和调大 GA 参数。

## Pecube 耦合

Pecube 以内置 vendor engine 形式接入。文件架构如下：

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
└── pecube_loss.py         # ThermochronologyLoss：热年代学观测 loss 的扩展点

ga_lem_inverter/workflows/
└── pecube_coupled.py      # runner 的 pecube_coupled 模式，生成 FastScape 序列并交给 PecubeEngine
```

调用链固定为：

```text
config.ini
  -> runner.py 读取 [Run] mode
  -> ga_lem_inverter/workflows/pecube_coupled.py 生成 topography/uplift/temperature 序列
  -> PecubeProjectBuilder 写 pecube/PGB01/input/Pecube.in 和 data/fastscape/topo0,uplift0,temp0...
  -> PecubeEngine 调 vendor/pecube/bin/Test 和 vendor/pecube/bin/Pecube
  -> PecubeOutputParser 读取 pecube/PGB01/output/*.csv
  -> pecube/pecube_result.json 和 pecube/pecube_metrics.json
```

Python API 入口是：

```python
from ga_lem_inverter.integrations.pecube import PecubeEngine
```

首次运行 Pecube 耦合模式前，先编译 Fortran engine：

```bash
bash tools/environment/build_pecube.sh
```

编译产物会写入 `vendor/pecube/bin/`，运行时 Pecube project 会写入本次输出目录的 `pecube/PGB01/`。`PGB01` 是 Pecube Fortran 程序兼容的 5 字符项目名。普通用户仍然只需要改 `config.ini`，把 `[Run] mode` 设为 `pecube_coupled`。

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
