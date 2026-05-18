# GA-LEM-Inverter

GA-LEM-Inverter couples Fastscape landscape evolution modeling with genetic-algorithm inversion to estimate tectonic uplift fields from terrain. The current user-facing workflow is intentionally simple: install the environment, edit `config.ini`, and run `python runner.py`.

## Install

macOS / Linux / Windows Git Bash:

```bash
bash tools/environment/setup_environment.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\environment\setup_environment.ps1
# Or from CMD:
.\tools\environment\setup_environment.bat
```

The setup scripts create a project-local Conda environment at `./.conda`, install pinned compatible versions of Fastscape, xarray-simlab, zarr, numpy, geospatial packages, PyTorch, LPIPS, Jupyter, compiler tools, initialize the LPIPS Alex visual-similarity model, and then build the vendored Pecube engine into `vendor/pecube/bin/`.
Environment setup and diagnostic files live under `tools/environment/` to keep the project root focused on `config.ini`, `runner.py`, and compatibility entry points.

Validate the environment:

```bash
conda activate ./.conda
python tools/environment/test_environment.py
```

## Run

The default `config.ini` points to the bundled demo DEM at `demo/data/demo1/demo_dem.tif`:

```bash
python runner.py
```

Choose the experiment mode in `config.ini`:

```ini
[Run]
mode = main
```

Supported modes:

- `main`: real DEM inversion.
- `synthetic`: synthetic terrain validation.
- `k_sensitivity`: scale-factor/K sensitivity experiment.
- `pecube_coupled`: FastScape-to-Pecube coupling smoke test. The normal setup script builds the vendored Pecube engine automatically.

Legacy files `main.py`, `run_synthetic_experiment.py`, and `k_sensitivity_experiment.py` are kept only as compatibility wrappers. The recommended entry point is always `python runner.py`.

## Outputs

Each run creates a structured directory such as:

```text
demo/outputs/0001_2026-05-12_17-45-30_main/
```

Every run contains:

```text
summary.md
run_manifest.json
config_used.ini
logs/
figures/
arrays/
metrics/
```

Start with `summary.md`; it lists the mode, key parameters, metrics, and main output locations. Files in `figures/` are automatically numbered in generation order, for example `01_original_dem.png`.

## Key Configuration Rules

Most users only edit `config.ini`.

- `[Run] mode`: choose `main`, `synthetic`, `k_sensitivity`, or `pecube_coupled`.
- `[Data] terrain_path`: input DEM path for `main` mode.
- `[Data] fault_shp_path`: optional fault-line Shapefile; use `none` to skip it.
- `[Data] study_area_shp_path`: optional study-area Shapefile; use `none` to use the whole DEM.
- `[Model] time_total`: FastScape simulation time in years.
- `[Pecube] total_time_myr`: Pecube thermal-history window in Ma. It must match `[Model] time_total / 1e6`. For example, `time_total = 2e6` requires `total_time_myr = 2.0`; `time_total = 10e6` requires `total_time_myr = 10.0`. The runner validates this at startup to prevent a FastScape/Pecube time-axis mismatch.
- `[Pecube] sample_observations`: thermochronology sample CSV. Set it to `none` to disable the Pecube thermochronology constraint. The recommended input is Pecube's native wide CSV schema: `SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT`.
- `[Pecube] nskip`: horizontal subsampling used by Pecube. The default `4` keeps optimization outputs small; use `1` or `2` only for final high-resolution checks.
- `[Pecube] run_vtk`: default `false`; keep it disabled during GA optimization to avoid very large VTK output.

## Thermochronology CSV Input

`sample_observations` is a CSV file path. The recommended format is Pecube's native wide table, so the column names match Pecube documentation and existing Pecube datasets. The runner validates coordinates and writes the table into the generated Pecube project.

Recommended header:

```text
SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT
```

Column meanings:

| Column | Meaning | Unit / Format |
| --- | --- | --- |
| `SAMPLE` | Sample name | String; repeated analyses can be split into names such as `S01_AHE_1`, `S01_AHE_2` |
| `LON` | Longitude | Real geographic longitude, WGS84 / EPSG:4326 by default |
| `LAT` | Latitude | Real geographic latitude, WGS84 / EPSG:4326 by default |
| `HEIGHT` | Sample elevation | m |
| `AHE` / `DAHE` | AHe age / 1-sigma uncertainty | Ma |
| `AFT` / `DAFT` | AFT age / 1-sigma uncertainty | Ma |
| `ZHE` / `DZHE` | ZHe age / 1-sigma uncertainty | Ma |
| `ZFT` / `DZFT` | ZFT age / 1-sigma uncertainty | Ma |

Example:

```csv
SAMPLE,LON,LAT,HEIGHT,AHE,DAHE,AFT,DAFT,ZHE,DZHE,ZFT,DZFT
S01,103.6400,31.3800,3901,2.4,0.3,5.8,0.8,7.1,1.0,18.3,2.1
S01_AHE_2,103.6400,31.3800,3901,2.7,0.4,,,,,,
S02,103.8120,31.5260,2300,,,,,6.4,0.9,,
```

Rules:

- One row may contain multiple systems for the same sample, for example AHe + AFT + ZHe.
- If the same sample has repeated ages for the same system, Pecube's native wide table cannot store them in one row. Split them into multiple rows with distinct `SAMPLE` names such as `S01_AHE_1` and `S01_AHE_2`.
- Age and uncertainty columns must be filled in pairs, for example `AHE` requires `DAHE`.
- All ages and uncertainties are in Ma.
- The default `observation_coordinate_system = geographic` means `LON/LAT` are real geographic coordinates. The runner reads the DEM CRS/transform, derives the Pecube geographic grid, and checks whether samples fall inside the modeled region.
- The older long-table schema `sample_id,lon,lat,elevation,system,observed_age,sigma` is still accepted for backward compatibility, but it is no longer the recommended format.

## Pecube Coupling

Pecube is integrated as a vendored engine:

```text
vendor/pecube/source/              # upstream Pecube Fortran source, docs, license
vendor/pecube/bin/                 # built Pecube/Test/Vtk executables, not committed
vendor/pecube/UPSTREAM.md          # upstream repository, commit, and license record
ga_lem_inverter/integrations/
  pecube.py                        # PecubeEngine public Python API
  pecube_project.py                # writes Pecube.in and topo/uplift/temp grids
  pecube_parser.py                 # parses output/*.csv
  pecube_loss.py                   # thermochronology loss extension point
ga_lem_inverter/workflows/
  pecube_coupled.py                # runner mode that bridges FastScape and PecubeEngine
```

The fixed call chain is:

```text
config.ini
  -> runner.py
  -> workflows/pecube_coupled.py
  -> PecubeProjectBuilder
  -> vendor/pecube/bin/Test and vendor/pecube/bin/Pecube
  -> PecubeOutputParser
  -> pecube/pecube_result.json and pecube/pecube_metrics.json
```

Build products are written to `vendor/pecube/bin/`, while generated Pecube projects are written inside each run directory under `pecube/PGB01/`. `PGB01` is the five-character run folder expected by Pecube's Fortran command-line programs. The Python API is:

```python
from ga_lem_inverter.integrations.pecube import PecubeEngine
```

For the smoke workflow, users only need to set:

```ini
[Run]
mode = pecube_coupled
```
