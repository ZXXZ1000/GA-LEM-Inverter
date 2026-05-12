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

The setup scripts create a project-local Conda environment at `./.conda` and install pinned compatible versions of Fastscape, xarray-simlab, zarr, numpy, geospatial packages, PyTorch, LPIPS, Jupyter, and related tools.
Environment setup and diagnostic files live under `tools/environment/` to keep the project root focused on `config.ini`, `runner.py`, and compatibility entry points.

Validate the environment:

```bash
conda activate ./.conda
python tools/environment/test_environment.py
```

## Run

The default `config.ini` points to the bundled demo DEM at `demo/data/demo_dem.tif`:

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
- `pecube_coupled`: FastScape-to-Pecube coupling smoke test. Build the vendored Pecube engine before first use with `bash tools/environment/build_pecube.sh`.

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
