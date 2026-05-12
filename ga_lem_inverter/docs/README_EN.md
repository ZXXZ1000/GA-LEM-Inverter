# GA-LEM-Inverter

GA-LEM-Inverter couples Fastscape landscape evolution modeling with genetic-algorithm inversion to estimate tectonic uplift fields from terrain. The current user-facing workflow is intentionally simple: install the environment, edit `config.ini`, and run `python runner.py`.

## Install

macOS / Linux / Windows Git Bash:

```bash
bash setup_environment.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
```

The setup scripts create a project-local Conda environment at `./.conda` and install pinned compatible versions of Fastscape, xarray-simlab, zarr, numpy, geospatial packages, PyTorch, LPIPS, Jupyter, and related tools.

Validate the environment:

```bash
conda activate ./.conda
python test_environment.py
```

## Run

The default `config.ini` points to the bundled demo DEM:

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

Legacy files `main.py`, `run_synthetic_experiment.py`, and `k_sensitivity_experiment.py` are kept only as compatibility wrappers. The recommended entry point is always `python runner.py`.

## Outputs

Each run creates a structured directory such as:

```text
outputs/0001_2026-05-12_17-45-30_main/
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

Start with `summary.md`; it lists the mode, key parameters, metrics, and main output locations.
