[中文版说明](README.md)

# GA-LEM-Inverter

This repository contains the implementation of a GA-based inversion method for deciphering tectonic uplift fields from landscape topography. Features include coupling with the Fastscape LEM, a multi-dimensional fitness function with perceptual similarity (LPIPS), and dimensionality reduction. See accompanying paper: [In Submission].


## Installation

```bash
git clone https://github.com/ZXXZ1000/GA-LEM-Inverter.git
cd GA-LEM-Inverter
```
Then install the required dependencies with the platform bootstrapper:
```bash
# macOS / Linux / Windows Git Bash
bash setup_environment.sh
```

```powershell
# Windows PowerShell / CMD
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1
# or run setup_environment.bat
```

The setup scripts diagnose base tools, install or reuse Miniconda, create the local `.conda` environment, install pinned compatible dependencies, register the Jupyter kernel, and run a Fastscape smoke test. For diagnostics only:

```bash
bash setup_environment.sh --diagnose-only
```

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_environment.ps1 -DiagnoseOnly
```

The scripts always target the project-local `.conda` directory. They may reuse
an existing base/system conda executable as the package manager, but every conda
operation is run with `-p <repo>/.conda`, followed by a `sys.prefix` check.

## Run the Built-In Demo

The repository includes a lightweight DEM, fault shapefile, and study-area shapefile. `config.ini` already points to these demo files, so after installation you can run:

```bash
python main.py
```

Results are written to `demo_outputs/Expt_timestamp/`, including logs, DEM figures, erosion coefficient field, uplift field, optimization history, final simulated terrain, and comparison plots. The default demo also writes `demo_metrics.txt` and `demo_true_vs_inverted_uplift.png` so users can check the recovered uplift against the built-in truth field and the simulated terrain against the target terrain.

The synthetic experiment is also configured as a small default demo:

```bash
python run_synthetic_experiment.py
```

It writes synthetic DEMs, true/inverted uplift fields, fitness history, and metrics under `demo_outputs/synthetic_experiments/`. For full experiments, increase the grid size, population, and iteration count in `run_synthetic_experiment.py`.

## Three Operation Modes

### 1. Synthetic Terrain Experiment

Synthetic terrain experiments are used to test algorithm performance, using artificially generated uplift fields to verify whether the algorithm can effectively recover the original uplift patterns.

#### Steps to Run:

1. Ensure all code files are in the same folder

2. Run directly:

   ```bash
   python run_synthetic_experiment.py
   ```

   By default this runs a lightweight `simple` demo. For full experiments, adjust the commented configuration block at the top of `run_synthetic_experiment.py`.

#### Main Parameters (can be modified in run_synthetic_experiment.py):

- `shape`: Terrain grid size, default demo is (64, 64); full experiments can use (100, 100) or higher

- `patterns`: Uplift patterns to test, options are 'simple', 'medium', 'complex'

- `scale_factor`: Dimensionality reduction factor, higher values result in faster computation but lower precision

- `ga_params` : Genetic algorithm parameters

  - `pop`: Population size, default 100
  - `max_iter`: Maximum iterations, default 150
  - `prob_cross`: Crossover probability, default 0.7
  - `prob_mut`: Mutation probability, default 0.05
  - `lb` and `ub`: Lower and upper bounds for uplift rates, in mm/yr

#### Experiment Results:

Results will be saved in the `synthetic_experiments` directory, including:

- Comparison between true and inverted uplift fields
- Comparison between target and simulated terrains
- Fitness evolution history
- Error analysis charts

### 2. Scale Factor Sensitivity Experiment

The scale factor (K) sensitivity experiment evaluates how different dimensionality reduction factors affect inversion results, helping users choose the optimal value.

#### Steps to Run:

1. Ensure all code files are in the same folder
2. Run directly:
```bash
python k_sensitivity_experiment.py
```
#### Main Parameters (can be modified in k_sensitivity_experiment.py):

- `k_values`: List of scale factor values to test, e.g., [3, 5, 7, 10, 15]
- `repetitions`: Number of repetitions for each K value to obtain statistical significance
- `pattern`: Uplift pattern to test, options are 'simple', 'medium', or 'complex'
- `shape`: Terrain grid size
- `ga_params`: Genetic algorithm parameters (same as synthetic experiment)

#### Experiment Results:

Results will be saved in the `sensitivity_experiments` directory, including:

- Comprehensive analysis charts: showing relationships between K values and RMSE, computation time, R²
- Best K value recommendation: based on combined scores of accuracy and computational efficiency
- Visual comparisons of DEMs and uplift fields for different K values
- Detailed statistics and performance metrics

#### Results Interpretation:

- RMSE vs. Parameter Count: Shows the trade-off between accuracy and parameter space size
- K vs. Computation Time: Shows the impact of the scale factor on computational efficiency
- K vs. R²: Shows the impact of the scale factor on fitting quality
- Combined Score: Recommends the best K value by balancing accuracy and efficiency

### 3. Real Terrain Analysis

Real terrain analysis uses actual DEM data to invert for uplift fields.

#### Steps to Run:

1. The default `config.ini` already points to the built-in demo data. For real studies, replace the DEM, fault, and study-area paths.

2. Run:

   ```bash
   python main.py
   ```

#### config.ini Parameters:

**[Paths] Section**

- `terrain_path`: DEM file path (.tif format supported)
- `fault_shp_path`: Fault shapefile path; optional. If omitted, a uniform erosion coefficient field is used.
- `study_area_shp_path`: Study area shapefile path; optional. If omitted, the full DEM is used without rotation.
- `output_path`: Results output directory

**[Model] Section**

- `k_sp_value`: Base erosion coefficient, default 6.92e-6
- `ksp_fault`: Fault zone erosion coefficient, default 2e-5
- `d_diff_value`: Hillslope diffusion coefficient, default 19.2
- `boundary_status`: Boundary condition, typically "fixed_value"
- `area_exp`: Area exponent, default 0.42
- `slope_exp`: Slope exponent, default 1.0
- `time_total`: Total simulation time (years), default demo value is 2e5; reset it for real studies based on the geological history of the study area

**[GeneticAlgorithm] Section**

- `ga_pop_size`: Population size, larger values provide better exploration but longer computation time
- `ga_max_iter`: Maximum iterations
- `ga_prob_cross`: Crossover probability
- `ga_prob_mut`: Mutation probability
- `lb` and `ub`: Lower and upper bounds for uplift rates, based on the actual situation of the study area, active orogens are about 1-10 mm/yr
- `n_jobs`: Number of parallel processes, -1 for all CPU cores
- `decay_rate`: Population size decay rate
- `patience`: Early stopping patience, stops after this many generations without improvement
- `random_seed`: Random seed, fixed to 42 in the default demo for reproducible first-run results

**[Preprocessing] Section**

- `smooth_sigma`: Smoothing coefficient
- `scale_factor`: Dimensionality reduction factor, typically 5-10, higher values result in faster computation but lower precision
- `ratio`: DEM downsampling ratio, between 0-1
- `target_crs`: Target coordinate system (if reprojection is needed)

#### Results:

Results will be saved in the specified output directory, including:

- Visualization of original and rotated DEMs
- Erosion coefficient field visualization
- Inverted uplift rate field
- Simulated terrain based on the inverted uplift field
- Comparison with target terrain
- Default demo metrics and true-vs-inverted uplift comparison
- Uplift rate distribution plots
- 3D terrain visualization
- Optimization process records

## Troubleshooting

1. **Insufficient Memory**:
   - Reduce the `ratio` value to lower DEM resolution
   - Increase the `scale_factor` value to reduce parameter space
2. **Long Runtime**:
   - Reduce `ga_pop_size` and `ga_max_iter`
   - Increase `n_jobs` to use more CPU cores for parallel computation
   - Increase `scale_factor` to reduce computation load
3. **Convergence Issues**:
   - Adjust `lb` and `ub` based on the actual situation of the study area to define a more reasonable search range
   - Increase `patience` to allow more iterations without improvement
   - Check if erosion coefficients are reasonable
4. **Coordinate System Errors**:
   - Ensure all input files use the same coordinate system
   - Use the `target_crs` parameter for reprojection

## Citation

If you use this tool in your research, please cite our paper: Citation information will be updated after paper publication.

## Contact

For any questions or suggestions, please contact:

- Email: [xiangzhao@zju.edu.cn](
