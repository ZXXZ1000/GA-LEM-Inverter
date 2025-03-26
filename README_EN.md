[中文版说明](README.md)

# GA-LEM-Inverter

This repository contains the implementation of a GA-based inversion method for deciphering tectonic uplift fields from landscape topography. Features include coupling with the Fastscape LEM, a multi-dimensional fitness function with perceptual similarity (LPIPS), and dimensionality reduction. See accompanying paper: [In Submission].


## Installation

```bash
git clone https://github.com/ZXXZ1000/GA-LEM-Inverter.git
cd GA-LEM-Inverter
```
Then install the required dependencies:
```bash
Copypip install -r requirements.txt
```
If you encounter installation issues, you may need to install some dependencies separately, such as GDAL library, Fastscape library, etc.

## Three Operation Modes

### 1. Synthetic Terrain Experiment

Synthetic terrain experiments are used to test algorithm performance, using artificially generated uplift fields to verify whether the algorithm can effectively recover the original uplift patterns.

#### Steps to Run:

1. Ensure all code files are in the same folder

2. Run directly:

   ```bash
   python run_synthetic_experiment.py
   ```

#### Main Parameters (can be modified in run_synthetic_experiment.py):

- `shape`: Terrain grid size, default (100, 100)

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

1. Prepare experimental data according to the config.ini file

2. Run:

   ```bash
   python main.py
   ```

#### config.ini Parameters:

**[Paths] Section**

- `terrain_path`: DEM file path (.tif format supported)
- `fault_shp_path`: Fault shapefile path
- `study_area_shp_path`: Study area shapefile path
- `output_path`: Results output directory

**[Model] Section**

- `k_sp_value`: Base erosion coefficient, default 6.92e-6
- `ksp_fault`: Fault zone erosion coefficient, default 2e-5
- `d_diff_value`: Hillslope diffusion coefficient, default 19.2
- `boundary_status`: Boundary condition, typically "fixed_value"
- `area_exp`: Area exponent, default 0.43
- `slope_exp`: Slope exponent, default 1.0
- `time_total`: Total simulation time (years), based on the geological history of the study area, typically millions to tens of millions of years

**[GeneticAlgorithm] Section**

- `ga_pop_size`: Population size, larger values provide better exploration but longer computation time
- `ga_max_iter`: Maximum iterations
- `ga_prob_cross`: Crossover probability
- `ga_prob_mut`: Mutation probability
- `lb` and `ub`: Lower and upper bounds for uplift rates, based on the actual situation of the study area, active orogens are about 1-10 mm/yr
- `n_jobs`: Number of parallel processes, -1 for all CPU cores
- `decay_rate`: Population size decay rate
- `patience`: Early stopping patience, stops after this many generations without improvement

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
