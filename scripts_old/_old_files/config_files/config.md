# Brusselator Model Simulation Configuration

This documentation describes the configuration file structure for running simulations of reaction-diffusion systems, with a primary focus on the **Brusselator** model and other supported models like Gray-Scott and FitzHugh-Nagumo.

## Configuration Overview

The configuration is structured as a Python dictionary containing parameters that control various aspects of the simulation:

```python
CONFIG = {
    "model": str,                 # Model type to simulate
    "run_id": str,                # Unique identifier for the simulation run
    "run_type": str,              # Simulation sampling strategy
    "center_definition": str,     # Method for defining parameter centers
    "df_path": str,               # Optional: Path to parameter data file
    "grid_mode": str,             # Optional: Method for grid parameter specification
    "grid_params": dict,          # Optional: Grid parameter values
    "sim_params": dict,           # Core simulation parameters
    "sampling_std": dict,         # Optional: Parameter deviation for ball sampling
    "initial_conditions": list,   # List of initial condition configurations
    "num_samples_per_point": int, # Number of samples in each parameter ball
    "num_samples_per_ic": int,    # Repetitions per initial condition
}
```

## Parameter Details

### Core Configuration

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str` | The reaction-diffusion model to simulate. Supported values: `"bruss"` (Brusselator), `"gray_scott"`, or `"fhn"` (FitzHugh-Nagumo) |
| `run_id` | `str` | Unique identifier for the simulation run. Generated files will be stored in `<data_dir>/<model>/<run_id>/` |
| `run_type` | `str` | Simulation sampling approach:<br>• `"one_trajectory"`: Single simulation for each parameter set<br>• `"ball"`: Multiple simulations with parameters sampled around specified centers |

### Parameter Space Definition

| Parameter | Type | Description |
|-----------|------|-------------|
| `center_definition` | `str` | Method for defining parameter centers:<br>• `"from_df"`: Use points from a data file specified by `df_path`<br>• `"from_grid"`: Generate points based on a regular grid defined by `grid_mode` and `grid_params` |
| `df_path` | `str` | Path to a CSV file containing parameter values with columns `A`, `B`, `Du`, `Dv`. Required when `center_definition` is `"from_df"` |
| `grid_mode` | `str` | Required when `center_definition` is `"from_grid"`:<br>• `"absolute"`: Direct parameter values for `A`, `B`, `Du`, `Dv`<br>• `"relative"`: Parameters given as `A`, `B_over_A`, `Du`, `Dv_over_Du` |
| `grid_params` | `dict` | Dictionary containing parameter lists based on `grid_mode`. Keys should match the parameters defined by the selected mode |
| `sampling_std` | `dict` | Maximum relative parameter deviations when using ball sampling. Required keys: `A`, `B`, `Du`, `Dv` |

### Simulation Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sim_params` | `dict` | Core simulation parameters applied to all trajectories: |

The `sim_params` dictionary must contain:

```python
{
    "Nx": int,           # Number of grid points along x-axis
    "dx": float,         # Grid spacing
    "Nt": int,           # Number of simulation timesteps
    "dt": float,         # Time step size
    "n_snapshots": int   # Number of state snapshots to save
}
```

### Initial Conditions

The `initial_conditions` parameter is a list of dictionaries, each specifying a different initial condition configuration. Each dictionary must contain a `"type"` key and additional parameters specific to that type:

| Initial Condition Type | Additional Parameters | Description |
|------------------------|----------------------|-------------|
| `"normal"` | `"sigma_u"`, `"sigma_v"` | Steady state with Gaussian noise. The parameters specify standard deviations for variables u and v |
| `"point_sources"` | `"density"` | Zero everywhere except for randomly placed point sources. The `density` parameter controls the proportion of grid points that are point sources |
| `"uniform"` | `"u_min"`, `"u_max"`, `"v_min"`, `"v_max"` | Steady state with uniform noise within the specified ranges for u and v |
| `"hex_pattern"` | `"amplitude"`, `"wavelength"` | Steady state with a randomly scaled sinusoidal pattern having the specified amplitude and wavelength |

### Sampling Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `num_samples_per_point` | `int` | Number of parameter samples to generate around each center point when using ball sampling (`run_type` = `"ball"`) |
| `num_samples_per_ic` | `int` | Number of repeated simulations for each parameter set and initial condition (only the random seed varies) |
