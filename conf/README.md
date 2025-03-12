## Usage

Run the data generation script with the default configuration:

```bash
python scripts/make_inputs.py
```

Run with a specific configuration file:

```bash
python scripts/make_inputs.py --config-name=gray_scott_ball
```

Override specific configuration values:

```bash
python scripts/make_inputs.py model=gray_scott run_id=my_custom_run
```

## Configuration Structure

### Base Configuration

The `config.yaml` file contains the base configuration with default values. All other configuration files typically inherit from this base.

### Configuration Files

- `config.yaml`: Default configuration
- `gray_scott_ball.yaml`: Gray-Scott model with ball sampling
- `gray_scott_single.yaml`: Single trajectory Gray-Scott model
- `dataset.yaml`: Dataset generation with high-resolution parameters

## Configuration Fields

The configuration schema follows the structure described in `scripts/config_files/config.md`:

- `model`: Model type (e.g., "bruss", "gray_scott", "fhn")
- `ds_id`: Unique identifier for the dataset
- `ds_type`: Sampling approach ("one_trajectory" or "ball")
- `workdir_env_var`: Specifies the environment variable containing the path to the working directory (note that the actual outputs go into $workdir_env_var/data/<model>/ds_id/)
- `center_definition`: Method for defining parameter centers ("from_df" or "from_grid")
- `df_path`: Path to parameter CSV (required when center_definition="from_df")
- `grid_mode`: Method for grid parameter specification ("absolute" or "relative")
- `grid_params`: Grid parameter values
- `sim_params`: Core simulation parameters (Nx, dx, Nt, dt, n_snapshots)
- `initial_conditions`: List of initial condition configurations
- `sampling_std`: Parameter deviation for ball sampling
- `num_samples_per_point`: Number of samples in each parameter ball
- `num_samples_per_ic`: Repetitions per initial condition

## Advanced Usage

For more advanced usage, refer to the [Hydra documentation](https://hydra.cc/docs/intro/).