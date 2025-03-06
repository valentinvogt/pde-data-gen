# Consolidating Old Format Datasets

This document explains how to consolidate datasets that were generated using the old format into the new consolidated NetCDF format.

## What This Script Does

The `consolidate_old_format.py` script:

1. Scans a directory for JSON files in the old format
2. Creates a consolidated NetCDF dataset file
3. Adds metadata from each JSON file to the dataset
4. Consolidates the corresponding output NetCDF files into a single structured dataset

## Usage

```bash
python scripts/consolidate_old_format.py <input_directory> [output_file]
```

Arguments:
- `input_directory`: Directory containing the JSON files and their corresponding NetCDF output files
- `output_file` (optional): Path to save the consolidated NetCDF file. If not provided, it creates a file named `_dataset.nc` in the input directory.

## Example

If you have a directory with JSON and NetCDF files from an old format dataset:

```bash
python scripts/consolidate_old_format.py /path/to/old/dataset
```

This will create `/path/to/old/dataset/_dataset.nc` with all the metadata and simulation outputs consolidated.

## Expected Input Format

The script expects:

1. Each JSON file to contain metadata about a simulation run, including:
   - Model parameters (A, B, Du, Dv)
   - Simulation parameters (Nx, dx, Nt, dt, n_snapshots)
   - Initial condition information
   - Path to the output NetCDF file (`filename` field)

2. Each referenced output NetCDF file to contain simulation data with a structure compatible with the RD solver output format.

## Output

The consolidated output file will have:

1. A `run` dimension with metadata for each simulation
2. A `snapshot` dimension for time points
3. Structured dimensions for the spatial grid data
4. Variables containing all simulation data

This format is compatible with all the analysis tools in this project.