# PDE Data Generation

Tools for generating and processing reaction-diffusion PDE simulation data.

## Setup

```bash
pip install -e .
```

## Structure

- `src/` - Core modules for data generation and processing
- `scripts/` - Runner and processing scripts
- `analysis/` - Analysis notebooks and scripts
- `data/` - Input parameter sets
- `conf/` - Configuration files, default: `conf/config.yaml`
- `test/` - Scripts for running the whole pipeline

## Solvers
There are two solvers to choose from:
- `run_from_netcdf`, based on a [fork](https://github.com/valentinvogt/pde-solvers-cuda) of Louis Hurschler's [pde-solvers-cuda](https://github.com/LouisHurschler/pde-solvers-cuda). See `test/nc_pipeline.sh` for an example.
- `ready`, using a fork (to be uploaded) of [Ready](https://https://github.com/GollyGang/ready). See `test/vti_pipeline.sh`.