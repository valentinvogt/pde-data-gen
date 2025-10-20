This repository contains convenience code to simulate the Brusselator and Gray-Scott models using a [fork](https://github.com/valentinvogt/pde-solvers-cuda) of Louis Hurschler's [pde-solvers-cuda](https://github.com/LouisHurschler/pde-solvers-cuda).

## Workflow
1. Generate inputs from a config file. For `conf/example.yaml`, run:
    ```sh
    python scripts/create_inputs.py --config-name=example
    ```
    See `conf/README.md` for details on the format.
There are two available solvers: `pde-solvers-cuda` and `ready`.

### Ready
1. Create inputs:
    ```sh
    python scripts/create_inputs.py --config-name=example --use-vti=True
    ```
2. Run the solver. For this, the executable `rdy` should be in the `build` directory. Make sure to adjust "model" and "dataset_id" in the script:
    ```sh
    sbatch scripts/run_vti.sh
    ```
3. Run 
    ```
    scripts/consolidate_vti.sh
    ```
    with the same "model" and "dataset_id".

### pde-solvers-cuda
1. Create inputs:
    ```sh
    python scripts/create_inputs.py --config-name=example
    ```
2. Run the solver. For this, the executable `run_from_netcdf` should be in the `build` directory. 
    ```
    sbatch scripts/run_from_inputs.sh
    ```
3. Run 
    ``` 

## Documentation
Before running python files:
```sh
set -a && source .env
```
