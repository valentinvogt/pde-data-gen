#!/bin/bash
#SBATCH --job-name=make-inp
#SBATCH --output=make-inp-%j.out
#SBATCH --error=make-inp-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

HDF5_USE_FILE_LOCKING=FALSE
set -a && source .env

python3 scripts/make_inputs.py --config-name=param_sweep
