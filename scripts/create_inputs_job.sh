#!/bin/bash
#SBATCH --job-name=make-inp
#SBATCH --output=make-inp-%j.out
#SBATCH --error=make-inp-%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00

module load stack/2024-06
module load gcc/12.2.0
module load python/3.11.6

HDF5_USE_FILE_LOCKING=FALSE
set -a && source .env

PYTHON scripts/create_inputs.py --config-name=rep_samples
