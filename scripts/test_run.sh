#!/bin/bash
#SBATCH --job-name=test-runner
#SBATCH --output=test-runner-%j.out
#SBATCH --error=test-runner-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

#SBATCH --gpus=rtx_3090:1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=0:30:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

build/run_from_netcdf /cluster/scratch/vogtva/data/bruss/test/13667a56-a2a3-4145-a3ae-f835d24727f7.nc 1
