#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

HDF5_USE_FILE_LOCKING=FALSE

# python3 src/classify.py --model "bruss" --ds_id "param_sweep" --time_ratio 0.2
python3 scripts/process_dataset_for_training.py /cluster/work/math/vogtva/data/bruss/ball_big/_dataset.nc