#!/bin/bash
#SBATCH --job-name=make-inp
#SBATCH --output=make-inp-%j.out
#SBATCH --error=make-inp-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=01:00:00

module load stack/2024-06
module load gcc/12.2.0
# module load cmake/3.27.7
# module load cuda/12.1.1
# module load hdf5/1.14.3
# module load openmpi/4.1.6
# module load netcdf-c/4.9.2
module load python/3.11.6

HDF5_USE_FILE_LOCKING=FALSE
set -a && source .env

for DT in 0.0025 0.0015 0.001 0.0005; do
    NT=$(echo "150 / $DT" | bc)
    echo "Generating inputs for dt=$DT, Nt=$NT"
    python3 scripts/create_inputs.py --config-name=redo 'sim_params={Nt:'$NT', dt:'$DT'}'
done