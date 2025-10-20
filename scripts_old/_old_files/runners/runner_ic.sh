#!/bin/bash
#SBATCH --job-name=vary-sigma
#SBATCH --output=vary-sigma-%j.out
#SBATCH --error=vary-sigma-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=01:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

# A: float = 5
# B: float = 9
# Nx: int = 100
# dx: float = 1.0
# Nt: int = 1000
# dt: float = 0.01
# Du: float = 2.0
# Dv: float = 22.0
# n_snapshots: int = 100

DATAPATH="/cluster/scratch/vogtva/data/vary-seed"

A=5
B=9
Nx=50
dx=1.0
Nt=1_000
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=100

for random_seed in {0..9}; do
        FILENAME="${DATAPATH}/bruss_seed_${random_seed}.nc"

        FILE=$(python3 scripts/rd_runner.py --model bruss --A $A --B $B \
                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv --random_seed $random_seed \
                --n_snapshots $n_snapshots --filename $FILENAME)

        build/run_from_netcdf $FILE 1
done

# Sanitization
for file in ${DATAPATH}/*; do
        mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
done
