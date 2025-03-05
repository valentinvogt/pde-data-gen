#!/bin/bash
#SBATCH --job-name=bruss-dist
#SBATCH --output=bruss-dist-%j.out
#SBATCH --error=bruss-dist-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=04:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

DATAPATH="/cluster/scratch/vogtva/data"

A=5
B=9
Nx=200
dx=1.0
Nt=10_000
dt=0.0025
Du=2.0
Dv=22.0
n_snapshots=100
model="bruss"
run_id="dist2"

mkdir -p $DATAPATH/$model

for seed in $(seq 1 50); do
        for eps_u in 0.01 0.05 0.1 0.5; do
                for eps_v in 0.01 0.05 0.1 0.5; do
                    FILENAME="${DATAPATH}/${model}/$(uuidgen).nc"
                    echo $FILENAME
                    FILE=$(python3 scripts/rd_runner.py --model $model --A $A --B $B \
                            --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv --sigma_ic_u $eps_u \
                            --sigma_ic_v $eps_v --random_seed $seed \
                            --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id)
                    build/run_from_netcdf $FILE 1
                done
        done
done

# Sanitization
for file in ${DATAPATH}/*; do
        mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
done
