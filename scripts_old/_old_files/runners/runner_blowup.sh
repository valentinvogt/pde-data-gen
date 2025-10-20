#!/bin/bash
#SBATCH --job-name=vary-both
#SBATCH --output=vary-both-%j.out
#SBATCH --error=vary-both-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
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

# A: float = 5
# B: float = 9
# Nx: int = 100
# dx: float = 1.0
# Nt: int = 1000
# dt: float = 0.01
# Du: float = 2.0
# Dv: float = 22.0
# n_snapshots: int = 100

DATAPATH="/cluster/scratch/vogtva/data"

# A=5
# B=9
Nx=128
dx=1.0
Nt=5_000
dt=0.0025
Du=1.0
Dv=18.0
n_snapshots=500
model="bruss"
run_id="blowup"
mkdir -p $DATAPATH/$model/$run_id

for A in 9 10 11; do
        for B_mult in 3 3.5 4 4.5 5 5.5; do
                start=`date +%s`
                B=$(python -c "print($A * $B_mult)")
                FILENAME="${DATAPATH}/${model}/${run_id}/$(uuidgen).nc"
                echo "(A, B) = ($A, $B)"
                FILE=$(python scripts/rd_runner.py --model $model --A $A --B $B \
                        --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                        --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id)
                build/run_from_netcdf $FILE 1
                end=`date +%s`
                runtime=$((end-start))
                echo "Took $runtime seconds"
        done
done

# Sanitization
# for file in ${DATAPATH}/*; do
#         mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
# done
