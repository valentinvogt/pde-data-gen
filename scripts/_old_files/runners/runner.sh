#!/bin/bash
#SBATCH --job-name=abd-big
#SBATCH --output=abd-big-%j.out
#SBATCH --error=abd-big-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=20:00:00
#SBATCH --mail-type=END

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

Nx=128
dx=1.0
Nt=40_000
dt=0.0025
n_snapshots=100
model="bruss"
run_id="ball_prelim"

mkdir -p $DATAPATH/$model/$run_id

A_vals=(0.5 1.5 4.5)
B_mults=(1.5 2 2.5)
Du_vals=(0.5 1 2)
Dv_mults=(11.0 15.0 19.0)

for A in "${A_vals[@]}"; do
        for B_mult in "${B_mults[@]}"; do
                for Du in "${Du_vals[@]}"; do
                        for D_mult in "${Dv_mults[@]}"; do
                                for seed in $(seq 1 2); do
                                        start=`date +%s`
                                        B=$(python -c "print($A * $B_mult)")
                                        Dv=$(python -c "print($Du * $D_mult)")
                                        FILENAME="${DATAPATH}/${model}/${run_id}/$(uuidgen).nc"
                                        echo "(A, B, Du, Dv) = ($A, $B, $Du, $Dv)"
                                        FILE=$(python scripts/rd_runner.py --model $model --A $A --B $B \
                                                --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
                                                --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id \
                                                --random_seed $seed)
                                        build/run_from_netcdf $FILE 1
                                        end=`date +%s`
                                        runtime=$((end-start))
                                        echo "Took $runtime seconds"
                                done
                        done
                done
        done
done

# # Sanitization
for file in ${DATAPATH}/*; do
        mv "$file" "$(echo "$file" | sed 's/\xA0//g')"
done
