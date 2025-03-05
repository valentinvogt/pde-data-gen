#!/bin/bash
#SBATCH --job-name=bruss-ml
#SBATCH --output=bruss-ml-%j.out
#SBATCH --error=bruss-ml-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=48:00:00
#SBATCH --mail-type=END

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

DATAPATH="/cluster/scratch/vogtva/data"
Nx=128
dx=1.0
Nt=40000
dt=0.0025
n_snapshots=100
model="bruss"
run_id="phase_transition_dataset"

# mkdir -p $DATAPATH/$model/$run_id

# Parameters for steady-state regime (B/A < 1.8) ~15% of dataset
A_steady=(0.5 1.0 1.5 2.0)
B_A_steady=(1.0 1.2 1.4 1.6)
seeds_steady=5  # Fewer runs in this regime

# Parameters for critical region (1.8 <= B/A <= 2.2) ~45% of dataset
A_critical=(0.5 1.0 1.5 2.0)
B_A_critical=(1.8 1.9 2.0 2.1 2.2)
seeds_critical=15  # More runs near the phase transition

# Parameters for oscillatory regime (B/A > 2.2) ~40% of dataset
A_oscillatory=(0.5 1.0 1.5 2.0)
B_A_oscillatory=(2.3 2.6 3.0 3.5 4.0 4.5 5.0)
seeds_oscillatory=10  # More runs in oscillatory regime

# Diffusion coefficients
Du_vals=(0.001 0.005 0.01 0.05 0.1)
Dv_Du_ratios=(0.5 1.0 2.0 5.0 10.0)

# Initial condition types (1=uniform, 2=random, 3=gaussian, 4=sinusoidal)
ic_types=(1 2 3 4)

# Counter for total trajectories
total=0

# Function to run simulation with parameters
run_simulation() {
    # local A=$1
    # local B=$2
    # local Du=$3
    # local Dv=$4
    # local seed=$5
    # local ic_type=$6
    
    # start=`date +%s`
    # FILENAME="${DATAPATH}/${model}/${run_id}/$(uuidgen).nc"
    # echo "Parameters: (A, B, Du, Dv, B/A) = ($A, $B, $Du, $Dv, $(python -c "print($B/$A)"))"
    
    # FILE=$(python scripts/rd_runner.py --model $model --A $A --B $B \
    # --Nx $Nx --dx $dx --Nt $Nt --dt $dt --Du $Du --Dv $Dv \
    # --n_snapshots $n_snapshots --filename $FILENAME --run_id=$run_id \
    # --random_seed $seed --ic_type $ic_type)
    
    # build/run_from_netcdf $FILE 1
    
    # end=`date +%s`
    # runtime=$((end-start))
    # echo "Took $runtime seconds"
    
    # # Increment counter
    total=$((total+1))
    # echo "Total trajectories generated: $total"
}

# 1. Steady-state regime runs
echo "GENERATING STEADY-STATE REGIME DATA"
for A in "${A_steady[@]}"; do
    for B_A in "${B_A_steady[@]}"; do
        B=$(python -c "print($A * $B_A)")
        for Du in "${Du_vals[@]}"; do
            for ratio in "${Dv_Du_ratios[@]}"; do
                Dv=$(python -c "print($Du * $ratio)")
                for seed in $(seq 1 $seeds_steady); do
                    for ic_type in "${ic_types[@]}"; do
                        if [ $((RANDOM % 4)) -eq 0 ]; then  # Random subsampling to control quantity
                            run_simulation $A $B $Du $Dv $seed $ic_type
                        fi
                    done
                done
            done
        done
    done
done

# 2. Critical region runs
echo "GENERATING CRITICAL REGION DATA"
for A in "${A_critical[@]}"; do
    for B_A in "${B_A_critical[@]}"; do
        B=$(python -c "print($A * $B_A)")
        for Du in "${Du_vals[@]}"; do
            for ratio in "${Dv_Du_ratios[@]}"; do
                Dv=$(python -c "print($Du * $ratio)")
                for seed in $(seq 1 $seeds_critical); do
                    for ic_type in "${ic_types[@]}"; do
                        run_simulation $A $B $Du $Dv $seed $ic_type
                    done
                done
            done
        done
    done
done

# 3. Oscillatory regime runs
echo "GENERATING OSCILLATORY REGIME DATA"
for A in "${A_oscillatory[@]}"; do
    for B_A in "${B_A_oscillatory[@]}"; do
        B=$(python -c "print($A * $B_A)")
        for Du in "${Du_vals[@]}"; do
            for ratio in "${Dv_Du_ratios[@]}"; do
                Dv=$(python -c "print($Du * $ratio)")
                for seed in $(seq 1 $seeds_oscillatory); do
                    for ic_type in "${ic_types[@]}"; do
                        if [ $((RANDOM % 2)) -eq 0 ]; then  # Random subsampling to control quantity
                            run_simulation $A $B $Du $Dv $seed $ic_type
                        fi
                    done
                done
            done
        done
    done
done

echo "DATASET GENERATION COMPLETE. Total trajectories: $total"