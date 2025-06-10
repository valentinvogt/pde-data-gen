#!/bin/bash
#SBATCH --job-name=input-runner-gs
#SBATCH --output=input-runner-gs-%j.out
#SBATCH --error=input-runner-gs-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=12:00:00
#SBATCH --mail-type=END

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

source .env

# ADAPT THESE
model="gray_scott"
dataset_id="gs_rough"

DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"
echo $DATAPATH
# Check if we're using the consolidated file approach
if [[ -f "$DATAPATH/_dataset.nc" ]]; then
    echo "Using consolidated output approach"

    for file in "$DATAPATH"/*.nc; do
        # Skip files that aren't input files
        if [[ "$file" == *_output.nc || "$file" == *_dataset.nc ]]; then
            continue
        fi

        # echo "Processing $file"
        build/run_from_netcdf "$file" 1
    done
    python scripts/consolidate_outputs.py $DATAPATH/_dataset.nc

else
    # Original approach - process each file individually
    echo "Using original individual files approach"

    for file in "$DATAPATH"/*.nc; do
        # Skip output files
        if [[ "$file" == *_output.nc ]]; then
            continue
        fi

        build/run_from_netcdf "$file" 1
    done
    python3 scripts/consolidate_old_format.py $DATAPATH
fi
