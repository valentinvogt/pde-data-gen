#!/bin/bash
#SBATCH --job-name=input-runner
#SBATCH --output=input-runner-%j.out
#SBATCH --error=input-runner-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=8192
#SBATCH --time=3:00:00
#SBATCH --mail-type=END

source .env
set -eo pipefail
USE_CUDA=0
VERBOSE=0

if [ "$ON_SLURM" = true ] ; then
    module load stack/2024-06
    module load gcc/12.2.0
    module load cmake/3.27.7
    module load cuda/12.1.1
    module load hdf5/1.14.3
    module load openmpi/4.1.6
    module load netcdf-c/4.9.2
    module load python/3.11.6
    USE_CUDA=1
fi

# Accept command-line arguments with defaults
# Usage: ./run_from_nc.sh [model] [dataset_id]
# Or with sbatch: sbatch scripts/run_from_nc.sh model my_dataset
model="${1:-bruss}"
dataset_id="${2:-default_bruss}"

DATAPATH="$WORK_DIR/$model/$dataset_id"

for file in "$DATAPATH"/*.nc; do
    # Skip files that aren't input files
    if [[ "$file" == *_output.nc || "$file" == *_dataset.nc ]]; then
        continue
    fi

    # echo "Processing $file"
    build/run_from_netcdf "$file" $USE_CUDA $VERBOSE
done
echo "All simulations for dataset '$dataset_id' completed."

# Consolidate outputs and inject metadata
$PYTHON_CMD scripts/consolidate_nc.py "$DATAPATH"
