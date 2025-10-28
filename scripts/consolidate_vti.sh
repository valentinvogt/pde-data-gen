#!/bin/bash
#SBATCH --job-name=cons-vti
#SBATCH --output=cons-vti-%j.out
#SBATCH --error=cons-vti-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=1:00:00

source .env
if [ "$ON_SLURM" = true ] ; then
    module load stack/2024-06 python/3.11.6
fi
set -eo pipefail

# Accept command-line arguments with defaults
# Usage: ./consolidate_vti.sh [model] [dataset_id]
# Or with sbatch: sbatch scripts/consolidate_vti.sh gs my_dataset
model="${1:-bruss}"
dataset_id="${2:-default_vti}"
DATAPATH="$WORK_DIR/$model/$dataset_id"

for dir in "$DATAPATH"/out-*; do
    start=$(date +%s.%N)
    # get the last part of the directory name after the out-
    dir_name=$(basename "$dir")
    dir_name=${dir_name#out-}
    
    # Check if the output file already exists
    output_file="$DATAPATH/$dir_name.nc"
    if [ -f "$output_file" ]; then
        echo "Output file $output_file already exists. Skipping directory: $dir_name"
        continue
    fi
    echo "Processing directory: $dir_name"
    python scripts/gather_vti.py "$dir" --input-filename "$DATAPATH/$dir_name.vti" --output "$output_file"
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$dir_name took $duration seconds"
    rm "$dir"/*.vti
done

./scripts/merge_nc_trajectories.sh "$DATAPATH"

# Inject metadata from JSON files into the merged dataset
python scripts/inject_metadata.py "$DATAPATH/_dataset.nc"
