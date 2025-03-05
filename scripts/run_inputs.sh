#!/bin/bash
#SBATCH --job-name=ball
#SBATCH --output=ball-%j.out
#SBATCH --error=ball-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=4096
#SBATCH --time=8:00:00

module load stack/2024-06
module load gcc/12.2.0
module load cmake/3.27.7
module load cuda/12.1.1
module load hdf5/1.14.3
module load openmpi/4.1.6
module load netcdf-c/4.9.2
module load python/3.11.6

DATAPATH="/cluster/scratch/vogtva/data"

# ADAPT THESE
model="bruss"
run_id="test"

run_dir="$DATAPATH/$model/$run_id"
echo $run_dir
# Check if we're using the consolidated file approach
if [[ -f "$run_dir/_dataset.nc" ]]; then
    echo "Using consolidated output approach"
    
    # Process individual input files but track their outputs for later consolidation
    output_files=()
    
    for file in "$run_dir"/*.nc; do
        # Skip files that aren't input files
        if [[ "$file" == *_output.nc || "$file" == *_dataset.nc ]]; then
            continue
        fi
        
        echo "Processing $file"
        build/run_from_netcdf "$file" 1
        
        # Add the output file to our list
        output_file="${file%.nc}_output.nc"
        if [[ -f "$output_file" ]]; then
            output_files+=("$output_file")
        fi
    done
    python scripts/consolidate_outputs.py $run_dir/_dataset.nc
    
else
    # Original approach - process each file individually
    echo "Using original individual files approach"
    
    for file in "$run_dir"/*.nc; do
        # Skip output files
        if [[ "$file" == *_output.nc ]]; then
            continue
        fi
        
        build/run_from_netcdf "$file" 1
    done
fi