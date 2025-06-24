#!/bin/bash
#SBATCH --job-name=cons
#SBATCH --output=cons-%j.out
#SBATCH --error=cons-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=1:00:00

module load stack/2024-06 python/3.11.6

source .env
model="bruss"
dataset_id="final_2"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

for dir in $DATAPATH/out-*; do
    
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
    python scripts/gather_vti.py --path $dir --output $output_file
    # rm $dir/*.vti
done