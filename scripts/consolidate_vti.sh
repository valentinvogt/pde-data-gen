#!/bin/bash
#SBATCH --job-name=cons
#SBATCH --output=cons-%j.out
#SBATCH --error=cons-%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=12:00:00

module load stack/2024-06 python/3.11.6

source .env
model="gray_scott"
dataset_id="gs_single"
DATAPATH="$SCRATCHDIR/data/$model/$dataset_id"

for dir in $DATAPATH/out-*; do
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
    python scripts/gather_vti.py $dir --input-filename "$DATAPATH/$dir_name.vti" --output $output_file
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    echo "$dir_name took $duration seconds"
    # rm $dir/*.vti
done

./scripts/merge.sh $DATAPATH

python3 src/classify.py --model $model --ds_id $dataset_id --time_ratio 0.2 --directory_var SCRATCHDIR --mode new
